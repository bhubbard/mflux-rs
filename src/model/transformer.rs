use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::concatenate;
use mlx_rs::Array;
use crate::config::ModelConfig;
use crate::error::Result;
use crate::model::attention::{JointAttention, SingleBlockAttention};
use crate::model::embeddings::TimeTextEmbed;
use crate::model::norm::{AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle};

pub struct FeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
    pub approx: bool,
}

impl FeedForward {
    pub fn new(dim: usize, hidden_dim: usize, approx: bool) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(dim as i32, hidden_dim as i32)?,
            linear2: Linear::new(hidden_dim as i32, dim as i32)?,
            approx,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.linear1.forward(x)?;
        let act = if self.approx {
            mlx_rs::nn::gelu_approximate(&h)?
        } else {
            mlx_rs::nn::gelu(&h)?
        };
        Ok(self.linear2.forward(&act)?)
    }
}

pub struct JointTransformerBlock {
    pub norm1: AdaLayerNormZero,
    pub norm1_context: AdaLayerNormZero,
    pub attn: JointAttention,
    pub ff: FeedForward,
    pub ff_context: FeedForward,
}

impl JointTransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            norm1: AdaLayerNormZero::new(hidden_size)?,
            norm1_context: AdaLayerNormZero::new(hidden_size)?,
            attn: JointAttention::new(hidden_size, num_heads, head_dim)?,
            ff: FeedForward::new(hidden_size, 4 * hidden_size, false)?,
            ff_context: FeedForward::new(hidden_size, 4 * hidden_size, true)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        text_embeddings: &Array,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Array, Array)> {
        // 1. Norm
        let norm_img = self.norm1.forward(hidden_states, text_embeddings)?;
        let norm_txt = self.norm1_context.forward(encoder_hidden_states, text_embeddings)?;

        // 2. Attention
        let (attn_img, attn_txt) = self.attn.forward(
            &norm_img.hidden_states,
            &norm_txt.hidden_states,
            cos,
            sin,
        )?;

        // 3. Image residual + FF
        let gated_attn_img = attn_img.multiply(&norm_img.gate_msa)?;
        let res_img = hidden_states.add(&gated_attn_img)?;
        let ff_img = self.ff.forward(&res_img)?;
        let gated_ff_img = ff_img.multiply(&norm_img.gate_mlp)?;
        let next_hidden = res_img.add(&gated_ff_img)?;

        // 4. Text residual + FF
        let gated_attn_txt = attn_txt.multiply(&norm_txt.gate_msa)?;
        let res_txt = encoder_hidden_states.add(&gated_attn_txt)?;
        let ff_txt = self.ff_context.forward(&res_txt)?;
        let gated_ff_txt = ff_txt.multiply(&norm_txt.gate_mlp)?;
        let next_encoder = res_txt.add(&gated_ff_txt)?;

        Ok((next_encoder, next_hidden))
    }
}

pub struct SingleTransformerBlock {
    pub norm: AdaLayerNormZeroSingle,
    pub attn: SingleBlockAttention,
    pub proj_mlp: Linear,
    pub proj_out: Linear,
}

impl SingleTransformerBlock {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let mlp_dim = (4 * hidden_size) as i32;
        Ok(Self {
            norm: AdaLayerNormZeroSingle::new(hidden_size)?,
            attn: SingleBlockAttention::new(hidden_size, num_heads, head_dim)?,
            proj_mlp: Linear::new(hidden_size as i32, mlp_dim)?,
            proj_out: Linear::new((hidden_size as i32) + mlp_dim, hidden_size as i32)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        text_embeddings: &Array,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Array> {
        let norm_res = self.norm.forward(hidden_states, text_embeddings)?;

        let attn_out = self.attn.forward(&norm_res.hidden_states, cos, sin)?;
        let mlp_out = self.proj_mlp.forward(&norm_res.hidden_states)?;
        let mlp_act = mlx_rs::nn::gelu_approximate(&mlp_out)?;

        let concat = concatenate(&[&attn_out, &mlp_act], 2)?;
        let proj = self.proj_out.forward(&concat)?;
        let gated = proj.multiply(&norm_res.gate_msa)?;

        Ok(hidden_states.add(&gated)?)
    }
}

pub struct FluxTransformer {
    pub config: ModelConfig,
    pub x_embedder: Linear,
    pub context_embedder: Linear,
    pub time_text_embed: TimeTextEmbed,
    pub joint_blocks: Vec<JointTransformerBlock>,
    pub single_blocks: Vec<SingleTransformerBlock>,
    pub norm_out: AdaLayerNormContinuous,
    pub proj_out: Linear,
}

impl FluxTransformer {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let x_in = config.x_embedder_input_dim();
        let heads = config.num_heads;
        let head_dim = config.head_dim;

        let x_embedder = Linear::new(x_in as i32, hidden as i32)?;
        let context_embedder = Linear::new(config.joint_attention_dim as i32, hidden as i32)?;
        let time_text_embed = TimeTextEmbed::new(&config)?;

        let mut joint_blocks = Vec::with_capacity(config.num_transformer_blocks);
        for _ in 0..config.num_transformer_blocks {
            joint_blocks.push(JointTransformerBlock::new(hidden, heads, head_dim)?);
        }

        let mut single_blocks = Vec::with_capacity(config.num_single_transformer_blocks);
        for _ in 0..config.num_single_transformer_blocks {
            single_blocks.push(SingleTransformerBlock::new(hidden, heads, head_dim)?);
        }

        let norm_out = AdaLayerNormContinuous::new(hidden, hidden)?;
        let proj_out = Linear::new(hidden as i32, config.out_channels as i32)?;

        Ok(Self {
            config,
            x_embedder,
            context_embedder,
            time_text_embed,
            joint_blocks,
            single_blocks,
            norm_out,
            proj_out,
        })
    }

    pub fn forward(
        &mut self,
        t_step: &Array,
        latents: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        guidance: Option<&Array>,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Array> {
        // 1. Embeddings
        let mut hidden_states = self.x_embedder.forward(latents)?;
        let mut encoder_hidden_states = self.context_embedder.forward(prompt_embeds)?;
        let text_embeddings = self.time_text_embed.forward(t_step, pooled_prompt_embeds, guidance)?;

        let txt_len = encoder_hidden_states.shape()[1] as usize;

        // 2. Joint transformer blocks
        for block in &mut self.joint_blocks {
            let (next_enc, next_img) = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &text_embeddings,
                cos,
                sin,
            )?;
            encoder_hidden_states = next_enc;
            hidden_states = next_img;
        }

        // 3. Concatenate text + image states
        let mut combined = concatenate(&[&encoder_hidden_states, &hidden_states], 1)?;

        // 4. Single transformer blocks
        for block in &mut self.single_blocks {
            combined = block.forward(&combined, &text_embeddings, cos, sin)?;
        }

        // 5. Slice image portion and project out using IndexOp
        let total_len = combined.shape()[1] as i32;
        let img_slice = combined.index((.., txt_len as i32..total_len, ..));
        let normed = self.norm_out.forward(&img_slice, &text_embeddings)?;
        Ok(self.proj_out.forward(&normed)?)
    }
}
