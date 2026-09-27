use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate, softmax_axis, transpose_axes};
use mlx_rs::Array;
use crate::error::Result;
use crate::model::rope::EmbedND;

pub struct JointAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub to_out: Linear,
    pub add_q_proj: Linear,
    pub add_k_proj: Linear,
    pub add_v_proj: Linear,
    pub to_add_out: Linear,
}

impl JointAttention {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let dim = (num_heads * head_dim) as i32;
        Ok(Self {
            num_heads,
            head_dim,
            to_q: Linear::new(hidden_size as i32, dim)?,
            to_k: Linear::new(hidden_size as i32, dim)?,
            to_v: Linear::new(hidden_size as i32, dim)?,
            to_out: Linear::new(dim, hidden_size as i32)?,
            add_q_proj: Linear::new(hidden_size as i32, dim)?,
            add_k_proj: Linear::new(hidden_size as i32, dim)?,
            add_v_proj: Linear::new(hidden_size as i32, dim)?,
            to_add_out: Linear::new(dim, hidden_size as i32)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Array, Array)> {
        let img_seq_len = hidden_states.shape()[1] as usize;
        let txt_seq_len = encoder_hidden_states.shape()[1] as usize;
        let joint_seq_len = txt_seq_len + img_seq_len;

        // Image QKV
        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        // Text QKV
        let enc_q = self.add_q_proj.forward(encoder_hidden_states)?;
        let enc_k = self.add_k_proj.forward(encoder_hidden_states)?;
        let enc_v = self.add_v_proj.forward(encoder_hidden_states)?;

        // Concatenate text + image along sequence dimension (axis 1)
        let full_q = concatenate(&[&enc_q, &q], 1)?;
        let full_k = concatenate(&[&enc_k, &k], 1)?;
        let full_v = concatenate(&[&enc_v, &v], 1)?;

        // Reshape [1, joint_seq_len, num_heads, head_dim] -> [1, num_heads, joint_seq_len, head_dim]
        let q_shaped = full_q.reshape(&[1, joint_seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let q_heads = transpose_axes(&q_shaped, &[0, 2, 1, 3])?;

        let k_shaped = full_k.reshape(&[1, joint_seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let k_heads = transpose_axes(&k_shaped, &[0, 2, 1, 3])?;

        let v_shaped = full_v.reshape(&[1, joint_seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let v_heads = transpose_axes(&v_shaped, &[0, 2, 1, 3])?;

        // Apply RoPE to full Q and K
        let q_rot = EmbedND::apply_rotary_emb(&q_heads, cos, sin, joint_seq_len, self.head_dim)?;
        let k_rot = EmbedND::apply_rotary_emb(&k_heads, cos, sin, joint_seq_len, self.head_dim)?;

        // Scaled dot product attention
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&k_rot, &[0, 1, 3, 2])?;
        let scores = q_rot.matmul(&k_t)?.multiply(&scale_arr)?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let context = attn_weights.matmul(&v_heads)?;

        // Reshape back [1, num_heads, joint_seq_len, head_dim] -> [1, joint_seq_len, num_heads * head_dim]
        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        let full_context = context_t.reshape(&[1, joint_seq_len as i32, (self.num_heads * self.head_dim) as i32])?;

        // Split text and image using IndexOp
        let enc_out = full_context.index((.., 0..txt_seq_len as i32, ..));
        let img_out = full_context.index((.., txt_seq_len as i32..joint_seq_len as i32, ..));

        let img_proj = self.to_out.forward(&img_out)?;
        let enc_proj = self.to_add_out.forward(&enc_out)?;

        Ok((img_proj, enc_proj))
    }
}

pub struct SingleBlockAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
}

impl SingleBlockAttention {
    pub fn new(hidden_size: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        let dim = (num_heads * head_dim) as i32;
        Ok(Self {
            num_heads,
            head_dim,
            to_q: Linear::new(hidden_size as i32, dim)?,
            to_k: Linear::new(hidden_size as i32, dim)?,
            to_v: Linear::new(hidden_size as i32, dim)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Array> {
        let seq_len = hidden_states.shape()[1] as usize;

        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(hidden_states)?;
        let v = self.to_v.forward(hidden_states)?;

        let q_shaped = q.reshape(&[1, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let q_heads = transpose_axes(&q_shaped, &[0, 2, 1, 3])?;

        let k_shaped = k.reshape(&[1, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let k_heads = transpose_axes(&k_shaped, &[0, 2, 1, 3])?;

        let v_shaped = v.reshape(&[1, seq_len as i32, self.num_heads as i32, self.head_dim as i32])?;
        let v_heads = transpose_axes(&v_shaped, &[0, 2, 1, 3])?;

        let q_rot = EmbedND::apply_rotary_emb(&q_heads, cos, sin, seq_len, self.head_dim)?;
        let k_rot = EmbedND::apply_rotary_emb(&k_heads, cos, sin, seq_len, self.head_dim)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_arr = Array::from_f32(scale);
        let k_t = transpose_axes(&k_rot, &[0, 1, 3, 2])?;
        let scores = q_rot.matmul(&k_t)?.multiply(&scale_arr)?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let context = attn_weights.matmul(&v_heads)?;

        let context_t = transpose_axes(&context, &[0, 2, 1, 3])?;
        Ok(context_t.reshape(&[1, seq_len as i32, (self.num_heads * self.head_dim) as i32])?)
    }
}
