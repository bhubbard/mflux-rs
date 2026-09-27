use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use crate::error::Result;

pub struct AdaLayerNormZero {
    pub linear: Linear,
    pub dim: usize,
}

pub struct AdaNormZeroOutput {
    pub hidden_states: Array,
    pub gate_msa: Array,
    pub shift_mlp: Array,
    pub scale_mlp: Array,
    pub gate_mlp: Array,
}

impl AdaLayerNormZero {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(dim as i32, (6 * dim) as i32)?,
            dim,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        text_embeddings: &Array,
    ) -> Result<AdaNormZeroOutput> {
        let silu_text = mlx_rs::nn::silu(text_embeddings)?;
        let emb = self.linear.forward(&silu_text)?;

        let chunk = self.dim as i32;
        let shift_msa = emb.index((.., 0..chunk));
        let scale_msa = emb.index((.., chunk..2 * chunk));
        let gate_msa = emb.index((.., 2 * chunk..3 * chunk));
        let shift_mlp = emb.index((.., 3 * chunk..4 * chunk));
        let scale_mlp = emb.index((.., 4 * chunk..5 * chunk));
        let gate_mlp = emb.index((.., 5 * chunk..6 * chunk));

        // LayerNorm affine=False
        let norm_hidden = layer_norm_no_affine(hidden_states, 1e-6)?;

        // hidden_states = norm(hidden) * (1 + scale_msa) + shift_msa
        let one = Array::from_f32(1.0f32);
        let scale_factor = one.add(&scale_msa)?;
        let scaled = norm_hidden.multiply(&scale_factor)?;
        let normalized = scaled.add(&shift_msa)?;

        Ok(AdaNormZeroOutput {
            hidden_states: normalized,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        })
    }
}

pub struct AdaLayerNormZeroSingle {
    pub linear: Linear,
    pub dim: usize,
}

pub struct AdaNormSingleOutput {
    pub hidden_states: Array,
    pub gate_msa: Array,
}

impl AdaLayerNormZeroSingle {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(dim as i32, (3 * dim) as i32)?,
            dim,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        text_embeddings: &Array,
    ) -> Result<AdaNormSingleOutput> {
        let silu_text = mlx_rs::nn::silu(text_embeddings)?;
        let emb = self.linear.forward(&silu_text)?;

        let chunk = self.dim as i32;
        let shift_msa = emb.index((.., 0..chunk));
        let scale_msa = emb.index((.., chunk..2 * chunk));
        let gate_msa = emb.index((.., 2 * chunk..3 * chunk));

        let norm_hidden = layer_norm_no_affine(hidden_states, 1e-6)?;
        let one = Array::from_f32(1.0f32);
        let scale_factor = one.add(&scale_msa)?;
        let scaled = norm_hidden.multiply(&scale_factor)?;
        let normalized = scaled.add(&shift_msa)?;

        Ok(AdaNormSingleOutput {
            hidden_states: normalized,
            gate_msa,
        })
    }
}

pub struct AdaLayerNormContinuous {
    pub linear: Linear,
    pub dim: usize,
}

impl AdaLayerNormContinuous {
    pub fn new(embedding_dim: usize, conditioning_dim: usize) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(conditioning_dim as i32, (2 * embedding_dim) as i32)?,
            dim: embedding_dim,
        })
    }

    pub fn forward(&mut self, x: &Array, text_embeddings: &Array) -> Result<Array> {
        let silu_text = mlx_rs::nn::silu(text_embeddings)?;
        let emb = self.linear.forward(&silu_text)?;

        let chunk = self.dim as i32;
        let scale = emb.index((.., 0..chunk));
        let shift = emb.index((.., chunk..2 * chunk));

        let norm_x = layer_norm_no_affine(x, 1e-6)?;
        let one = Array::from_f32(1.0f32);
        let scale_factor = one.add(&scale)?;
        let scaled = norm_x.multiply(&scale_factor)?;
        Ok(scaled.add(&shift)?)
    }
}

/// Standard LayerNorm across last dimension without affine parameters
pub fn layer_norm_no_affine(x: &Array, eps: f32) -> Result<Array> {
    let mean = x.mean_axis(-1, true)?;
    let diff = x.subtract(&mean)?;
    let diff_sq = diff.square()?;
    let var = diff_sq.mean_axis(-1, true)?;
    let eps_arr = Array::from_f32(eps);
    let var_eps = var.add(&eps_arr)?;
    let std = var_eps.sqrt()?;
    Ok(diff.divide(&std)?)
}
