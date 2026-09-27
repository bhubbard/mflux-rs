use mlx_rs::module::Module;
use mlx_rs::nn::Linear;
use mlx_rs::Array;
use crate::config::ModelConfig;
use crate::error::Result;

pub struct TimestepEmbedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new(256, dim as i32)?,
            linear_2: Linear::new(dim as i32, dim as i32)?,
        })
    }

    pub fn forward(&mut self, sample: &Array) -> Result<Array> {
        let x = self.linear_1.forward(sample)?;
        let x = mlx_rs::nn::silu(&x)?;
        Ok(self.linear_2.forward(&x)?)
    }
}

pub struct GuidanceEmbedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl GuidanceEmbedder {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new(256, dim as i32)?,
            linear_2: Linear::new(dim as i32, dim as i32)?,
        })
    }

    pub fn forward(&mut self, sample: &Array) -> Result<Array> {
        let x = self.linear_1.forward(sample)?;
        let x = mlx_rs::nn::silu(&x)?;
        Ok(self.linear_2.forward(&x)?)
    }
}

pub struct TextEmbedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl TextEmbedder {
    pub fn new(in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::new(in_dim as i32, out_dim as i32)?,
            linear_2: Linear::new(out_dim as i32, out_dim as i32)?,
        })
    }

    pub fn forward(&mut self, caption: &Array) -> Result<Array> {
        let x = self.linear_1.forward(caption)?;
        let x = mlx_rs::nn::silu(&x)?;
        Ok(self.linear_2.forward(&x)?)
    }
}

pub struct TimeTextEmbed {
    pub text_embedder: TextEmbedder,
    pub guidance_embedder: Option<GuidanceEmbedder>,
    pub timestep_embedder: TimestepEmbedder,
}

impl TimeTextEmbed {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        let text_embedder = TextEmbedder::new(config.pooled_projection_dim, config.hidden_size)?;
        let guidance_embedder = if config.guidance_embed {
            Some(GuidanceEmbedder::new(config.hidden_size)?)
        } else {
            None
        };
        let timestep_embedder = TimestepEmbedder::new(config.hidden_size)?;

        Ok(Self {
            text_embedder,
            guidance_embedder,
            timestep_embedder,
        })
    }

    /// Sinusoidal time embedding projection:
    /// max_period = 10000, half_dim = 128
    pub fn time_proj(time_steps: &Array) -> Result<Array> {
        let max_period = 10000.0f32;
        let half_dim = 128usize;

        let mut exponent = Vec::with_capacity(half_dim);
        let log_max_period = max_period.ln();
        for i in 0..half_dim {
            exponent.push(-log_max_period * (i as f32) / (half_dim as f32));
        }

        let total = time_steps.size();
        let flat_steps = time_steps.reshape(&[total as i32])?;
        let steps_slice = flat_steps.as_slice::<f32>();

        // For each step, create [sin, cos] flipped as [cos, sin]
        let mut out = Vec::with_capacity(total * half_dim * 2);
        for &t in steps_slice {
            let mut sin_part = Vec::with_capacity(half_dim);
            let mut cos_part = Vec::with_capacity(half_dim);

            for exp in &exponent {
                let emb = t * exp.exp();
                sin_part.push(emb.sin());
                cos_part.push(emb.cos());
            }

            out.extend_from_slice(&cos_part);
            out.extend_from_slice(&sin_part);
        }

        Ok(Array::from_slice(&out, &[total as i32, (half_dim * 2) as i32]))
    }

    pub fn forward(
        &mut self,
        time_step: &Array,
        pooled_projection: &Array,
        guidance: Option<&Array>,
    ) -> Result<Array> {
        let time_proj = Self::time_proj(time_step)?;
        let mut time_steps_emb = self.timestep_embedder.forward(&time_proj)?;

        if let (Some(ref mut ge), Some(g)) = (&mut self.guidance_embedder, guidance) {
            let g_proj = Self::time_proj(g)?;
            let g_emb = ge.forward(&g_proj)?;
            time_steps_emb = time_steps_emb.add(&g_emb)?;
        }

        let pooled_emb = self.text_embedder.forward(pooled_projection)?;
        Ok(time_steps_emb.add(&pooled_emb)?)
    }
}
