use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluxVariant {
    Schnell,
    Dev,
}

impl Default for FluxVariant {
    fn default() -> Self {
        Self::Schnell
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub variant: FluxVariant,
    pub num_transformer_blocks: usize,
    pub num_single_transformer_blocks: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub joint_attention_dim: usize,
    pub pooled_projection_dim: usize,
    pub guidance_embed: bool,
    pub theta: f32,
    pub axes_dim: [usize; 3],
    pub requires_sigma_shift: bool,
}

impl ModelConfig {
    pub fn schnell() -> Self {
        Self {
            variant: FluxVariant::Schnell,
            num_transformer_blocks: 19,
            num_single_transformer_blocks: 38,
            in_channels: 64,
            out_channels: 64,
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            joint_attention_dim: 4096,
            pooled_projection_dim: 768,
            guidance_embed: false,
            theta: 10000.0,
            axes_dim: [16, 56, 56],
            requires_sigma_shift: true,
        }
    }

    pub fn dev() -> Self {
        Self {
            variant: FluxVariant::Dev,
            num_transformer_blocks: 19,
            num_single_transformer_blocks: 38,
            in_channels: 64,
            out_channels: 64,
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            joint_attention_dim: 4096,
            pooled_projection_dim: 768,
            guidance_embed: true,
            theta: 10000.0,
            axes_dim: [16, 56, 56],
            requires_sigma_shift: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            variant: FluxVariant::Schnell,
            num_transformer_blocks: 1,
            num_single_transformer_blocks: 1,
            in_channels: 64,
            out_channels: 64,
            hidden_size: 256,
            num_heads: 2,
            head_dim: 128,
            joint_attention_dim: 4096,
            pooled_projection_dim: 768,
            guidance_embed: false,
            theta: 10000.0,
            axes_dim: [16, 56, 56],
            requires_sigma_shift: true,
        }
    }

    pub fn x_embedder_input_dim(&self) -> usize {
        self.in_channels
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub width: u32,
    pub height: u32,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl RuntimeConfig {
    pub fn for_variant(variant: FluxVariant, width: u32, height: u32, seed: u64) -> Self {
        match variant {
            FluxVariant::Schnell => Self {
                width,
                height,
                num_inference_steps: 4,
                guidance_scale: 0.0,
                seed,
            },
            FluxVariant::Dev => Self {
                width,
                height,
                num_inference_steps: 25,
                guidance_scale: 3.5,
                seed,
            },
        }
    }

    pub fn latent_height(&self) -> usize {
        (self.height as usize) / 16
    }

    pub fn latent_width(&self) -> usize {
        (self.width as usize) / 16
    }

    pub fn image_seq_len(&self) -> usize {
        self.latent_height() * self.latent_width()
    }
}
