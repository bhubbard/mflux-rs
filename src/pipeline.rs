use std::path::Path;
use mlx_rs::Array;
use crate::config::{FluxVariant, ModelConfig, RuntimeConfig};
use crate::error::Result;
use crate::latent::FluxLatentCreator;
use crate::model::rope::EmbedND;
use crate::model::{FluxTextEncoder, FluxTransformer, FluxVae};
use crate::schedulers::FlowMatchEulerDiscreteScheduler;

pub struct FluxPipeline {
    pub model_config: ModelConfig,
    pub transformer: FluxTransformer,
    pub text_encoder: FluxTextEncoder,
    pub vae: FluxVae,
}

impl FluxPipeline {
    pub fn new(variant: FluxVariant) -> Result<Self> {
        let model_config = match variant {
            FluxVariant::Schnell => ModelConfig::schnell(),
            FluxVariant::Dev => ModelConfig::dev(),
        };
        Self::with_config(model_config)
    }

    pub fn with_config(model_config: ModelConfig) -> Result<Self> {
        let transformer = FluxTransformer::new(model_config.clone())?;
        let text_encoder = FluxTextEncoder::default();
        let vae = FluxVae::default();

        Ok(Self {
            model_config,
            transformer,
            text_encoder,
            vae,
        })
    }

    /// Generate an image from a text prompt
    pub fn generate(&mut self, prompt: &str, runtime_config: &RuntimeConfig) -> Result<image::RgbImage> {
        // 1. Text encoding
        let enc_out = self.text_encoder.encode_prompt(prompt)?;
        let txt_len = enc_out.prompt_embeds.shape()[1] as usize;

        // 2. Initialize Gaussian noise latents [1, (H/16)*(W/16), 64]
        let mut latents = FluxLatentCreator::create_noise(
            runtime_config.seed,
            runtime_config.height,
            runtime_config.width,
        )?;

        // 3. Prepare positional RoPE coordinates
        let img_len = runtime_config.image_seq_len();
        let joint_seq_len = txt_len + img_len;

        // RoPE positional frequencies for axes [16, 56, 56]
        let positions: Vec<f32> = (0..joint_seq_len).map(|i| i as f32).collect();
        let (cos, sin) = EmbedND::compute_rope_freqs(&positions, self.model_config.head_dim, self.model_config.theta);

        // 4. Initialize Flow-Matching scheduler
        let scheduler = FlowMatchEulerDiscreteScheduler::new(runtime_config.num_inference_steps);

        // Guidance tensor if supported
        let guidance_arr = if self.model_config.guidance_embed {
            Some(Array::from_f32(runtime_config.guidance_scale * 1000.0f32))
        } else {
            None
        };

        // 5. Flow-matching ODE integration loop
        for step in 0..runtime_config.num_inference_steps {
            let sigma = scheduler.sigmas.get(step).copied().unwrap_or(1.0);
            let t = sigma * 1000.0f32;
            let t_arr = Array::from_f32(t);

            // Predict velocity field
            let velocity = self.transformer.forward(
                &t_arr,
                &latents,
                &enc_out.prompt_embeds,
                &enc_out.pooled_prompt_embeds,
                guidance_arr.as_ref(),
                &cos,
                &sin,
            )?;

            // Euler step
            latents = scheduler.step(&velocity, step, &latents)?;
        }

        // 6. Unpack latents from [1, (H/16)*(W/16), 64] to [1, 16, H/8, W/8]
        let unpacked = FluxLatentCreator::unpack_latents(
            &latents,
            runtime_config.height,
            runtime_config.width,
        )?;

        // 7. VAE decode to [1, 3, H, W]
        let rgb_tensor = self.vae.decode(
            &unpacked,
            runtime_config.height,
            runtime_config.width,
        )?;

        // 8. Convert to RGB image
        FluxVae::to_rgb_image(
            &rgb_tensor,
            runtime_config.height,
            runtime_config.width,
        )
    }

    /// Generate and save image directly to disk
    pub fn generate_and_save(
        &mut self,
        prompt: &str,
        runtime_config: &RuntimeConfig,
        output_path: &Path,
    ) -> Result<()> {
        let img = self.generate(prompt, runtime_config)?;
        img.save(output_path)?;
        Ok(())
    }
}
