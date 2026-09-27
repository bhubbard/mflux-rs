use mlx_rs::Array;
use crate::config::ModelConfig;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct LinearScheduler {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl LinearScheduler {
    pub fn new(
        num_inference_steps: usize,
        width: u32,
        height: u32,
        model_config: &ModelConfig,
    ) -> Self {
        let n = num_inference_steps;
        let mut raw_sigmas: Vec<f32> = (0..n)
            .map(|i| {
                1.0f32 - (i as f32) / (n as f32) * (1.0f32 - 1.0f32 / (n as f32))
            })
            .collect();

        if model_config.requires_sigma_shift {
            let base_shift = 0.5f32;
            let max_shift = 1.15f32;
            let base_seq_len = 256.0f32;
            let max_seq_len = 4096.0f32;

            let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
            let b = base_shift - m * base_seq_len;
            let img_seq_len = ((width * height) / 256) as f32;
            let mu = m * img_seq_len + b;
            let exp_mu = mu.exp();

            let mut shifted: Vec<f32> = raw_sigmas
                .iter()
                .map(|&s| exp_mu / (exp_mu + (1.0f32 / s - 1.0f32)))
                .collect();

            // terminal shift if configured
            let terminal = 0.02f32;
            let one_minus: Vec<f32> = shifted.iter().map(|s| 1.0f32 - s).collect();
            let scale = one_minus.last().copied().unwrap_or(1.0) / (1.0f32 - terminal);
            shifted = one_minus.iter().map(|om| 1.0f32 - (om / scale)).collect();
            shifted.push(0.0);
            raw_sigmas = shifted;
        } else {
            raw_sigmas.push(0.0);
        }

        let timesteps: Vec<f32> = (0..n).map(|i| i as f32).collect();

        Self {
            sigmas: raw_sigmas,
            timesteps,
        }
    }

    pub fn step(
        &self,
        noise: &Array,
        step_idx: usize,
        latents: &Array,
    ) -> Result<Array> {
        let s1 = self.sigmas[step_idx + 1];
        let s2 = self.sigmas[step_idx];
        let dt = s1 - s2;
        let dt_arr = Array::from_f32(dt);
        let scaled_noise = noise.multiply(&dt_arr)?;
        Ok(latents.add(&scaled_noise)?)
    }
}
