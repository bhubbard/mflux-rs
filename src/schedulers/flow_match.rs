use mlx_rs::Array;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: f32,
    pub shift_terminal: f32,
    pub num_inference_steps: usize,
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let num_train_timesteps = 1000.0f32;
        let shift_terminal = 0.02f32;

        let (sigmas, timesteps) = Self::compute_timesteps_and_sigmas(
            num_inference_steps,
            num_train_timesteps,
            shift_terminal,
        );

        Self {
            num_train_timesteps,
            shift_terminal,
            num_inference_steps,
            sigmas,
            timesteps,
        }
    }

    pub fn with_image_seq_len(num_inference_steps: usize, image_seq_len: usize) -> Self {
        let num_train_timesteps = 1000.0f32;
        let shift_terminal = 0.02f32;
        let (timesteps, sigmas) = Self::get_timesteps_and_sigmas(
            image_seq_len,
            num_inference_steps,
            num_train_timesteps,
        );

        Self {
            num_train_timesteps,
            shift_terminal,
            num_inference_steps,
            sigmas,
            timesteps,
        }
    }

    pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
        let a1 = 8.73809524e-05f32;
        let b1 = 1.89833333f32;
        let a2 = 0.00016927f32;
        let b2 = 0.45666666f32;
        let n = image_seq_len as f32;
        let s = num_steps as f32;

        if image_seq_len > 4300 {
            return a2 * n + b2;
        }

        let m_200 = a2 * n + b2;
        let m_10 = a1 * n + b1;
        let a = (m_200 - m_10) / 190.0f32;
        let b = m_200 - 200.0f32 * a;
        a * s + b
    }

    pub fn time_shift_exponential(mu: f32, sigma_power: f32, t: f32) -> f32 {
        let exp_mu = mu.exp();
        exp_mu / (exp_mu + (1.0f32 / t - 1.0f32).powf(sigma_power))
    }

    pub fn stretch_to_terminal(sigmas: &[f32], shift_terminal: f32) -> Vec<f32> {
        let one_minus_sigmas: Vec<f32> = sigmas.iter().map(|s| 1.0f32 - s).collect();
        let scale_factor = one_minus_sigmas.last().copied().unwrap_or(1.0) / (1.0f32 - shift_terminal);
        one_minus_sigmas
            .iter()
            .map(|oms| 1.0f32 - (oms / scale_factor))
            .collect()
    }

    pub fn compute_timesteps_and_sigmas(
        num_steps: usize,
        num_train_timesteps: f32,
        shift_terminal: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        if num_steps <= 1 {
            return (vec![1.0, 0.0], vec![num_train_timesteps]);
        }

        let sigma_min = 1.0f32 / num_train_timesteps;
        let sigma_max = 1.0f32;

        let timesteps_linear: Vec<f32> = (0..num_steps)
            .map(|i| {
                sigma_max * num_train_timesteps
                    - (i as f32) * (sigma_max - sigma_min) * num_train_timesteps
                        / ((num_steps - 1) as f32)
            })
            .collect();

        let sigmas_linear: Vec<f32> = timesteps_linear
            .iter()
            .map(|t| t / num_train_timesteps)
            .collect();

        let sigmas_shifted: Vec<f32> = sigmas_linear
            .iter()
            .map(|s| Self::time_shift_exponential(1.0, 1.0, *s))
            .collect();

        let sigmas_final = Self::stretch_to_terminal(&sigmas_shifted, shift_terminal);
        let timesteps: Vec<f32> = sigmas_final.iter().map(|s| s * num_train_timesteps).collect();

        let mut sigmas_with_zero = sigmas_final;
        sigmas_with_zero.push(0.0);

        (sigmas_with_zero, timesteps)
    }

    pub fn get_timesteps_and_sigmas(
        image_seq_len: usize,
        num_inference_steps: usize,
        num_train_timesteps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        if num_inference_steps <= 1 {
            return (vec![num_train_timesteps], vec![1.0, 0.0]);
        }

        let mu = Self::compute_empirical_mu(image_seq_len, num_inference_steps);
        let mut sigmas: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let frac = 1.0f32 - (i as f32) / (num_inference_steps as f32) * (1.0f32 - 1.0f32 / (num_inference_steps as f32));
                Self::time_shift_exponential(mu, 1.0, frac)
            })
            .collect();

        let timesteps: Vec<f32> = sigmas.iter().map(|s| s * num_train_timesteps).collect();
        sigmas.push(0.0);
        (timesteps, sigmas)
    }

    /// Step formula: x_{next} = x + (sigma_{i+1} - sigma_i) * noise
    pub fn step(
        &self,
        noise: &Array,
        step_idx: usize,
        latents: &Array,
    ) -> Result<Array> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(crate::error::MfluxError::Scheduler(format!(
                "Step index {} out of bounds for sigmas length {}",
                step_idx,
                self.sigmas.len()
            )));
        }

        let s1 = self.sigmas[step_idx + 1];
        let s2 = self.sigmas[step_idx];
        let dt = s1 - s2;

        let dt_arr = Array::from_f32(dt);
        let scaled_noise = noise.multiply(&dt_arr)?;
        let next_latents = latents.add(&scaled_noise)?;

        Ok(next_latents)
    }
}
