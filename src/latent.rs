use mlx_rs::ops::transpose_axes;
use mlx_rs::Array;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::error::Result;

pub struct FluxLatentCreator;

impl FluxLatentCreator {
    /// Create random Gaussian noise for FLUX latents of shape [1, (H/16)*(W/16), 64]
    pub fn create_noise(seed: u64, height: u32, width: u32) -> Result<Array> {
        let h_patches = (height / 16) as usize;
        let w_patches = (width / 16) as usize;
        let seq_len = h_patches * w_patches;
        let channels = 64;
        let total = seq_len * channels;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(total);

        // Box-Muller transform for standard normal distribution
        for _ in 0..(total / 2) {
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(r * theta.cos());
            data.push(r * theta.sin());
        }
        if total % 2 != 0 {
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos());
        }

        Ok(Array::from_slice(&data, &[1, seq_len as i32, channels as i32]))
    }

    /// Pack latents from [1, 16, H/8, W/8] to [1, (H/16)*(W/16), 64]
    pub fn pack_latents(latents: &Array, height: u32, width: u32) -> Result<Array> {
        let h_patches = (height / 16) as i32;
        let w_patches = (width / 16) as i32;

        let shaped = latents.reshape(&[1, 16, h_patches, 2, w_patches, 2])?;
        let transposed = transpose_axes(&shaped, &[0, 2, 4, 1, 3, 5])?;
        Ok(transposed.reshape(&[1, h_patches * w_patches, 64])?)
    }

    /// Unpack latents from [1, (H/16)*(W/16), 64] to [1, 16, H/8, W/8]
    pub fn unpack_latents(latents: &Array, height: u32, width: u32) -> Result<Array> {
        let h_patches = (height / 16) as i32;
        let w_patches = (width / 16) as i32;

        let shaped = latents.reshape(&[1, h_patches, w_patches, 16, 2, 2])?;
        let transposed = transpose_axes(&shaped, &[0, 3, 1, 4, 2, 5])?;
        Ok(transposed.reshape(&[1, 16, h_patches * 2, w_patches * 2])?)
    }
}
