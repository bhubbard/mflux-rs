use mlx_rs::Array;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct EmbedND {
    pub dim: usize,
    pub theta: f32,
    pub axes_dim: [usize; 3],
}

impl Default for EmbedND {
    fn default() -> Self {
        Self::new(3072, 10000.0, [16, 56, 56])
    }
}

impl EmbedND {
    pub fn new(dim: usize, theta: f32, axes_dim: [usize; 3]) -> Self {
        Self {
            dim,
            theta,
            axes_dim,
        }
    }

    /// Prepare text position IDs of shape [1, seq_len, 3] filled with zeros
    pub fn prepare_text_ids(seq_len: usize) -> Result<Array> {
        let zeros = vec![0.0f32; seq_len * 3];
        Ok(Array::from_slice(&zeros, &[1, seq_len as i32, 3]))
    }

    /// Prepare latent image position IDs of shape [1, (H/16) * (W/16), 3]
    /// axis 0: 0, axis 1: y coord (0..h), axis 2: x coord (0..w)
    pub fn prepare_latent_image_ids(height: u32, width: u32) -> Result<Array> {
        let latent_h = (height / 16) as usize;
        let latent_w = (width / 16) as usize;
        let total_tokens = latent_h * latent_w;

        let mut data = Vec::with_capacity(total_tokens * 3);
        for y in 0..latent_h {
            for x in 0..latent_w {
                data.push(0.0f32);
                data.push(y as f32);
                data.push(x as f32);
            }
        }

        Ok(Array::from_slice(&data, &[1, total_tokens as i32, 3]))
    }

    /// Compute 1D RoPE frequencies (cos and sin) for a given axis
    pub fn compute_rope_freqs(pos: &[f32], dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
        let half_dim = dim / 2;
        let mut cos = Vec::with_capacity(pos.len() * half_dim);
        let mut sin = Vec::with_capacity(pos.len() * half_dim);

        for &p in pos {
            for i in 0..half_dim {
                let scale = (2 * i) as f32 / dim as f32;
                let omega = 1.0f32 / theta.powf(scale);
                let angle = p * omega;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        (cos, sin)
    }

    /// Apply Rotary Position Embedding to Query or Key vectors of shape [batch, heads, seq_len, head_dim]
    pub fn apply_rotary_emb(
        x: &Array,
        cos: &[f32],
        sin: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Array> {
        let shape = x.shape();
        let total_size = x.size();
        let flat = x.reshape(&[total_size as i32])?;
        let src = flat.as_slice::<f32>();

        let half_head_dim = head_dim / 2;
        let num_heads = if shape.len() >= 2 { shape[1] as usize } else { 1 };
        let batch_size = if !shape.is_empty() { shape[0] as usize } else { 1 };

        let mut out = src.to_vec();

        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let freq_base = s * half_head_dim;

                    for i in 0..half_head_dim {
                        let idx0 = base_idx + 2 * i;
                        let idx1 = base_idx + 2 * i + 1;

                        if idx1 < out.len() && freq_base + i < cos.len() {
                            let c = cos[freq_base + i];
                            let s_val = sin[freq_base + i];
                            let x0 = src[idx0];
                            let x1 = src[idx1];

                            out[idx0] = x0 * c - x1 * s_val;
                            out[idx1] = x0 * s_val + x1 * c;
                        }
                    }
                }
            }
        }

        Ok(Array::from_slice(&out, shape))
    }
}
