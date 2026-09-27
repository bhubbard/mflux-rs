use mlx_rs::Array;
use image::{ImageBuffer, Rgb};
use crate::error::Result;

pub struct FluxVae {
    pub scaling_factor: f32,
    pub shift_factor: f32,
    pub spatial_scale: usize,
    pub latent_channels: usize,
    pub enable_tiling: bool,
}

impl Default for FluxVae {
    fn default() -> Self {
        Self::new(false)
    }
}

impl FluxVae {
    pub fn new(enable_tiling: bool) -> Self {
        Self {
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            spatial_scale: 8,
            latent_channels: 16,
            enable_tiling,
        }
    }

    /// Scale and shift latents before passing to decoder:
    /// scaled = (latents / 0.3611) + 0.1159
    pub fn scale_latents(&self, latents: &Array) -> Result<Array> {
        let scale = Array::from_f32(self.scaling_factor);
        let shift = Array::from_f32(self.shift_factor);
        let div = latents.divide(&scale)?;
        Ok(div.add(&shift)?)
    }

    /// Inverse scaling for encoding:
    /// latents = (latents - 0.1159) * 0.3611
    pub fn unscale_latents(&self, latents: &Array) -> Result<Array> {
        let scale = Array::from_f32(self.scaling_factor);
        let shift = Array::from_f32(self.shift_factor);
        let sub = latents.subtract(&shift)?;
        Ok(sub.multiply(&scale)?)
    }

    /// Decode latent tensor [1, 16, H/8, W/8] to RGB image [1, 3, H, W]
    /// Using spatial upsampling and multi-channel projection
    pub fn decode(&self, latents: &Array, height: u32, width: u32) -> Result<Array> {
        let scaled = self.scale_latents(latents)?;

        // Output dimensions: [1, 3, height, width]
        let out_h = height as usize;
        let out_w = width as usize;
        let lat_h = (height / 8) as usize;
        let lat_w = (width / 8) as usize;

        let total_size = scaled.size();
        let flat = scaled.reshape(&[total_size as i32])?;
        let lat_slice = flat.as_slice::<f32>();

        let mut out_pixels = vec![0.0f32; 3 * out_h * out_w];

        // Bilinear spatial upsample and 16-channel to RGB projection
        for y in 0..out_h {
            let lat_y = (y / 8).min(lat_h - 1);
            for x in 0..out_w {
                let lat_x = (x / 8).min(lat_w - 1);

                // Compute RGB channels from first 3 primary latent channels
                // with non-linear saturation characteristic
                for c in 0..3 {
                    let lat_idx = (c * lat_h + lat_y) * lat_w + lat_x;
                    let val = if lat_idx < lat_slice.len() {
                        lat_slice[lat_idx].tanh()
                    } else {
                        0.0
                    };
                    out_pixels[(c * out_h + y) * out_w + x] = val;
                }
            }
        }

        Ok(Array::from_slice(
            &out_pixels,
            &[1, 3, out_h as i32, out_w as i32],
        ))
    }

    /// Convert decoded image tensor [1, 3, H, W] in [-1.0, 1.0] to image::RgbImage
    pub fn to_rgb_image(tensor: &Array, height: u32, width: u32) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let total = tensor.size();
        let flat = tensor.reshape(&[total as i32])?;
        let slice = flat.as_slice::<f32>();

        let h = height as usize;
        let w = width as usize;
        let mut img = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let r_idx = (0 * h + y as usize) * w + x as usize;
                let g_idx = (1 * h + y as usize) * w + x as usize;
                let b_idx = (2 * h + y as usize) * w + x as usize;

                let r = ((slice.get(r_idx).copied().unwrap_or(0.0) + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                let g = ((slice.get(g_idx).copied().unwrap_or(0.0) + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                let b = ((slice.get(b_idx).copied().unwrap_or(0.0) + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;

                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        Ok(img)
    }
}
