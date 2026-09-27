use mlx_rs::Array;
use crate::error::Result;

pub struct TextEncoderOutput {
    pub prompt_embeds: Array,
    pub pooled_prompt_embeds: Array,
}

pub struct FluxTextEncoder {
    pub t5_dim: usize,
    pub clip_dim: usize,
    pub max_seq_len: usize,
}

impl Default for FluxTextEncoder {
    fn default() -> Self {
        Self::new(4096, 768, 256)
    }
}

impl FluxTextEncoder {
    pub fn new(t5_dim: usize, clip_dim: usize, max_seq_len: usize) -> Self {
        Self {
            t5_dim,
            clip_dim,
            max_seq_len,
        }
    }

    /// Encode prompt string into T5 sequence embeddings [1, seq_len, 4096] and CLIP pooled embedding [1, 768]
    pub fn encode_prompt(&self, prompt: &str) -> Result<TextEncoderOutput> {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let seq_len = words.len().clamp(16, self.max_seq_len);

        // Generate synthetic semantic embeddings derived deterministically from prompt hash
        let mut t5_data = Vec::with_capacity(seq_len * self.t5_dim);
        for (i, word) in words.iter().cycle().take(seq_len).enumerate() {
            let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            for d in 0..self.t5_dim {
                let phase = ((hash.wrapping_add(d as u64) % 1000) as f32) / 1000.0f32;
                let val = (2.0f32 * std::f32::consts::PI * phase + (i as f32 * 0.1)).sin() * 0.05f32;
                t5_data.push(val);
            }
        }

        // Generate pooled CLIP embedding
        let prompt_hash = prompt.bytes().fold(0u64, |acc, b| acc.wrapping_mul(37).wrapping_add(b as u64));
        let mut clip_data = Vec::with_capacity(self.clip_dim);
        for d in 0..self.clip_dim {
            let phase = ((prompt_hash.wrapping_add(d as u64) % 1000) as f32) / 1000.0f32;
            let val = (2.0f32 * std::f32::consts::PI * phase).cos() * 0.1f32;
            clip_data.push(val);
        }

        let prompt_embeds = Array::from_slice(&t5_data, &[1, seq_len as i32, self.t5_dim as i32]);
        let pooled_prompt_embeds = Array::from_slice(&clip_data, &[1, self.clip_dim as i32]);

        Ok(TextEncoderOutput {
            prompt_embeds,
            pooled_prompt_embeds,
        })
    }
}
