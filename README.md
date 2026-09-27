# mflux-rs ⚡

[![CI](https://github.com/bhubbard/mflux-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/mflux-rs/actions/workflows/ci.yml)
[![Pages](https://github.com/bhubbard/mflux-rs/actions/workflows/pages.yml/badge.svg)](https://github.com/bhubbard/mflux-rs/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

High-performance native Apple Silicon Rust port of the **FLUX.1** generative text-to-image architecture, powered by MLX.

Interactive documentation and trajectory simulator: **[code.brandonhubbard.com/mflux-rs](http://code.brandonhubbard.com/mflux-rs/)**

---

## Highlights

- **Pure Rust MMDiT**: Native implementation of the Multimodal Diffusion Transformer (19 dual-stream joint attention blocks + 38 single-stream concatenated blocks).
- **Zero Python Runtime**: No Python interpreter, GIL contention, or virtual environment overhead on Apple Silicon unified memory.
- **3D RoPE Coordinates**: Multi-dimensional Rotary Position Embeddings along axes `[16, 56, 56]` mapping row, column, and frame positioning.
- **Flow-Matching Euler Solver**: FlowMatchEulerDiscreteScheduler ODE integrator with terminal stretch $\sigma_{\text{terminal}} = 0.02$.
- **High-Speed Tiled VAE**: Spatial patch packing $(16, 2, 2) \leftrightarrow 64$ with low-memory tiled upsampling.
- **Dual Text Conditioning**: Seamless T5-XXL 4096-dim context sequences and CLIP-L 768-dim pooled representations.

---

## Installation

```bash
# Clone and build
git clone https://github.com/bhubbard/mflux-rs.git
cd mflux-rs
cargo build --release
```

---

## CLI Usage

### Generate with FLUX.1-schnell (4 Steps)
```bash
mflux generate \
  --prompt "A cinematic wide shot of a misty neon cyberpunk alleyway in Tokyo" \
  --model schnell \
  --steps 4 \
  --output tokyo.png
```

### Generate with FLUX.1-dev (Guidance Scale 3.5)
```bash
mflux generate \
  --prompt "Hyperrealistic portrait of an astronaut on Mars during golden hour" \
  --model dev \
  --steps 25 \
  --guidance 3.5 \
  --width 1024 \
  --height 1024 \
  --output mars.png
```

### Benchmark Step Latency
```bash
mflux bench --model schnell --steps 4
```

### Inspect Architecture & Unified VRAM
```bash
mflux info --model dev
```

---

## Rust Library API

```rust
use mflux_rs::config::{FluxVariant, RuntimeConfig};
use mflux_rs::pipeline::FluxPipeline;
use std::path::Path;

fn main() -> mflux_rs::Result<()> {
    // Initialize pipeline with FLUX.1-schnell
    let mut pipeline = FluxPipeline::new(FluxVariant::Schnell)?;

    // Configure generation parameters
    let runtime_cfg = RuntimeConfig::for_variant(
        FluxVariant::Schnell,
        1024, // width
        1024, // height
        42,   // seed
    );

    // Generate and save directly
    pipeline.generate_and_save(
        "A serene minimalist Japanese garden in winter",
        &runtime_cfg,
        Path::new("garden.png"),
    )?;

    Ok(())
}
```

---

## Test Suite

```bash
cargo test
```

Passes all 13 unit and pipeline integration tests covering FlowMatch ODE schedules, 3D RoPE embeddings, latent packing/unpacking, and VAE decoding.

---

## License

Dual licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

Ported from [filipstrand/mflux](https://github.com/filipstrand/mflux) and FLUX.1 by Black Forest Labs.
