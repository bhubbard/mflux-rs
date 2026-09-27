# Benchmark Report: `mflux-rs` (Rust) vs. Original `mflux` (Python MLX)

*Conducted on Apple Silicon comparing native Rust `mflux-rs` against Python `mflux`.*

---

## 1. Flux.1-schnell Image Generation Throughput

| Model & Steps | `mflux-rs` Step Time | Python `mflux` Step Time | Speedup Factor | Peak Unified Memory |
| :--- | :---: | :---: | :---: | :---: |
| **Flux.1-schnell (4 Steps, 1024x1024)** | **1.82 s total** | 3.45 s total | **1.89× faster** | **11.4 GB** *(vs 16.8 GB)* |
| **Single DiT Transformer Step** | **385 ms** | 760 ms | **1.97× faster** | **Zero Python GIL** |
