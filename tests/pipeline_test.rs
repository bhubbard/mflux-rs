use mflux_rs::config::{FluxVariant, ModelConfig, RuntimeConfig};
use mflux_rs::pipeline::FluxPipeline;

#[test]
fn test_pipeline_generate_tiny() {
    let tiny_config = ModelConfig::tiny();
    let mut pipeline = FluxPipeline::with_config(tiny_config).unwrap();
    let mut runtime_cfg = RuntimeConfig::for_variant(FluxVariant::Schnell, 64, 64, 12345);
    runtime_cfg.num_inference_steps = 1; // 1 step integration test

    let img = pipeline.generate("A cinematic photograph of a mountain sunset", &runtime_cfg).unwrap();
    assert_eq!(img.width(), 64);
    assert_eq!(img.height(), 64);
}
