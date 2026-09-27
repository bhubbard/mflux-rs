use mlx_rs::Array;
use mflux_rs::model::vae::FluxVae;

#[test]
fn test_vae_factors() {
    let vae = FluxVae::default();
    assert!((vae.scaling_factor - 0.3611).abs() < 1e-4);
    assert!((vae.shift_factor - 0.1159).abs() < 1e-4);
}

#[test]
fn test_vae_scale_unscale_roundtrip() {
    let vae = FluxVae::default();
    let sample = Array::from_slice(&[1.0f32, -0.5, 0.25, 2.0], &[1, 4]);

    let scaled = vae.scale_latents(&sample).unwrap();
    let unscaled = vae.unscale_latents(&scaled).unwrap();

    let flat = unscaled.reshape(&[4]).unwrap();
    let slice = flat.as_slice::<f32>();

    let orig = [1.0f32, -0.5, 0.25, 2.0];
    for (a, b) in slice.iter().zip(orig.iter()) {
        assert!((a - b).abs() < 1e-4, "roundtrip mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_vae_decode_to_rgb() {
    let vae = FluxVae::default();
    let latents = Array::zeros::<f32>(&[1, 16, 16, 16]).unwrap();
    let decoded = vae.decode(&latents, 128, 128).unwrap();

    assert_eq!(decoded.shape(), &[1, 3, 128, 128]);

    let img = FluxVae::to_rgb_image(&decoded, 128, 128).unwrap();
    assert_eq!(img.width(), 128);
    assert_eq!(img.height(), 128);
}
