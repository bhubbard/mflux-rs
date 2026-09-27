use mlx_rs::Array;
use mflux_rs::latent::FluxLatentCreator;

#[test]
fn test_create_noise_dimensions() {
    let noise = FluxLatentCreator::create_noise(42, 512, 512).unwrap();
    assert_eq!(noise.shape(), &[1, 1024, 64]);
}

#[test]
fn test_latent_packing_unpacking_roundtrip() {
    // Spatial latents for 512x512 image: [1, 16, 512/8, 512/8] = [1, 16, 64, 64]
    let zeros = vec![0.5f32; 1 * 16 * 64 * 64];
    let original = Array::from_slice(&zeros, &[1, 16, 64, 64]);

    // Pack
    let packed = FluxLatentCreator::pack_latents(&original, 512, 512).unwrap();
    assert_eq!(packed.shape(), &[1, 1024, 64]);

    // Unpack
    let unpacked = FluxLatentCreator::unpack_latents(&packed, 512, 512).unwrap();
    assert_eq!(unpacked.shape(), &[1, 16, 64, 64]);
}
