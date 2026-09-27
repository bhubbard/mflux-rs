use mflux_rs::model::rope::EmbedND;

#[test]
fn test_prepare_text_ids() {
    let ids = EmbedND::prepare_text_ids(128).unwrap();
    assert_eq!(ids.shape(), &[1, 128, 3]);
}

#[test]
fn test_prepare_latent_image_ids() {
    let ids = EmbedND::prepare_latent_image_ids(512, 512).unwrap();
    // 512/16 = 32 -> 32*32 = 1024 tokens
    assert_eq!(ids.shape(), &[1, 1024, 3]);
}

#[test]
fn test_rope_frequencies_computation() {
    let positions = vec![0.0f32, 1.0, 2.0, 3.0];
    let (cos, sin) = EmbedND::compute_rope_freqs(&positions, 128, 10000.0);

    // 4 positions * (128 / 2) = 256 values
    assert_eq!(cos.len(), 256);
    assert_eq!(sin.len(), 256);

    // Position 0: cos(0) == 1, sin(0) == 0
    for i in 0..64 {
        assert!((cos[i] - 1.0).abs() < 1e-5);
        assert!(sin[i].abs() < 1e-5);
    }
}
