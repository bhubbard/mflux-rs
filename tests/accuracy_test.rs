//! Analytical and Numerical Accuracy Verification Tests for mflux-rs
//!
//! Validates:
//! 1. Exponential time-shift asymptotic limits and monotonic curvature.
//! 2. Empirical mu piecewise continuity for dynamic resolution scaling.
//! 3. Latent space packing/unpacking lossless spatial bijection.
//! 4. Box-Muller Gaussian distribution mean and variance conservation.

use mflux_rs::latent::FluxLatentCreator;
use mflux_rs::schedulers::FlowMatchEulerDiscreteScheduler;
use mlx_rs::Array;

#[test]
fn test_exponential_time_shift_asymptotics_and_monotonicity() {
    for &mu in &[0.5f32, 1.0, 2.0, 3.5] {
        // Asymptote at t -> 1.0
        let val_at_one = FlowMatchEulerDiscreteScheduler::time_shift_exponential(mu, 1.0, 1.0);
        assert!(
            (val_at_one - 1.0).abs() < 1e-6,
            "Shift at t=1.0 must be 1.0 (got {})",
            val_at_one
        );

        // Limit as t -> 0
        let val_near_zero = FlowMatchEulerDiscreteScheduler::time_shift_exponential(mu, 1.0, 1e-5);
        assert!(
            val_near_zero < 1e-3,
            "Shift at t->0 must approach 0.0 (got {})",
            val_near_zero
        );

        // Strict monotonic increase: for t1 < t2 => f(t1) < f(t2)
        let mut prev = 0.0f32;
        for step in 1..=50 {
            let t = step as f32 / 50.0;
            let current = FlowMatchEulerDiscreteScheduler::time_shift_exponential(mu, 1.0, t);
            assert!(
                current > prev,
                "Monotonicity violation at t={}: prev {}, current {}",
                t,
                prev,
                current
            );
            prev = current;
        }
    }
}

#[test]
fn test_flux_empirical_mu_piecewise_continuity() {
    // For n > 4300, formula is a2 * n + b2
    let n_large = 5000;
    let mu_5000 = FlowMatchEulerDiscreteScheduler::compute_empirical_mu(n_large, 28);
    let expected = 0.00016927f32 * (n_large as f32) + 0.45666666f32;
    assert!(
        (mu_5000 - expected).abs() < 1e-5,
        "Empirical mu for large sequence: got {}, expected {}",
        mu_5000,
        expected
    );

    // Continuity around boundary n = 4300
    let mu_pre = FlowMatchEulerDiscreteScheduler::compute_empirical_mu(4299, 200);
    let mu_post = FlowMatchEulerDiscreteScheduler::compute_empirical_mu(4301, 200);
    assert!(
        (mu_pre - mu_post).abs() < 0.01,
        "Piecewise continuity gap at boundary: pre {}, post {}",
        mu_pre,
        mu_post
    );
}

#[test]
fn test_latent_packing_unpacking_lossless_bijection() {
    let height = 32;
    let width = 32;
    let h_half = (height / 8) as usize; // 4
    let w_half = (width / 8) as usize; // 4
    let channels = 16;
    let total_elements = channels * h_half * w_half; // 16 * 4 * 4 = 256

    let data: Vec<f32> = (0..total_elements).map(|i| (i as f32) * 1.5 - 20.0).collect();
    let original = Array::from_slice(&data, &[1, channels as i32, h_half as i32, w_half as i32]);

    let packed = FluxLatentCreator::pack_latents(&original, height, width).expect("pack failed");
    assert_eq!(packed.shape(), &[1, 4, 64]);

    let unpacked =
        FluxLatentCreator::unpack_latents(&packed, height, width).expect("unpack failed");
    assert_eq!(unpacked.shape(), original.shape());

    let orig_slice: Vec<f32> = original.as_slice().to_vec();
    let unp_slice: Vec<f32> = unpacked.as_slice().to_vec();

    for i in 0..total_elements {
        let diff = (orig_slice[i] - unp_slice[i]).abs();
        assert_eq!(
            diff, 0.0,
            "Loss of fidelity at index {}: original {}, unpacked {}",
            i, orig_slice[i], unp_slice[i]
        );
    }
}

#[test]
fn test_box_muller_gaussian_statistics() {
    let noise = FluxLatentCreator::create_noise(42, 64, 64).expect("noise creation");
    let slice: Vec<f32> = noise.as_slice().to_vec();
    let n = slice.len() as f32;

    let mean = slice.iter().sum::<f32>() / n;
    let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    // Normal distribution N(0, 1) sample statistics with 1024 samples
    assert!(
        mean.abs() < 0.05,
        "Noise sample mean deviates from 0.0: got {}",
        mean
    );
    assert!(
        (variance - 1.0).abs() < 0.1,
        "Noise sample variance deviates from 1.0: got {}",
        variance
    );
}
