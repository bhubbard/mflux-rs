use mflux_rs::config::ModelConfig;
use mflux_rs::schedulers::{FlowMatchEulerDiscreteScheduler, LinearScheduler};

#[test]
fn test_one_step_schedule_is_full_denoise() {
    let scheduler = FlowMatchEulerDiscreteScheduler::new(1);
    assert_eq!(scheduler.sigmas, vec![1.0, 0.0]);
    assert_eq!(scheduler.timesteps, vec![1000.0]);
}

#[test]
fn test_two_step_schedule() {
    let scheduler = FlowMatchEulerDiscreteScheduler::new(2);
    let expected_sigmas = vec![1.0, 0.02, 0.0];
    let expected_timesteps = vec![1000.0, 20.0];

    assert_eq!(scheduler.sigmas.len(), expected_sigmas.len());
    for (a, b) in scheduler.sigmas.iter().zip(expected_sigmas.iter()) {
        assert!((a - b).abs() < 1e-4, "sigma mismatch: {} vs {}", a, b);
    }
    for (a, b) in scheduler.timesteps.iter().zip(expected_timesteps.iter()) {
        assert!((a - b).abs() < 1e-2, "timestep mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_four_step_schedule_matches_upstream() {
    let scheduler = FlowMatchEulerDiscreteScheduler::new(4);
    let expected_sigmas = vec![1.0, 0.847524, 0.584184, 0.02, 0.0];
    let expected_timesteps = vec![1000.0, 847.5239, 584.1836, 20.0];

    assert_eq!(scheduler.sigmas.len(), expected_sigmas.len());
    for (a, b) in scheduler.sigmas.iter().zip(expected_sigmas.iter()) {
        assert!((a - b).abs() < 1e-4, "sigma mismatch: {} vs {}", a, b);
    }
    for (a, b) in scheduler.timesteps.iter().zip(expected_timesteps.iter()) {
        assert!((a - b).abs() < 1e-2, "timestep mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_linear_scheduler_initialization() {
    let model_config = ModelConfig::schnell();
    let scheduler = LinearScheduler::new(4, 512, 512, &model_config);
    assert_eq!(scheduler.sigmas.len(), 5);
    assert_eq!(scheduler.timesteps.len(), 4);
    assert_eq!(scheduler.sigmas.last().copied(), Some(0.0));
}
