use thiserror::Error;

#[derive(Error, Debug)]
pub enum MfluxError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Shape mismatch: expected {expected}, found {found}")]
    ShapeMismatch { expected: String, found: String },

    #[error("Scheduler error: {0}")]
    Scheduler(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

pub type Result<T> = std::result::Result<T, MfluxError>;
