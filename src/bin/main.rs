use std::path::PathBuf;
use std::time::Instant;
use clap::{Parser, Subcommand, ValueEnum};
use mflux_rs::config::{FluxVariant, ModelConfig, RuntimeConfig};
use mflux_rs::error::Result;
use mflux_rs::pipeline::FluxPipeline;

#[derive(Parser, Debug)]
#[command(
    name = "mflux",
    about = "Native Apple Silicon FLUX.1 generative image synthesis using MLX",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum ModelChoice {
    Schnell,
    Dev,
}

impl From<ModelChoice> for FluxVariant {
    fn from(choice: ModelChoice) -> Self {
        match choice {
            ModelChoice::Schnell => FluxVariant::Schnell,
            ModelChoice::Dev => FluxVariant::Dev,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate an image from a prompt using FLUX.1
    Generate {
        /// Text prompt describing image to generate
        #[arg(short, long)]
        prompt: String,

        /// Model variant (schnell or dev)
        #[arg(short, long, value_enum, default_value_t = ModelChoice::Schnell)]
        model: ModelChoice,

        /// Number of denoising steps (default 4 for schnell, 25 for dev)
        #[arg(short, long)]
        steps: Option<usize>,

        /// Guidance scale (default 0.0 for schnell, 3.5 for dev)
        #[arg(short, long)]
        guidance: Option<f32>,

        /// Image width in pixels
        #[arg(long, default_value_t = 512)]
        width: u32,

        /// Image height in pixels
        #[arg(long, default_value_t = 512)]
        height: u32,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output image file path
        #[arg(short, long, default_value = "flux_output.png")]
        output: PathBuf,
    },

    /// Benchmark FLUX.1 step latency on Apple Silicon
    Bench {
        /// Model variant
        #[arg(short, long, value_enum, default_value_t = ModelChoice::Schnell)]
        model: ModelChoice,

        /// Steps to benchmark
        #[arg(short, long, default_value_t = 4)]
        steps: usize,
    },

    /// Display model parameters, architecture, and memory footprint
    Info {
        /// Model variant
        #[arg(short, long, value_enum, default_value_t = ModelChoice::Schnell)]
        model: ModelChoice,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            prompt,
            model,
            steps,
            guidance,
            width,
            height,
            seed,
            output,
        } => {
            let variant = FluxVariant::from(model);
            let mut runtime_cfg = RuntimeConfig::for_variant(variant, width, height, seed);
            if let Some(s) = steps {
                runtime_cfg.num_inference_steps = s;
            }
            if let Some(g) = guidance {
                runtime_cfg.guidance_scale = g;
            }

            println!("⚡ mflux-rs: Native Apple Silicon FLUX.1 Generative Engine");
            println!("  Variant:        {:?}", variant);
            println!("  Prompt:         \"{}\"", prompt);
            println!("  Resolution:     {}x{}", width, height);
            println!("  Steps:          {}", runtime_cfg.num_inference_steps);
            println!("  Guidance:       {:.1}", runtime_cfg.guidance_scale);
            println!("  Seed:           {}", seed);
            println!("  Output:         {}", output.display());

            let start = Instant::now();
            let mut pipeline = FluxPipeline::new(variant)?;
            println!("  Model loaded in {:.2?}", start.elapsed());

            let gen_start = Instant::now();
            pipeline.generate_and_save(&prompt, &runtime_cfg, &output)?;
            println!("✓ Image successfully generated and saved in {:.2?}!", gen_start.elapsed());
        }

        Commands::Bench { model, steps } => {
            let variant = FluxVariant::from(model);
            let runtime_cfg = RuntimeConfig::for_variant(variant, 512, 512, 42);
            let mut pipeline = FluxPipeline::new(variant)?;

            println!("Benchmarking FLUX.1 ({:?}) for {} steps at 512x512...", variant, steps);
            let start = Instant::now();
            let _img = pipeline.generate("Benchmark prompt", &runtime_cfg)?;
            let elapsed = start.elapsed();
            let per_step = elapsed.as_secs_f64() / (steps as f64);
            println!("✓ Total time: {:.2?}, {:.2} ms/step", elapsed, per_step * 1000.0);
        }

        Commands::Info { model } => {
            let variant = FluxVariant::from(model);
            let config = match variant {
                FluxVariant::Schnell => ModelConfig::schnell(),
                FluxVariant::Dev => ModelConfig::dev(),
            };

            println!("FLUX.1 Model Architecture Info:");
            println!("  Variant:                      {:?}", config.variant);
            println!("  Dual-Stream Blocks:           {}", config.num_transformer_blocks);
            println!("  Single-Stream Blocks:         {}", config.num_single_transformer_blocks);
            println!("  Hidden Size:                  {}", config.hidden_size);
            println!("  Attention Heads:              {}", config.num_heads);
            println!("  Head Dimension:               {}", config.head_dim);
            println!("  RoPE Frequency Axes:          {:?}", config.axes_dim);
            println!("  Guidance Embedder:            {}", config.guidance_embed);
            println!("  Total MMDiT Parameters:       ~12.0 Billion");
            println!("  FP16 Memory Footprint:        ~24 GB unified RAM");
            println!("  4-bit Quantized Footprint:    ~8.2 GB unified RAM");
        }
    }

    Ok(())
}
