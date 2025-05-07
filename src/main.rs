use clap::{Parser, Subcommand};
use std::f64::consts::TAU;

mod setup;
use setup::Setup;
mod simulate;
use simulate::{simulate, SimParams};

#[derive(Parser, Clone, Copy, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone, Copy, Debug)]
enum Command {
    /// Compute values for a set of different parameters.
    Compute {
        /// Minimum angle of incidence.
        #[arg(long, default_value_t = 10.)]
        angle_min: f64,
        /// Maximum angle of incidence.
        #[arg(long, default_value_t = 80.)]
        angle_max: f64,
        /// Number of simulations to run with different angle parameters.
        #[arg(long, default_value_t = 10)]
        angle_resolution: usize,
        /// Minimum frequency.
        #[arg(long, default_value_t = 2.0)]
        freq_min: f64,
        /// Maximum frequency.
        #[arg(long, default_value_t = 5.0)]
        freq_max: f64,
        /// Number of simulations to run with different frequency parameters.
        #[arg(long, default_value_t = 4)]
        freq_resolution: usize,
    },
    /// Run only one simulation and visualize it.
    Visualize {
        /// Incident angle of the wave.
        #[arg(long, default_value_t = 60.0)]
        angle: f64,
        /// Frequency of the wave.
        #[arg(long, default_value_t = 2.0)]
        frequency: f64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = Cli::parse();
    let setup = setup::Setup::new()?;

    match cli.command {
        Command::Compute {
            angle_min,
            angle_max,
            angle_resolution,
            freq_min,
            freq_max,
            freq_resolution,
        } => {
            let angle_step = (angle_max - angle_min) / (angle_resolution as f64 - 1.);
            let freq_step = (freq_max - freq_min) / (freq_resolution as f64 - 1.);

            for angle_idx in 0..angle_resolution {
                // angle is given in degrees for convenience on the command line,
                // remember to convert to radians
                let angle = (angle_min + angle_idx as f64 * angle_step) * TAU / 360.;
                for freq_idx in 0..freq_resolution {
                    let frequency = freq_min + freq_idx as f64 * freq_step;
                    let params = SimParams {
                        angle,
                        frequency,
                        visualize: false,
                    };
                    let measurements = simulate(params, &setup);
                    dbg!(params);
                    dbg!(measurements);
                }
            }
        }
        Command::Visualize { angle, frequency } => {
            let params = SimParams {
                angle: angle * TAU / 360.,
                frequency,
                visualize: true,
            };
            simulate(params, &setup);
        }
    }

    Ok(())
}
