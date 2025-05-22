use clap::{Parser, Subcommand};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
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
        /// Minimum angle of incidence in degrees.
        #[arg(long, default_value_t = 10.)]
        angle_min: f64,
        /// Maximum angle of incidence in degrees.
        #[arg(long, default_value_t = 80.)]
        angle_max: f64,
        /// Number of simulations to run with different angle parameters.
        #[arg(long, default_value_t = 10)]
        angle_resolution: usize,
        /// Minimum frequency in radians per second.
        #[arg(long, default_value_t = 2.0)]
        freq_min: f64,
        /// Maximum frequency in radians per second.
        #[arg(long, default_value_t = 5.0)]
        freq_max: f64,
        /// Number of simulations to run with different frequency parameters.
        #[arg(long, default_value_t = 4)]
        freq_resolution: usize,
        /// Disable interpolated coupling between shear and pressure waves.
        #[arg(long)]
        uncoupled: bool,
    },
    /// Run only one simulation and visualize it.
    Visualize {
        /// Incident angle of the wave in degrees.
        #[arg(long, default_value_t = 60.0)]
        angle: f64,
        /// Frequency of the wave in radians per second.
        #[arg(long, default_value_t = 2.0)]
        frequency: f64,
        /// Disable interpolated coupling between shear and pressure waves.
        #[arg(long)]
        uncoupled: bool,
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
            uncoupled,
        } => {
            println!(
                "Running {} simulations on {} threads",
                angle_resolution * freq_resolution,
                rayon::current_num_threads()
            );

            let angle_step = (angle_max - angle_min) / (angle_resolution as f64 - 1.);
            let freq_step = (freq_max - freq_min) / (freq_resolution as f64 - 1.);

            let params: Vec<SimParams> =
                itertools::iproduct!(0..angle_resolution, 0..freq_resolution)
                    .map(|(angle_idx, freq_idx)| {
                        let angle = (angle_min + angle_idx as f64 * angle_step) * TAU / 360.;
                        let frequency = freq_min + freq_idx as f64 * freq_step;
                        SimParams {
                            angle,
                            frequency,
                            visualize: false,
                            coupled: !uncoupled,
                        }
                    })
                    .collect();

            params.par_iter().for_each(|sim_params| {
                let measurements = simulate(*sim_params, &setup);
                dbg!(sim_params);
                dbg!(measurements);
            });
        }
        Command::Visualize {
            angle,
            frequency,
            uncoupled,
        } => {
            let params = SimParams {
                angle: angle * TAU / 360.,
                frequency,
                visualize: true,
                coupled: !uncoupled,
            };
            simulate(params, &setup);
        }
    }

    Ok(())
}
