use dexterior as dex;
use dexterior::visuals as dv;
use itertools::Itertools;
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

use super::Setup;

// types of simulation variables

pub type Pressure = dex::Cochain<0, dex::Dual>;
pub type Flux = dex::Cochain<1, dex::Primal>;
pub type Shear = dex::Cochain<0, dex::Primal>;

// constants determining when a steady state is deemed to be reached:
// transmitted energy measurements stay within this range
// for this many wave periods
const STEADY_STATE_RANGE: f64 = 0.001;
const STEADY_STATE_PERIODS: usize = 2;

/// Maximum number of timesteps to simulate,
/// as a failsafe in case the simulation does not converge
const MAX_STEPS: usize = 50000;

/// Parameters that vary between simulations.
#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    /// Angle of incidence of the incoming plane wave,
    /// measured in radians, between 0 and π/2.
    pub angle: f64,
    /// Frequency of the incoming wave in radians per second.
    pub frequency: f64,
    /// Whether to apply an incident pressure wave.
    pub p_source_active: bool,
    /// Whether to apply an incident shear wave.
    pub w_source_active: bool,
    /// Whether to visualize the simulation.
    pub visualize: bool,
}

/// Varying state within a single simulation.
#[derive(Clone, Debug)]
pub struct State {
    pub t: f64,
    pub p: Pressure,
    pub q: Flux,
    pub w: Shear,
    pub measurements: MeasurementData,
    // false to draw the shear potential, true to draw pressure
    // (these parameters stored in state to enable keyboard controls
    // when running in visual mode)
    pub draw_pressure: bool,
    pub draw_arrows: bool,
}

impl dv::AnimationState for State {}

/// Transient data needed to compute our measurements
#[derive(Clone, Debug, Default)]
pub struct MeasurementData {
    /// Energy measured at the top edge for each timestep of the past wave period
    pub transmitted: VecDeque<f64>,
    /// Average transmitted energies over time,
    /// used to detect when we've reached a steady state
    pub transmitted_averages: Vec<f64>,
    /// Total potential energies over the past wave period.
    pub total: VecDeque<f64>,
}

impl MeasurementData {
    /// Check if the transmitted energy values haven't changed
    /// for long enough to be considered converged.
    pub fn has_converged(&self, step_count: usize) -> bool {
        let itertools::MinMaxResult::MinMax(min, max) = self
            .transmitted_averages
            .iter()
            .rev()
            .take(step_count)
            .minmax()
        else {
            return false;
        };

        max - min < STEADY_STATE_RANGE && *max > 0.1
    }
}

/// Final measurements derived from MeasurementData,
/// returned after reaching a steady state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measurements {
    /// Average potential energy density
    /// at the top edge of the structure over a wave period.
    pub transmitted: f64,
    /// Average potential energy density in the entire system over a wave period.
    pub total: f64,
}

impl From<&MeasurementData> for Measurements {
    fn from(data: &MeasurementData) -> Self {
        Self {
            transmitted: data.transmitted.iter().sum::<f64>() / data.transmitted.len() as f64,
            total: data.total.iter().sum::<f64>() / data.total.len() as f64,
        }
    }
}

pub fn simulate(params: SimParams, setup: &Setup) -> Measurements {
    //
    // parameters
    //

    let wave_dir = dex::Vec2::new(params.angle.cos(), params.angle.sin());
    // both pressure and shear wave have the same frequency to simplify measurements
    // (since we need to average over a wave period).
    // frequency in radians/second and oscillations/second
    let wave_period = TAU / params.frequency;
    let timesteps_per_period = (wave_period / setup.dt).round() as usize;

    let pressure_wavenumber = params.frequency / setup.subsets.layers[0].p_wave_speed;
    let pressure_wave_vector = pressure_wavenumber * wave_dir;
    let shear_wavenumber = params.frequency / setup.subsets.layers[0].s_wave_speed;
    let shear_wave_vector = shear_wavenumber * wave_dir;

    //
    // source terms
    //

    // for angled pulses we need to compute when it reaches the given point
    // to avoid discontinuities
    let p_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / setup.subsets.layers[0].p_wave_speed;
        t >= start_time
    };
    let s_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / setup.subsets.layers[0].s_wave_speed;
        t >= start_time
    };

    let pressure_source = |pos: dex::Vec2, t: f64| -> f64 {
        if !p_pulse_active(pos, t) {
            return 0.;
        }
        f64::sin(params.frequency * t - pressure_wave_vector.dot(&pos))
    };
    let shear_source = |pos: dex::Vec2, t: f64| -> f64 {
        if !s_pulse_active(pos, t) {
            return 0.;
        }
        f64::sin(params.frequency * t - shear_wave_vector.dot(&pos))
    };

    //
    // run simulation
    //

    let state = State {
        t: 0.,
        p: setup.mesh.new_zero_cochain(),
        q: setup.mesh.new_zero_cochain(),
        w: setup.mesh.new_zero_cochain(),
        measurements: MeasurementData::default(),
        draw_pressure: false,
        draw_arrows: false,
    };

    let initial_state = state.clone();

    // step function separated from animation
    // so we can choose between computing with or without visuals
    let step = |state: &mut State| {
        state.q += &setup.ops.q_step_p * &state.p;
        state.q += &setup.ops.q_step_w * &state.w;
        // absorbing boundary for pressure wave
        let absorpt = &setup.ops.p_absorber * &state.p;
        for edge in setup.mesh.simplices_in(&setup.subsets.top_edges) {
            state.q[edge] = absorpt[edge];
        }

        state.p += &setup.ops.p_step * &state.q;
        state.w += &setup.ops.w_step * &state.q;
        // absorbing boundary for shear wave
        state.w += &setup.ops.w_absorber * &state.w;

        state.t += setup.dt;

        // sources applied to pressure and shear at the bottom layer
        if params.p_source_active {
            for tri in setup.mesh.simplices_in(&setup.subsets.source_tris) {
                state.p[tri.dual()] = pressure_source(tri.circumcenter(), state.t);
            }
        }
        if params.w_source_active {
            for vert in setup.mesh.simplices_in(&setup.subsets.source_verts) {
                state.w[vert] = shear_source(vert.vertices().next().unwrap(), state.t);
            }
        }

        // measurements

        // division by density is to cancel these variables' values being scaled by density
        let pressure_pot_energy =
            |dv| 0.5 * setup.ops.stiffness_scaling[dv] * state.p[dv].powi(2) * dv.dual().volume();
        let shear_pot_energy =
            |vert| 0.5 * setup.ops.mu_scaling[vert] * state.w[vert].powi(2) * vert.dual_volume();

        let transmitted_pressure_pot = setup
            .mesh
            .simplices_in(&setup.subsets.measurement_tris)
            .map(|t| pressure_pot_energy(t.dual()))
            .sum::<f64>()
            / setup.stats.primal_measurement_area;
        let total_pressure_pot = setup
            .mesh
            .dual_cells::<0>()
            .map(pressure_pot_energy)
            .sum::<f64>()
            / setup.stats.total_mesh_area;
        let transmitted_shear_pot = setup
            .mesh
            .simplices_in(&setup.subsets.measurement_verts)
            .map(shear_pot_energy)
            .sum::<f64>()
            / setup.stats.dual_measurement_area;
        let total_shear_pot = setup
            .mesh
            .simplices::<0>()
            .map(shear_pot_energy)
            .sum::<f64>()
            / setup.stats.total_mesh_area;

        let transmitted_sum = transmitted_pressure_pot + transmitted_shear_pot;
        if state.measurements.transmitted.len() >= timesteps_per_period {
            state.measurements.transmitted.pop_front();
        }
        state.measurements.transmitted.push_back(transmitted_sum);

        let total_sum = total_pressure_pot + total_shear_pot;
        if state.measurements.total.len() >= timesteps_per_period {
            state.measurements.total.pop_front();
        }
        state.measurements.total.push_back(total_sum);

        // update measurement history
        let ms = Measurements::from(&state.measurements);
        state.measurements.transmitted_averages.push(ms.transmitted);
    };

    //
    // run without visuals
    //

    if !params.visualize {
        let mut state = state;
        for _step in 0..MAX_STEPS {
            step(&mut state);
            if state
                .measurements
                .has_converged(timesteps_per_period * STEADY_STATE_PERIODS)
            {
                break;
            }
        }

        return Measurements::from(&state.measurements);
    }

    //
    // run with visuals
    //

    let mut window =
        dv::RenderWindow::new(dv::WindowParams::default()).expect("Failed to create window");
    window
        .run_animation(dv::Animation {
            mesh: &setup.mesh,
            params: dv::AnimationParams {
                color_map_range: Some(-1.0..1.0),
                ..Default::default()
            },
            dt: setup.dt,
            state,
            step,
            draw: |state, draw| {
                draw.axes_2d(dv::AxesParams::default());

                if state.draw_pressure {
                    draw.triangle_colors_dual(&state.p);
                } else {
                    draw.vertex_colors(&state.w);
                }

                draw.wireframe(dv::WireframeParams {
                    width: dv::LineWidth::ScreenPixels(1.),
                    ..Default::default()
                });
                draw.dual_wireframe(dv::WireframeParams {
                    width: dv::LineWidth::ScreenPixels(1.),
                    color: dv::palette::LinSrgb::new(0.02, 0.02, 0.02),
                });
                for (idx, layer) in setup.subsets.layers.iter().enumerate() {
                    // layer boundaries with thicker lines
                    draw.wireframe_subset(
                        dv::WireframeParams {
                            width: dv::LineWidth::ScreenPixels(3.),
                            ..Default::default()
                        },
                        &layer.boundary,
                    );

                    // text for material parameters
                    let layer_height = PI / setup.subsets.layers.len() as f64;
                    let lambda = layer.stiffness - 2. * layer.mu;
                    draw.text(dv::TextParams {
                        text: &format!("λ: {}\nμ: {}\nρ: {}", lambda, layer.mu, layer.density),
                        position: dex::Vec2::new(PI + 0.1, (idx as f64 + 0.5) * layer_height),
                        anchor: dv::TextAnchor::MidLeft,
                        font_size: 20.,
                        line_height: 24.,
                        ..Default::default()
                    });
                }
                draw.wireframe_subset(
                    dv::WireframeParams {
                        width: dv::LineWidth::ScreenPixels(2.),
                        color: dv::palette::named::DARKBLUE.into(),
                    },
                    &(setup
                        .subsets
                        .measurement_edges
                        .union(&setup.subsets.source_edges)),
                );
                draw.wireframe_subset(
                    dv::WireframeParams {
                        width: dv::LineWidth::ScreenPixels(4.),
                        color: dv::palette::named::DARKRED.into(),
                    },
                    &setup.subsets.side_edges,
                );

                if state.draw_arrows {
                    draw.flux_arrows(&state.q, dv::ArrowParams::default());
                }

                draw.text(dv::TextParams {
                    text: if state.draw_pressure {
                        "Displaying pressure"
                    } else {
                        "Displaying shear"
                    },
                    position: dex::Vec2::new(0., PI),
                    anchor: dv::TextAnchor::BottomLeft,
                    font_size: 24.,
                    ..Default::default()
                });

                let measurements = Measurements::from(&state.measurements);
                let conv_text = if state
                    .measurements
                    .has_converged(timesteps_per_period * STEADY_STATE_PERIODS)
                {
                    " (converged!)"
                } else {
                    ""
                };
                draw.text(dv::TextParams {
                    text: &format!(
                        "Total: {:.3}\nTransmitted: {:.3}{}",
                        measurements.total, measurements.transmitted, conv_text
                    ),
                    position: dex::Vec2::new(PI / 2., PI),
                    anchor: dv::TextAnchor::BottomLeft,
                    font_size: 24.,
                    ..Default::default()
                });
            },
            on_key: |key, state| match key {
                dv::KeyCode::KeyP => {
                    state.draw_pressure = !state.draw_pressure;
                }
                dv::KeyCode::KeyA => {
                    state.draw_arrows = !state.draw_arrows;
                }
                dv::KeyCode::KeyR => {
                    *state = initial_state.clone();
                }
                _ => {}
            },
        })
        .unwrap();

    // we don't actually use the measurements in visual mode
    // but also don't want to complicate the API
    // so just return some nonsense
    Measurements::from(&initial_state.measurements)
}
