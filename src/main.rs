use clap::Parser;
use dexterior as dex;
use dexterior::visuals as dv;
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Minimum angle of incidence.
    #[arg(long, default_value_t = 0.)]
    angle_min: f64,
    /// Maximum angle of incidence.
    #[arg(long, default_value_t = 90.)]
    angle_max: f64,
    /// Display the simulation in a render window.
    #[arg(long)]
    visualize: bool,
}

type Pressure = dex::Cochain<0, dex::Dual>;
type Flux = dex::Cochain<1, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;
type Shear = dex::Cochain<0, dex::Dual>;

#[derive(Clone, Debug)]
struct State {
    t: f64,
    p: Pressure,
    q: Flux,
    w: Shear,
    v: Velocity,
    measurements: Measurements,
    // false to draw the shear potential, true to draw pressure
    // (these parameters stored in state to enable keyboard controls
    // when running in visual mode)
    draw_pressure: bool,
    draw_arrows: bool,
}

impl dv::AnimationState for State {}

#[derive(Clone, Debug, Default, serde::Serialize)]
struct Measurements {
    // energy measured at the top horizontal edge for each timestep
    transmitted_energy: VecDeque<f64>,
}

struct Ops {
    p_step: dex::Op<Flux, Pressure>,
    q_step: dex::Op<Pressure, Flux>,
    // operators that apply pressure interpolated onto primal vertices
    // into the shear wave and shear into the pressure wave.
    // these only have an effect at material boundaries
    q_step_interp: dex::Op<Shear, Flux>,
    w_step: dex::Op<Velocity, Shear>,
    v_step: dex::Op<Shear, Velocity>,
    v_step_interp: dex::Op<Pressure, Velocity>,
}

struct MaterialArea {
    edges: dex::Subset<1, dex::Primal>,
    tris: dex::Subset<2, dex::Primal>,
    boundary: dex::Subset<1, dex::Primal>,
    // lamè coefficients and other parameters of the material in this area
    // (stiffness = λ + 2μ)
    mu: f64,
    density: f64,
    stiffness: f64,
    p_wave_speed: f64,
    s_wave_speed: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli: Cli = Cli::parse();

    let msh_bytes = include_bytes!("./meshes/2d_layered.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    // spatially varying material parameters from mesh physical groups

    let mut layers: Vec<MaterialArea> = Vec::new();
    // loop until no more layers found
    // instead of hardcoding layer count
    // so that we can easily change this via gmsh parameters
    let mut layer = 1;
    loop {
        let group_id = format!("{layer}");
        let (Some(edges), Some(tris)) = (
            mesh.get_subset::<1>(&group_id),
            mesh.get_subset::<2>(&group_id),
        ) else {
            break;
        };

        let boundary = mesh.subset_boundary(&tris);

        let lambda = 1.;
        let mu = 1.;
        let density = layer as f64;
        let stiffness = lambda + 2. * mu;

        let p_wave_speed = f64::sqrt(stiffness / density);
        let s_wave_speed = f64::sqrt(mu / density);

        layers.push(MaterialArea {
            edges,
            tris,
            boundary,
            mu,
            density,
            stiffness,
            p_wave_speed,
            s_wave_speed,
        });
        layer += 1;
    }

    let boundary_edges = mesh.boundary::<1>();
    let bottom_edges = mesh.get_subset::<1>("990").expect("Subset not found");
    let top_edges = mesh.get_subset::<1>("991").expect("Subset not found");

    let mut layer_boundary_edges = layers[0].boundary.clone();
    for layer in layers.iter().skip(1) {
        layer_boundary_edges = layer_boundary_edges.union(&layer.boundary);
    }
    layer_boundary_edges = layer_boundary_edges.difference(&boundary_edges);

    // other parameters

    // timestep determined by shortest edge in the mesh
    let dt = 0.2
        * mesh
            .simplices::<1>()
            .map(|s| s.volume())
            .min_by(f64::total_cmp)
            .unwrap();
    let color_map_range = 2.;

    // angle of wave propagation between 0 and 90 degrees
    // (0 is horizontal, 90 vertical;
    // must be in this interval for source terms to work correctly)
    // TODO: dispatch a range of angles and frequencies based on given CLI arguments
    let wave_angle_deg = 60.;
    let wave_angle = wave_angle_deg * TAU / 360.;
    let wave_dir = dex::Vec2::new(wave_angle.cos(), wave_angle.sin());

    // both pressure and shear wave have the same frequency to simplify measurements
    // (since we need to average over a wave period).
    // frequency in oscillations/second and radians/second
    let frequency_osc_s = 0.5;
    let frequency_rad_s = frequency_osc_s * TAU;
    let wave_period = 1. / frequency_osc_s;
    let timesteps_per_period = (wave_period / dt).round() as usize;

    let pressure_wavenumber = frequency_rad_s / layers[0].p_wave_speed;
    let pressure_wave_vector = pressure_wavenumber * wave_dir;
    let shear_wavenumber = frequency_rad_s / layers[0].s_wave_speed;
    let shear_wave_vector = shear_wavenumber * wave_dir;

    // operators

    // spatially varying scaling factors
    let stiffness_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.stiffness
    });
    let density_scaling = mesh.scaling(|s| {
        let l = layers.iter().find(|l| l.edges.contains(s)).unwrap();
        1. / l.density
    });
    let mu_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.mu
    });

    let interp = dex::interpolate::dual_to_primal(&mesh);

    let ops = Ops {
        p_step: dt * stiffness_scaling * mesh.star() * mesh.d(),
        q_step: dt * density_scaling.clone() * mesh.star() * mesh.d(),
        // the interpolated operators have no effect everywhere except at material boundaries
        // (and break at the mesh boundary due to truncated dual cells)
        // so we can safely exclude the rest
        q_step_interp: (dt * density_scaling.clone() * mesh.d() * interp.clone())
            .exclude_subset(&mesh.subset_complement(&layer_boundary_edges)),
        w_step: dt * mu_scaling * mesh.star() * mesh.d(),
        v_step: dt * density_scaling.clone() * mesh.star() * mesh.d(),
        v_step_interp: (dt * density_scaling.clone() * mesh.d() * interp)
            .exclude_subset(&mesh.subset_complement(&layer_boundary_edges)),
    };

    // source terms

    // for angled pulses we need to compute when it reaches the given point
    // to avoid discontinuities
    let p_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / layers[0].p_wave_speed;
        t >= start_time
    };
    let s_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / layers[0].s_wave_speed;
        t >= start_time
    };

    let pressure_source_vec = |pos: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        if !p_pulse_active(pos, t) {
            return 0.;
        }
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel =
            -pressure_wave_vector * f64::sin(frequency_rad_s * t - pressure_wave_vector.dot(&pos));
        vel.dot(&normal)
    };
    let shear_source_vec = |pos: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        if !s_pulse_active(pos, t) {
            return 0.;
        }
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel = -shear_wave_vector * f64::sin(frequency_rad_s * t - shear_wave_vector.dot(&pos));
        vel.dot(&normal)
    };

    // run simulation

    let state = State {
        t: 0.,
        p: mesh.new_zero_cochain(),
        q: mesh.new_zero_cochain(),
        w: mesh.new_zero_cochain(),
        v: mesh.new_zero_cochain(),
        measurements: Measurements::default(),
        draw_pressure: false,
        draw_arrows: false,
    };

    let initial_state = state.clone();

    // step function separated from animation
    // so we can choose between computing with or without visuals
    let step = |state: &mut State| {
        state.q += &ops.q_step * &state.p + &ops.q_step_interp * &state.w;
        state.v += &ops.v_step * &state.w + &ops.v_step_interp * &state.p;

        // sources applied to the flux and velocity vectors
        mesh.integrate_overwrite(
            &mut state.q,
            &bottom_edges,
            dex::quadrature::GaussLegendre6(|p, d| pressure_source_vec(p, d, state.t)),
        );
        mesh.integrate_overwrite(
            &mut state.v,
            &bottom_edges,
            dex::quadrature::GaussLegendre6(|p, d| shear_source_vec(p, d, state.t)),
        );

        // absorbing boundary
        for layer in &layers {
            let edges_here = boundary_edges
                .intersection(&layer.edges)
                .difference(&bottom_edges);

            for edge in mesh.simplices_in(&edges_here) {
                let length = edge.volume();
                // pressure from the adjacent dual vertex
                let (orientation, tri) = edge.coboundary().next().unwrap();
                state.q[edge] = -state.p[tri.dual()] * length * orientation as f64
                    / (layer.p_wave_speed * layer.density);
                state.v[edge] = -state.w[tri.dual()] * length * orientation as f64
                    / (layer.s_wave_speed * layer.density);
            }
        }

        state.p += &ops.p_step * &state.q;
        state.w += &ops.w_step * &state.v;

        state.t += dt;

        // measurements

        let transmitted_kinetic_energy: f64 = mesh
            .simplices_in(&top_edges)
            // since density_scaling contains inverses of density,
            // the division here multiplies by local density
            .map(|e| 0.5 * (state.v[e].powi(2) + state.q[e].powi(2)) / density_scaling[e])
            .sum();
        if state.measurements.transmitted_energy.len() >= timesteps_per_period {
            state.measurements.transmitted_energy.pop_front();
        }
        state
            .measurements
            .transmitted_energy
            .push_back(transmitted_kinetic_energy);
    };

    if !cli.visualize {
        let mut state = state;
        const MAX_STEPS: usize = 1000;
        for _step in 0..MAX_STEPS {
            step(&mut state);
            // TODO stop when steady state reached, return measurements
        }

        return Ok(());
    }

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-color_map_range..color_map_range),
            ..Default::default()
        },
        dt,
        state,
        step,
        draw: |state, draw| {
            draw.axes_2d(dv::AxesParams::default());

            if state.draw_pressure {
                draw.triangle_colors_dual(&state.p);
            } else {
                draw.triangle_colors_dual(&state.w);
            }

            draw.wireframe(dv::WireframeParams {
                width: dv::LineWidth::ScreenPixels(1.),
                ..Default::default()
            });
            for (idx, layer) in layers.iter().enumerate() {
                // layer boundaries with thicker lines
                draw.wireframe_subset(
                    dv::WireframeParams {
                        width: dv::LineWidth::ScreenPixels(3.),
                        ..Default::default()
                    },
                    &layer.boundary,
                );

                // text for material parameters
                let layer_height = PI / layers.len() as f64;
                let lambda = layer.stiffness - 2. * layer.mu;
                draw.text(dv::TextParams {
                    text: &format!("λ: {}\nμ: {}\nρ: {}", lambda, layer.mu, layer.density),
                    position: dex::Vec2::new(TAU + 0.1, (idx as f64 + 0.5) * layer_height),
                    anchor: dv::TextAnchor::MidLeft,
                    font_size: 20.,
                    line_height: 24.,
                    ..Default::default()
                });
            }

            if state.draw_arrows {
                if state.draw_pressure {
                    draw.flux_arrows(&state.q, dv::ArrowParams::default());
                } else {
                    draw.velocity_arrows(&state.v, dv::ArrowParams::default());
                }
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

            let avg_energy = state.measurements.transmitted_energy.iter().sum::<f64>()
                / state.measurements.transmitted_energy.len() as f64;
            draw.text(dv::TextParams {
                text: &format!("Avg. energy transmitted: {:.3}", avg_energy),
                position: dex::Vec2::new(PI, PI),
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
    })?;

    Ok(())
}
