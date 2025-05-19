//! Data that is constant between different simulations
//! (mesh, operators, ...)

use dexterior as dex;

use super::simulate::{Flux, Pressure, Shear, Velocity};

pub struct Setup {
    pub mesh: dex::SimplicialMesh<2>,
    pub subsets: Subsets,
    pub ops: Ops,
    /// Timestep is constant but dependent on minimum mesh edge length
    pub dt: f64,
}

pub struct Ops {
    pub p_step: dex::Op<Flux, Pressure>,
    pub q_step: dex::Op<Pressure, Flux>,
    // operators that apply pressure interpolated onto primal vertices
    // into the shear wave and shear into the pressure wave.
    // these only have an effect at material boundaries
    pub q_step_interp: dex::Op<Shear, Flux>,
    pub w_step: dex::Op<Velocity, Shear>,
    pub v_step: dex::Op<Shear, Velocity>,
    pub v_step_interp: dex::Op<Pressure, Velocity>,
    // material scaling operators
    // are useful for looking up material parameters later
    pub mu_scaling: dex::DiagonalOperator<Shear, Shear>,
    pub stiffness_scaling: dex::DiagonalOperator<Pressure, Pressure>,
    pub inv_density_scaling: dex::DiagonalOperator<Velocity, Velocity>,
    // projection matrix that matches the right edge to the left
    pub periodic_projection: dex::MatrixOperator<Velocity, Velocity>,
}

pub struct Subsets {
    /// parts of the mesh made of different materials
    pub layers: Vec<MaterialArea>,
    /// entire outer boundary of the mesh
    pub boundary_edges: dex::Subset<1, dex::Primal>,
    /// boundaries between layers, not including outer boundary
    pub layer_boundary_edges: dex::Subset<1, dex::Primal>,
    pub bottom_edges: dex::Subset<1, dex::Primal>,
    /// region where source terms are applied
    pub source_tris: dex::Subset<2, dex::Primal>,
    pub source_edges: dex::Subset<1, dex::Primal>,
    pub top_edges: dex::Subset<1, dex::Primal>,
    // edges along which the domain is periodic
    pub left_edges: dex::Subset<1, dex::Primal>,
    pub right_edges: dex::Subset<1, dex::Primal>,
    pub side_edges: dex::Subset<1, dex::Primal>,
}

pub struct MaterialArea {
    pub edges: dex::Subset<1, dex::Primal>,
    pub tris: dex::Subset<2, dex::Primal>,
    pub boundary: dex::Subset<1, dex::Primal>,
    // lamè coefficients and other parameters of the material in this area
    // (stiffness = λ + 2μ)
    pub mu: f64,
    pub density: f64,
    pub stiffness: f64,
    pub p_wave_speed: f64,
    pub s_wave_speed: f64,
}

impl Setup {
    pub fn new() -> Result<Self, dex::gmsh::GmshError> {
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
        let bottom_verts = mesh.get_subset::<0>("990").expect("Subset not found");
        let top_edges = mesh.get_subset::<1>("991").expect("Subset not found");
        let right_edges = mesh.get_subset::<1>("992").expect("Subset not found");
        let left_edges = mesh.get_subset::<1>("993").expect("Subset not found");
        let side_edges = right_edges.union(&left_edges);

        let mut layer_boundary_edges = layers[0].boundary.clone();
        for layer in layers.iter().skip(1) {
            layer_boundary_edges = layer_boundary_edges.union(&layer.boundary);
        }
        layer_boundary_edges = layer_boundary_edges.difference(&boundary_edges);

        // source terms are applied over a band of triangles along the bottom
        let source_tris = dex::Subset::from_predicate(&mesh, |tri| {
            tri.vertex_indices()
                .any(|idx| bottom_verts.indices.contains(idx))
        });
        let source_edges = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&source_tris)
                .flat_map(|tri| tri.boundary().map(|(_, e)| e)),
        );

        let subsets = Subsets {
            layers,
            boundary_edges,
            layer_boundary_edges,
            bottom_edges,
            top_edges,
            left_edges,
            right_edges,
            side_edges,
            source_tris,
            source_edges,
        };

        // other constant parameters

        let dt = 0.2
            * mesh
                .simplices::<1>()
                .map(|s| s.volume())
                .min_by(f64::total_cmp)
                .unwrap();

        // operators

        // spatially varying scaling factors
        let stiffness_scaling = mesh.scaling_dual(|s| {
            let l = subsets
                .layers
                .iter()
                .find(|l| l.tris.contains(s.dual()))
                .unwrap();
            l.stiffness
        });
        let inv_density_scaling = mesh.scaling(|s| {
            let l = subsets.layers.iter().find(|l| l.edges.contains(s)).unwrap();
            1. / l.density
        });
        let mu_scaling = mesh.scaling_dual(|s| {
            let l = subsets
                .layers
                .iter()
                .find(|l| l.tris.contains(s.dual()))
                .unwrap();
            l.mu
        });

        // interpolation operator for coupling between the equations
        let interp = dex::interpolate::dual_to_primal(&mesh);

        // projection implementing periodic domain from right to left.
        // first we need to find which edges match;
        // existence is guaranteed by the periodic constraint applied with gmsh
        #[derive(Clone, Copy, Debug)]
        struct EdgeMatch {
            // indices of the matching edges
            left: usize,
            right: usize,
            // the relative orientation is needed
            // to make sure signs match after projection
            orientation: i8,
        }
        let edge_map: Vec<EdgeMatch> = mesh
            .simplices_in(&subsets.left_edges)
            .map(|left| {
                // handling the vertices of a simplex via iterator like this is annoying..
                // unfortunately the best alternative (arrays instead of iterators)
                // requires const generic arithmetic which Rust doesn't have yet
                let ys = {
                    let mut verts = left.vertices();
                    [verts.next().unwrap().y, verts.next().unwrap().y]
                };

                // the y coordinates should be approximately equal on both edges
                // (maybe even exactly equal? not sure how gmsh does this)
                // so padding the range slightly and finding the edge on the other side
                // that fits in this range should do the trick
                let y_range = (f64::min(ys[0], ys[1]) - 0.001)..(f64::max(ys[0], ys[1]) + 0.001);
                let right = mesh
                    .simplices_in(&subsets.right_edges)
                    .find(|right| right.vertices().all(|v| y_range.contains(&v.y)))
                    .expect("Periodic mesh edges weren't set up right");

                let first_right = right.vertices().next().unwrap();
                let orientation = if (ys[0] - first_right.y).abs() < left.volume() / 2. {
                    // first vertex has same y coordinate => orientations match
                    // (this seems to be the case every time with gmsh,
                    // but leaving this in just in case)
                    1
                } else {
                    -1
                };

                EdgeMatch {
                    left: left.index(),
                    right: right.index(),
                    orientation,
                }
            })
            .collect();

        let edge_count = mesh.simplex_count::<1>();
        let mut proj_coo = dex::nas::CooMatrix::new(edge_count, edge_count);
        // identity diagonal
        for i in 0..edge_count {
            proj_coo.push(i, i, 1.);
        }
        for em in &edge_map {
            proj_coo.push(em.left, em.right, em.orientation as f64);
            proj_coo.push(em.right, em.left, em.orientation as f64);
        }
        let periodic_projection = dex::MatrixOperator::from(dex::nas::CsrMatrix::from(&proj_coo));

        let ops = Ops {
            p_step: dt * stiffness_scaling.clone() * mesh.star() * mesh.d(),
            q_step: (dt
                * periodic_projection.clone()
                * inv_density_scaling.clone()
                * mesh.star()
                * mesh.d())
            .exclude_subset(&subsets.top_edges),
            // the interpolated operators have no effect everywhere except at material boundaries
            // (and break at the mesh boundary due to truncated dual cells)
            // so we can safely exclude the rest
            // (TODO: this isn't actually true, revise this whole idea)
            q_step_interp: (dt
                * periodic_projection.clone()
                * inv_density_scaling.clone()
                * mesh.d()
                * interp.clone())
            .exclude_subset(&mesh.subset_complement(&subsets.layer_boundary_edges)),
            w_step: dt * mu_scaling.clone() * mesh.star() * mesh.d(),
            v_step: dt
                * periodic_projection.clone()
                * inv_density_scaling.clone()
                * mesh.star()
                * mesh.d(),
            v_step_interp: (dt
                * periodic_projection.clone()
                * inv_density_scaling.clone()
                * mesh.d()
                * interp)
                .exclude_subset(&mesh.subset_complement(&subsets.layer_boundary_edges)),
            inv_density_scaling,
            mu_scaling,
            stiffness_scaling,
            periodic_projection,
        };

        Ok(Self {
            mesh,
            subsets,
            ops,
            dt,
        })
    }
}
