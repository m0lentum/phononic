//! Data that is constant between different simulations
//! (mesh, operators, ...)

use dexterior as dex;
use itertools::izip;

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
    // interpolation operator that accounts for the periodic boundary condition
    pub periodic_interp: dex::MatrixOperator<Pressure, dex::Cochain<0, dex::Primal>>,
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
    pub measurement_tris: dex::Subset<2, dex::Primal>,
    pub measurement_edges: dex::Subset<1, dex::Primal>,
    pub top_edges: dex::Subset<1, dex::Primal>,
    // edges along which the domain is periodic
    pub left_edges: dex::Subset<1, dex::Primal>,
    pub left_verts: dex::Subset<0, dex::Primal>,
    pub right_edges: dex::Subset<1, dex::Primal>,
    pub right_verts: dex::Subset<0, dex::Primal>,
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
        let top_verts = mesh.get_subset::<0>("991").expect("Subset not found");
        let right_edges = mesh.get_subset::<1>("992").expect("Subset not found");
        let right_verts = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&right_edges)
                .flat_map(|e| e.boundary().map(|(_, v)| v)),
        );
        let left_edges = mesh.get_subset::<1>("993").expect("Subset not found");
        let left_verts = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&left_edges)
                .flat_map(|e| e.boundary().map(|(_, v)| v)),
        );
        let side_edges = right_edges.union(&left_edges);

        let mut layer_boundary_edges = layers[0].boundary.clone();
        for layer in layers.iter().skip(1) {
            layer_boundary_edges = layer_boundary_edges.union(&layer.boundary);
        }
        layer_boundary_edges = layer_boundary_edges.difference(&boundary_edges);

        // source terms are applied over a band of triangles along the bottom
        // and measurements performed along a similar band at the top
        let source_tris = dex::Subset::from_predicate(&mesh, |tri| {
            tri.vertex_indices()
                .any(|idx| bottom_verts.indices.contains(idx))
        });
        let source_edges = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&source_tris)
                .flat_map(|tri| tri.boundary().map(|(_, e)| e)),
        );
        let measurement_tris = dex::Subset::from_predicate(&mesh, |tri| {
            tri.vertex_indices()
                .any(|idx| top_verts.indices.contains(idx))
        });
        let measurement_edges = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&measurement_tris)
                .flat_map(|tri| tri.boundary().map(|(_, e)| e)),
        );

        let subsets = Subsets {
            layers,
            boundary_edges,
            layer_boundary_edges,
            bottom_edges,
            top_edges,
            left_edges,
            left_verts,
            right_edges,
            right_verts,
            side_edges,
            source_tris,
            source_edges,
            measurement_tris,
            measurement_edges,
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

        // projection implementing periodic domain from right to left.
        // first we need to find which edges match;
        // existence is guaranteed by the periodic constraint applied with gmsh.
        // we first find matching vertices (also needed for adjusting the interpolation operator)
        // and then derive matching edges from these
        let vertex_map: Vec<(usize, usize)> = mesh
            .simplices_in(&subsets.left_verts)
            .map(|left| {
                let left_y = left.vertices().next().unwrap().y;
                let right = mesh
                    .simplices_in(&subsets.right_verts)
                    .find(|right| {
                        let right_y = right.vertices().next().unwrap().y;
                        (left_y - right_y).abs() < 1e-6
                    })
                    .expect("Periodic mesh edges weren't set up right");
                (left.index(), right.index())
            })
            .collect();

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
                let left_indices = {
                    let mut indices = left.vertex_indices();
                    [indices.next().unwrap(), indices.next().unwrap()]
                };

                let right_indices = left_indices.map(|li| {
                    vertex_map
                        .iter()
                        .find(|(l, _)| *l == li)
                        .map(|(_, r)| *r)
                        .unwrap()
                });

                let right = mesh
                    .simplices_in(&subsets.right_edges)
                    .find(|e| e.vertex_indices().all(|i| right_indices.contains(&i)))
                    .unwrap();

                // whether or not vertices are in the same order on both sides
                // tells us the relative orientation
                let orientation = if right.vertex_indices().next().unwrap() == right_indices[0] {
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

        // interpolation operator with correction at the periodic edges
        // based on the fact that exactly half
        // of the dual volume is on the other side
        // (gmsh keeps triangle shapes consistent)
        let interp = dex::interpolate::dual_to_primal(&mesh);

        let vert_count = mesh.simplex_count::<0>();
        let dual_vert_count = mesh.simplex_count::<2>();
        // values that will be added to the interpolation matrix
        // coming from dual volumes on the opposite edge
        // (these modify the sparsity pattern
        // so we can't easily do this in place on the original matrix)
        let mut projected_coefs = dex::nas::CooMatrix::new(vert_count, dual_vert_count);
        for &(l, r) in &vertex_map {
            // here we apply the weights of dual vertices on the right
            // to the vertices on the left edge
            let r_coef_row = interp.mat.row(r);
            for (r_idx, r_val) in izip!(r_coef_row.col_indices(), r_coef_row.values()) {
                projected_coefs.push(l, *r_idx, *r_val);
            }
            // ..and vice versa
            let l_coef_row = interp.mat.row(l);
            for (l_idx, l_val) in izip!(l_coef_row.col_indices(), l_coef_row.values()) {
                projected_coefs.push(r, *l_idx, *l_val);
            }
        }
        let projected_coefs = dex::nas::CsrMatrix::from(&projected_coefs);
        let periodic_interp: dex::MatrixOperator<
            dex::Cochain<0, dex::Dual>,
            dex::Cochain<0, dex::Primal>,
        > = dex::MatrixOperator::from(interp.mat + projected_coefs);

        // the previous modification creates doubled weights,
        // correct these with a diagonal matrix
        let mut weight_correction = dex::na::DVector::from_vec(vec![1.; mesh.simplex_count::<0>()]);
        for vert in mesh
            .simplices_in(&subsets.left_verts)
            .chain(mesh.simplices_in(&subsets.right_verts))
        {
            weight_correction[vert.index()] = 0.5;
        }
        type Interpolated = dex::Cochain<0, dex::Primal>;
        let weight_correction: dex::DiagonalOperator<Interpolated, Interpolated> =
            dex::DiagonalOperator::from(weight_correction);

        let periodic_interp = weight_correction * periodic_interp;

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
            q_step_interp: (dt * inv_density_scaling.clone() * mesh.d() * periodic_interp.clone()),
            w_step: dt * mu_scaling.clone() * mesh.star() * mesh.d(),
            v_step: dt
                * periodic_projection.clone()
                * inv_density_scaling.clone()
                * mesh.star()
                * mesh.d(),
            v_step_interp: (dt * inv_density_scaling.clone() * mesh.d() * periodic_interp.clone()),
            inv_density_scaling,
            mu_scaling,
            stiffness_scaling,
            periodic_projection,
            periodic_interp,
        };

        Ok(Self {
            mesh,
            subsets,
            ops,
            dt,
        })
    }
}
