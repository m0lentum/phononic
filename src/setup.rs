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
    pub q_step_p: dex::Op<Pressure, Flux>,
    pub q_step_w: dex::Op<Shear, Flux>,
    pub p_absorber: dex::Op<Pressure, Flux>,
    pub w_step: dex::Op<Flux, Shear>,
    pub w_absorber: dex::DiagonalOperator<Shear, Shear>,
    // material scaling operators
    // are useful for looking up material parameters later
    pub mu_scaling: dex::DiagonalOperator<Shear, Shear>,
    pub stiffness_scaling: dex::DiagonalOperator<Pressure, Pressure>,
    pub inv_density_scaling: dex::DiagonalOperator<Velocity, Velocity>,
    // projection matrix that matches the right edge to the left
    pub periodic_proj_edge: dex::MatrixOperator<Velocity, Velocity>,
    pub periodic_proj_vert:
        dex::MatrixOperator<dex::Cochain<0, dex::Primal>, dex::Cochain<0, dex::Primal>>,
    // interpolation operator that accounts for the periodic boundary condition
    pub periodic_interp_dtp: dex::MatrixOperator<Pressure, dex::Cochain<0, dex::Primal>>,
    pub periodic_star_0_inv:
        dex::DiagonalOperator<dex::Cochain<2, dex::Dual>, dex::Cochain<0, dex::Primal>>,
    pub interp_ptd: dex::MatrixOperator<dex::Cochain<0, dex::Primal>, Pressure>,
}

pub struct Subsets {
    /// parts of the mesh made of different materials
    pub layers: Vec<MaterialArea>,
    /// entire outer boundary of the mesh
    pub boundary_edges: dex::Subset<1, dex::Primal>,
    /// boundaries between layers, not including outer boundary
    pub layer_boundary_edges: dex::Subset<1, dex::Primal>,
    /// edges and dual vertices of the triangles adjacent to layer boundaries.
    /// interpolated coupling operators are only applied here,
    /// as elsewhere they give approximately zero but with some error
    pub layer_boundary_adjacent_edges: dex::Subset<1, dex::Primal>,
    pub layer_boundary_adjacent_dvs: dex::Subset<0, dex::Dual>,
    pub bottom_edges: dex::Subset<1, dex::Primal>,
    /// region where source terms are applied
    pub source_tris: dex::Subset<2, dex::Primal>,
    pub source_edges: dex::Subset<1, dex::Primal>,
    pub source_verts: dex::Subset<0, dex::Primal>,
    pub measurement_tris: dex::Subset<2, dex::Primal>,
    pub measurement_edges: dex::Subset<1, dex::Primal>,
    pub top_edges: dex::Subset<1, dex::Primal>,
    pub top_verts: dex::Subset<0, dex::Primal>,
    // edges along which the domain is periodic
    pub left_edges: dex::Subset<1, dex::Primal>,
    pub left_verts: dex::Subset<0, dex::Primal>,
    pub right_edges: dex::Subset<1, dex::Primal>,
    pub right_verts: dex::Subset<0, dex::Primal>,
    pub side_edges: dex::Subset<1, dex::Primal>,
}

pub struct MaterialArea {
    pub vertices: dex::Subset<0, dex::Primal>,
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

        // the coefficient on dt depends on material parameters to an extent,
        // turn this down if you see instability after modifying those

        let dt = 0.2
            * mesh
                .simplices::<1>()
                .map(|s| s.volume())
                .min_by(f64::total_cmp)
                .unwrap();

        // spatially varying material parameters from mesh physical groups

        let mut layers: Vec<MaterialArea> = Vec::new();
        // loop until no more layers found
        // instead of hardcoding layer count
        // so that we can easily change this via gmsh parameters
        let mut layer = 1;
        loop {
            let group_id = format!("{layer}");
            let (Some(vertices), Some(edges), Some(tris)) = (
                mesh.get_subset::<0>(&group_id),
                mesh.get_subset::<1>(&group_id),
                mesh.get_subset::<2>(&group_id),
            ) else {
                break;
            };

            let boundary = tris.manifold_boundary(&mesh);

            let lambda = 1. + ((layer - 1) % 2) as f64;
            let mu = 1. - 0.5 * (layer % 2) as f64;
            let density = layer as f64;
            let stiffness = lambda + 2. * mu;

            let p_wave_speed = f64::sqrt(stiffness / density);
            let s_wave_speed = f64::sqrt(mu / density);

            layers.push(MaterialArea {
                vertices,
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
        let layer_boundary_verts = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&layer_boundary_edges)
                .flat_map(|e| e.boundary().map(|(_, v)| v)),
        );
        let layer_boundary_adjacent_dvs = dex::Subset::from_predicate_dual(&mesh, |dv| {
            dv.dual()
                .vertex_indices()
                .any(|vi| layer_boundary_verts.indices.contains(vi))
        });
        let layer_boundary_adjacent_edges = dex::Subset::from_simplex_iter(
            mesh.dual_cells_in(&layer_boundary_adjacent_dvs)
                .flat_map(|dv| dv.dual().boundary().map(|(_, e)| e)),
        );

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
        let source_verts = dex::Subset::from_simplex_iter(
            mesh.simplices_in(&source_edges)
                .flat_map(|e| e.boundary().map(|(_, v)| v)),
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
            layer_boundary_adjacent_edges,
            layer_boundary_adjacent_dvs,
            bottom_edges,
            top_edges,
            top_verts,
            left_edges,
            left_verts,
            right_edges,
            right_verts,
            side_edges,
            source_tris,
            source_edges,
            source_verts,
            measurement_tris,
            measurement_edges,
        };

        // operators

        // spatially varying scaling factors
        let stiffness_scaling = mesh.scaling_dual(|dvert| {
            let l = subsets
                .layers
                .iter()
                .find(|l| l.tris.contains(dvert.dual()))
                .unwrap();
            l.stiffness
        });
        let inv_density_scaling = mesh.scaling(|edge| {
            // for density scaling we must account for dual edges
            // that straddle the line between layers.
            // take the average density
            // weighted by portion of dual length in that layer
            let mut dens_sum = 0.;
            for (_, tri) in edge.coboundary() {
                let layer = subsets
                    .layers
                    .iter()
                    .find(|l| l.tris.contains(tri))
                    .unwrap();
                let dual_vol = (edge.circumcenter() - tri.circumcenter()).magnitude();
                dens_sum += dual_vol * layer.density;
            }
            1. / (dens_sum / edge.dual_volume())
        });
        let mu_scaling = mesh.scaling(|vert| {
            let vert_pos = vert.vertices().next().unwrap();
            // similarly here we need to account for how much dual volume
            // lies on each side of the layer boundary.
            // do this by summing "elementary dual simplices" corresponding to each edge
            let mut sum = 0.;
            let mut weight_sum = 0.;
            for (_, edge) in vert.coboundary() {
                for (_, tri) in edge.coboundary() {
                    let layer = subsets
                        .layers
                        .iter()
                        .find(|l| l.tris.contains(tri))
                        .unwrap();
                    let ds_edges = [
                        tri.circumcenter() - vert_pos,
                        edge.circumcenter() - vert_pos,
                    ];
                    let ds_area =
                        0.5 * (ds_edges[0].x * ds_edges[1].y - ds_edges[0].y * ds_edges[1].x).abs();

                    sum += layer.mu * ds_area;
                    weight_sum += ds_area;
                }
            }
            // if this fails it's probably because the mesh isn't well-centered
            assert!(
                (weight_sum - vert.dual_volume()).abs() < 1e-6,
                "Didn't compute dual area correctly for vertex at {}, computed: {}, real: {}",
                vert_pos,
                weight_sum,
                vert.dual_volume(),
            );
            sum / weight_sum
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
        let mut edge_proj_coo = dex::nas::CooMatrix::new(edge_count, edge_count);
        // identity diagonal
        for i in 0..edge_count {
            edge_proj_coo.push(i, i, 1.);
        }
        for em in &edge_map {
            edge_proj_coo.push(em.left, em.right, em.orientation as f64);
            edge_proj_coo.push(em.right, em.left, em.orientation as f64);
        }
        let periodic_proj_edge =
            dex::MatrixOperator::from(dex::nas::CsrMatrix::from(&edge_proj_coo));

        // same for vertices
        let vert_count = mesh.simplex_count::<0>();
        let mut vert_proj_coo = dex::nas::CooMatrix::new(vert_count, vert_count);
        for i in 0..vert_count {
            vert_proj_coo.push(i, i, 1.);
        }
        for &(l, r) in &vertex_map {
            vert_proj_coo.push(l, r, 1.);
            vert_proj_coo.push(r, l, 1.);
        }
        let periodic_proj_vert =
            dex::MatrixOperator::from(dex::nas::CsrMatrix::from(&vert_proj_coo));

        // interpolation operator with correction at the periodic edges
        // based on the fact that exactly half
        // of the dual volume is on the other side
        // (gmsh keeps triangle shapes consistent)
        let interp_dtp = dex::interpolate::dual_to_primal(&mesh);
        let interp_ptd = dex::interpolate::primal_to_dual(&mesh);

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
            let r_coef_row = interp_dtp.mat.row(r);
            for (r_idx, r_val) in izip!(r_coef_row.col_indices(), r_coef_row.values()) {
                projected_coefs.push(l, *r_idx, *r_val);
            }
            // ..and vice versa
            let l_coef_row = interp_dtp.mat.row(l);
            for (l_idx, l_val) in izip!(l_coef_row.col_indices(), l_coef_row.values()) {
                projected_coefs.push(r, *l_idx, *l_val);
            }
        }
        let projected_coefs = dex::nas::CsrMatrix::from(&projected_coefs);
        let periodic_interp: dex::MatrixOperator<
            dex::Cochain<0, dex::Dual>,
            dex::Cochain<0, dex::Primal>,
        > = dex::MatrixOperator::from(interp_dtp.mat + projected_coefs);

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

        let periodic_interp_dtp = weight_correction * periodic_interp;

        // periodic boundary also needs a modified 0-star
        let mut periodic_star_0 = mesh.star::<0, dex::Primal>();
        for &(l, r) in &vertex_map {
            let sum = periodic_star_0.diagonal[l] + periodic_star_0.diagonal[r];
            periodic_star_0.diagonal[l] = sum;
            periodic_star_0.diagonal[r] = sum;
        }
        // TODO: this could be a method in dexterior
        let periodic_star_0_inv =
            dex::DiagonalOperator::from(periodic_star_0.diagonal.map(|e| e.recip()));

        let mut periodic_star_1 = mesh.star::<1, dex::Primal>();
        for em in &edge_map {
            let sum = periodic_star_1.diagonal[em.left] + periodic_star_1.diagonal[em.right];
            periodic_star_1.diagonal[em.left] = sum;
            periodic_star_1.diagonal[em.right] = sum;
        }
        let periodic_star_1_dual =
            dex::DiagonalOperator::from(-periodic_star_1.diagonal.map(|e| e.recip()));

        //
        // operators implementing absorbing boundaries for pressure and shear waves
        //

        let top_layer = subsets.layers.iter().last().unwrap();

        let mut p_absorber_coo =
            dex::nas::CooMatrix::new(mesh.simplex_count::<1>(), mesh.dual_cell_count::<0>());
        for edge in mesh.simplices_in(&subsets.top_edges) {
            let length = edge.volume();
            // pressure from the adjacent dual vertex
            let (orientation, tri) = edge.coboundary().next().unwrap();
            p_absorber_coo.push(
                edge.index(),
                tri.index(),
                -length * orientation as f64 / (top_layer.p_wave_speed * top_layer.density),
            );
        }
        let p_absorber = dex::Op::from(dex::nas::CsrMatrix::from(&p_absorber_coo));

        let mut w_absorber_diag = dex::na::DVector::zeros(mesh.simplex_count::<0>());
        for vert in mesh.simplices_in(&subsets.top_verts) {
            // "completing the circulation"
            // for the dual cell at the boundary
            let closing_edge_len = 0.5
                * vert
                    .coboundary()
                    .filter(|(_, e)| subsets.top_edges.contains(*e))
                    .map(|(_, e)| e.volume())
                    .sum::<f64>();
            w_absorber_diag[vert.index()] =
                -dt * closing_edge_len * periodic_star_0_inv[vert] * top_layer.s_wave_speed;
        }
        let w_absorber = dex::DiagonalOperator::from(w_absorber_diag);

        let ops = Ops {
            p_step: dt * stiffness_scaling.clone() * mesh.star() * mesh.d(),
            q_step_p: (dt
                * periodic_proj_edge.clone()
                * inv_density_scaling.clone()
                * periodic_star_1_dual
                * mesh.d())
            .exclude_subset(&subsets.top_edges),
            q_step_w: (-dt * periodic_proj_edge.clone() * inv_density_scaling.clone() * mesh.d())
                .exclude_subset(&subsets.top_edges),
            p_absorber,
            w_step: dt
                * periodic_proj_vert.clone()
                * mu_scaling.clone()
                * periodic_star_0_inv.clone()
                * mesh.d()
                * mesh.star(),
            w_absorber,
            inv_density_scaling,
            mu_scaling,
            stiffness_scaling,
            periodic_proj_edge,
            periodic_proj_vert,
            periodic_interp_dtp,
            periodic_star_0_inv,
            interp_ptd,
        };

        Ok(Self {
            mesh,
            subsets,
            ops,
            dt,
        })
    }
}
