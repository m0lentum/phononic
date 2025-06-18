//! Data that is constant between different simulations
//! (mesh, operators, ...)

use dexterior as dex;

use super::simulate::{Flux, Pressure, Shear};

pub struct Setup {
    pub mesh: dex::SimplicialMesh<2>,
    pub subsets: Subsets,
    pub ops: Ops,
    /// Timestep is constant but dependent on minimum mesh edge length
    pub dt: f64,
    pub stats: Stats,
}

pub struct Ops {
    pub p_step: dex::Op<Flux, Pressure>,
    pub q_step_p: dex::Op<Pressure, Flux>,
    pub q_step_w: dex::Op<Shear, Flux>,
    pub p_absorber: dex::Op<Pressure, Flux>,
    pub w_step: dex::Op<Flux, Shear>,
    pub w_absorber: dex::DiagonalOperator<Shear, Shear>,
    /// Material scalings are incorporated into the step operators
    /// but also stored separately for easy lookup of material parameters later
    pub stiffness_scaling: dex::DiagonalOperator<Pressure, Pressure>,
    pub mu_scaling: dex::DiagonalOperator<Shear, Shear>,
    pub inv_density_scaling: dex::DiagonalOperator<Flux, Flux>,
}

pub struct Subsets {
    /// parts of the mesh made of different materials
    pub layers: Vec<MaterialArea>,
    /// region where source terms are applied
    pub source_tris: dex::Subset<2, dex::Primal>,
    pub source_edges: dex::Subset<1, dex::Primal>,
    pub source_verts: dex::Subset<0, dex::Primal>,
    pub measurement_tris: dex::Subset<2, dex::Primal>,
    pub measurement_edges: dex::Subset<1, dex::Primal>,
    pub measurement_verts: dex::Subset<0, dex::Primal>,
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

/// Measurements about the mesh.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Total area of triangles used for energy measurement.
    pub primal_measurement_area: f64,
    /// Total area of dual cells used for energy measurement.
    pub dual_measurement_area: f64,
    /// Area of the entire mesh.
    pub total_mesh_area: f64,
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

        //
        // mesh subsets
        //

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

        let bottom_verts = mesh.get_subset::<0>("990").expect("Subset not found");
        let top_edges = mesh.get_subset::<1>("991").expect("Subset not found");
        let top_verts = mesh.get_subset::<0>("991").expect("Subset not found");
        let right_edges = mesh.get_subset::<1>("992").expect("Subset not found");
        let right_verts = mesh.get_subset::<0>("992").expect("Subset not found");
        let left_edges = mesh.get_subset::<1>("993").expect("Subset not found");
        let left_verts = mesh.get_subset::<0>("993").expect("Subset not found");
        let side_edges = right_edges.union(&left_edges);

        // source terms are applied over a band of triangles along the bottom
        // and measurements performed along a similar band at the top
        let source_tris = dex::Subset::from_predicate(&mesh, |tri| {
            tri.vertex_indices()
                .any(|idx| bottom_verts.indices.contains(idx))
        });
        let source_edges = source_tris.boundary(&mesh);
        let source_verts = source_edges.boundary(&mesh);
        let measurement_tris = dex::Subset::from_predicate(&mesh, |tri| {
            tri.vertex_indices()
                .any(|idx| top_verts.indices.contains(idx))
        });
        let measurement_edges = measurement_tris.boundary(&mesh);
        let measurement_verts = measurement_edges.boundary(&mesh).difference(&top_verts);

        let subsets = Subsets {
            layers,
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
            measurement_verts,
        };

        //
        // spatially varying scaling factors
        //

        let stiffness_scaling = mesh.scaling_dual(|dvert| {
            let l = subsets
                .layers
                .iter()
                .find(|l| l.tris.contains(dvert.dual()))
                .unwrap();
            l.stiffness
        });
        let p_speed_scaling = mesh.scaling_dual(|dvert| {
            let l = subsets
                .layers
                .iter()
                .find(|l| l.tris.contains(dvert.dual()))
                .unwrap();
            l.p_wave_speed.powi(2)
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
        // TODO: this is the exact same code as mu_scaling
        // except with a different value being summed.
        // refactor the reduce repetition
        let s_speed_scaling = mesh.scaling(|vert| {
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

                    sum += layer.s_wave_speed.powi(2) * ds_area;
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
                -length * orientation as f64 / top_layer.p_wave_speed,
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
            p_step: dt * p_speed_scaling * mesh.star() * mesh.d(),
            q_step_p: (dt * periodic_proj_edge.clone() * periodic_star_1_dual * mesh.d())
                .exclude_subset(&subsets.top_edges),
            q_step_w: (-dt * periodic_proj_edge.clone() * mesh.d())
                .exclude_subset(&subsets.top_edges),
            p_absorber,
            w_step: dt
                * periodic_proj_vert.clone()
                * s_speed_scaling
                * periodic_star_0_inv.clone()
                * mesh.d()
                * mesh.star(),
            w_absorber,
            stiffness_scaling,
            mu_scaling,
            inv_density_scaling,
        };

        let stats = Stats {
            primal_measurement_area: mesh
                .simplices_in(&subsets.measurement_tris)
                .map(|tri| tri.volume())
                .sum(),
            dual_measurement_area: mesh
                .simplices_in(&subsets.measurement_verts)
                .map(|vert| vert.dual_volume())
                .sum(),
            total_mesh_area: mesh.simplices::<2>().map(|tri| tri.volume()).sum(),
        };

        Ok(Self {
            mesh,
            subsets,
            ops,
            dt,
            stats,
        })
    }
}
