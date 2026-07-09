#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

const SIMPLEX_VERTICES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendId {
    SafeRust,
    Cpp,
    AsmAvx512,
    Cuda,
    Triton,
    PyTorch,
    TensorFlow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCapability {
    PersistentHomology,
    TimeDelayEmbedding,
    NativeExtension,
    SimdAcceleration,
    FrameworkAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendWarning {
    PlannedOnly,
    MissingImplementation,
    CpuidGate,
    CorrectnessGate,
    OptionalDependency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMetadata {
    pub id: BackendId,
    pub name: &'static str,
    pub active: bool,
    pub available: bool,
    pub planned: bool,
    pub capabilities: &'static [BackendCapability],
    pub gates: &'static [&'static str],
    pub warnings: &'static [BackendWarning],
}

const SAFE_RUST_CAPABILITIES: &[BackendCapability] = &[
    BackendCapability::PersistentHomology,
    BackendCapability::TimeDelayEmbedding,
];
const CPP_CAPABILITIES: &[BackendCapability] = &[
    BackendCapability::PersistentHomology,
    BackendCapability::NativeExtension,
];
const ASM_AVX512_CAPABILITIES: &[BackendCapability] = &[BackendCapability::SimdAcceleration];
const CUDA_CAPABILITIES: &[BackendCapability] = &[
    BackendCapability::NativeExtension,
    BackendCapability::SimdAcceleration,
];
const FRAMEWORK_CAPABILITIES: &[BackendCapability] = &[BackendCapability::FrameworkAdapter];

const NO_GATES: &[&str] = &[];
const CPP_GATES: &[&str] = &["portable C ABI", "barcode equivalence"];
const ASM_AVX512_GATES: &[&str] = &["CPUID AVX-512 support", "barcode equivalence"];
const CUDA_GATES: &[&str] = &[
    "nvcc compiler",
    "CUDA runtime",
    "CUDA device",
    "NumPy distance equivalence",
];
const TRITON_GATES: &[&str] = &[
    "optional torch dependency",
    "optional triton dependency",
    "CUDA device",
    "torch.cdist parity",
];
const PYTORCH_GATES: &[&str] = &[
    "optional torch dependency",
    "dense fallback",
    "torch.compile-safe behavior",
];
const TENSORFLOW_GATES: &[&str] = &[
    "optional tensorflow dependency",
    "eager parity",
    "graph-mode parity",
];

const NO_WARNINGS: &[BackendWarning] = &[];
const ASM_AVX512_WARNINGS: &[BackendWarning] =
    &[BackendWarning::CpuidGate, BackendWarning::CorrectnessGate];
const OPTIONAL_FRAMEWORK_WARNINGS: &[BackendWarning] = &[BackendWarning::OptionalDependency];

pub const BACKEND_METADATA: &[BackendMetadata] = &[
    BackendMetadata {
        id: BackendId::SafeRust,
        name: "safe_rust",
        active: true,
        available: true,
        planned: false,
        capabilities: SAFE_RUST_CAPABILITIES,
        gates: NO_GATES,
        warnings: NO_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::Cpp,
        name: "cpp",
        active: true,
        available: true,
        planned: false,
        capabilities: CPP_CAPABILITIES,
        gates: CPP_GATES,
        warnings: NO_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::AsmAvx512,
        name: "asm_avx512",
        active: true,
        available: false,
        planned: false,
        capabilities: ASM_AVX512_CAPABILITIES,
        gates: ASM_AVX512_GATES,
        warnings: ASM_AVX512_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::Cuda,
        name: "cuda",
        active: true,
        available: false,
        planned: false,
        capabilities: CUDA_CAPABILITIES,
        gates: CUDA_GATES,
        warnings: OPTIONAL_FRAMEWORK_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::Triton,
        name: "triton",
        active: true,
        available: false,
        planned: false,
        capabilities: FRAMEWORK_CAPABILITIES,
        gates: TRITON_GATES,
        warnings: OPTIONAL_FRAMEWORK_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::PyTorch,
        name: "pytorch",
        active: true,
        available: false,
        planned: false,
        capabilities: FRAMEWORK_CAPABILITIES,
        gates: PYTORCH_GATES,
        warnings: OPTIONAL_FRAMEWORK_WARNINGS,
    },
    BackendMetadata {
        id: BackendId::TensorFlow,
        name: "tensorflow",
        active: true,
        available: false,
        planned: false,
        capabilities: FRAMEWORK_CAPABILITIES,
        gates: TENSORFLOW_GATES,
        warnings: OPTIONAL_FRAMEWORK_WARNINGS,
    },
];

pub fn backend_metadata(id: BackendId) -> Option<&'static BackendMetadata> {
    BACKEND_METADATA.iter().find(|backend| backend.id == id)
}

pub fn select_backend(id: BackendId) -> Option<&'static BackendMetadata> {
    backend_metadata(id).filter(|backend| backend.active && backend.available)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud<const D: usize> {
    points: Vec<[f64; D]>,
}

impl<const D: usize> PointCloud<D> {
    pub fn new(points: Vec<[f64; D]>) -> Result<Self, TopomlError> {
        if D == 0 {
            return Err(TopomlError::InvalidDimension);
        }
        if points.is_empty() {
            return Err(TopomlError::EmptyInput);
        }
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(TopomlError::NonFiniteCoordinate);
        }
        Ok(Self { points })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub const fn dimension(&self) -> usize {
        D
    }

    pub fn points(&self) -> &[[f64; D]] {
        &self.points
    }

    fn distance(&self, left: usize, right: usize) -> f64 {
        let squared = self.points[left]
            .iter()
            .zip(self.points[right].iter())
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f64>();
        libm::sqrt(squared)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexKind {
    VietorisRips,
    Witness { max_landmarks: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistenceConfig {
    pub max_homology_dim: usize,
    pub max_points: usize,
    pub max_simplices: usize,
    pub max_radius: f64,
    pub complex_kind: ComplexKind,
}

impl PersistenceConfig {
    pub const fn builder() -> PersistenceConfigBuilder {
        PersistenceConfigBuilder {
            config: Self {
                max_homology_dim: 1,
                max_points: 64,
                max_simplices: 65_536,
                max_radius: f64::INFINITY,
                complex_kind: ComplexKind::VietorisRips,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PersistenceConfigBuilder {
    config: PersistenceConfig,
}

impl PersistenceConfigBuilder {
    pub const fn max_homology_dim(mut self, value: usize) -> Self {
        self.config.max_homology_dim = value;
        self
    }

    pub const fn max_points(mut self, value: usize) -> Self {
        self.config.max_points = value;
        self
    }

    pub const fn max_simplices(mut self, value: usize) -> Self {
        self.config.max_simplices = value;
        self
    }

    pub const fn max_radius(mut self, value: f64) -> Self {
        self.config.max_radius = value;
        self
    }

    pub const fn complex_kind(mut self, value: ComplexKind) -> Self {
        self.config.complex_kind = value;
        self
    }

    pub const fn build(self) -> PersistenceConfig {
        self.config
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopomlError {
    EmptyInput,
    InvalidDimension,
    InvalidRadius,
    NonFiniteCoordinate,
    TooManyPoints { actual: usize, max: usize },
    TooManySimplices { max: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BettiNumbers {
    pub beta0: u32,
    pub beta1: u32,
    pub beta2: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceDiagram {
    pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        Self { pairs }
    }

    pub fn pairs(&self) -> &[PersistencePair] {
        &self.pairs
    }

    pub fn betti_at(&self, radius: f64) -> BettiNumbers {
        self.pairs
            .iter()
            .fold(BettiNumbers::default(), |mut acc, pair| {
                let alive =
                    pair.birth <= radius && pair.death.map(|death| radius < death).unwrap_or(true);
                if alive {
                    match pair.dimension {
                        0 => acc.beta0 += 1,
                        1 => acc.beta1 += 1,
                        2 => acc.beta2 += 1,
                        _ => {}
                    }
                }
                acc
            })
    }
}

#[derive(Debug, Clone)]
struct Simplex {
    vertices: [usize; SIMPLEX_VERTICES],
    len: usize,
    dimension: usize,
    filtration: f64,
}

pub fn time_delay_embedding<const D: usize>(
    samples: &[f64],
    tau: usize,
) -> Result<PointCloud<D>, TopomlError> {
    if D == 0 || tau == 0 {
        return Err(TopomlError::InvalidDimension);
    }
    let needed = (D - 1) * tau + 1;
    if samples.len() < needed {
        return Err(TopomlError::EmptyInput);
    }

    let mut points = Vec::with_capacity(samples.len() + 1 - needed);
    for start in 0..=(samples.len() - needed) {
        let mut point = [0.0; D];
        for dim in 0..D {
            point[dim] = samples[start + dim * tau];
        }
        points.push(point);
    }
    PointCloud::new(points)
}

pub fn persistent_homology<const D: usize>(
    cloud: &PointCloud<D>,
    config: PersistenceConfig,
) -> Result<PersistenceDiagram, TopomlError> {
    validate_config(cloud.len(), config)?;
    let source = match config.complex_kind {
        ComplexKind::VietorisRips => cloud.clone(),
        ComplexKind::Witness { max_landmarks } => select_landmarks(cloud, max_landmarks)?,
    };
    validate_config(source.len(), config)?;
    reduce_z2(
        &build_vietoris_rips_simplices(&source, config)?,
        config.max_homology_dim,
    )
}

fn validate_config(point_count: usize, config: PersistenceConfig) -> Result<(), TopomlError> {
    if config.max_homology_dim > 2 {
        return Err(TopomlError::InvalidDimension);
    }
    if config.max_radius.is_nan() || config.max_radius < 0.0 {
        return Err(TopomlError::InvalidRadius);
    }
    if point_count == 0 {
        return Err(TopomlError::EmptyInput);
    }
    if point_count > config.max_points {
        return Err(TopomlError::TooManyPoints {
            actual: point_count,
            max: config.max_points,
        });
    }
    Ok(())
}

fn select_landmarks<const D: usize>(
    cloud: &PointCloud<D>,
    max_landmarks: usize,
) -> Result<PointCloud<D>, TopomlError> {
    if max_landmarks == 0 || cloud.len() <= max_landmarks {
        return Ok(cloud.clone());
    }

    let mut selected = vec![false; cloud.len()];
    let mut landmarks = Vec::with_capacity(max_landmarks);
    landmarks.push(cloud.points[0]);
    selected[0] = true;

    while landmarks.len() < max_landmarks {
        let mut best_idx = None;
        let mut best_distance = -1.0;
        for (idx, is_selected) in selected.iter().enumerate() {
            if *is_selected {
                continue;
            }
            let nearest = landmarks
                .iter()
                .map(|landmark| euclidean(&cloud.points[idx], landmark))
                .fold(f64::INFINITY, f64::min);
            if nearest > best_distance {
                best_distance = nearest;
                best_idx = Some(idx);
            }
        }
        let Some(idx) = best_idx else { break };
        selected[idx] = true;
        landmarks.push(cloud.points[idx]);
    }

    PointCloud::new(landmarks)
}

fn build_vietoris_rips_simplices<const D: usize>(
    cloud: &PointCloud<D>,
    config: PersistenceConfig,
) -> Result<Vec<Simplex>, TopomlError> {
    let n = cloud.len();
    let mut distances = vec![0.0; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let distance = cloud.distance(i, j);
            distances[i * n + j] = distance;
            distances[j * n + i] = distance;
        }
    }

    let mut simplices = Vec::new();
    for i in 0..n {
        push_simplex(
            &mut simplices,
            simplex([i, 0, 0, 0], 1, 0.0),
            config.max_simplices,
        )?;
    }
    for i in 0..n {
        for j in i + 1..n {
            let r = distances[i * n + j];
            if r <= config.max_radius {
                push_simplex(
                    &mut simplices,
                    simplex([i, j, 0, 0], 2, r),
                    config.max_simplices,
                )?;
            }
        }
    }
    if config.max_homology_dim >= 1 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let r = distances[i * n + j]
                        .max(distances[i * n + k])
                        .max(distances[j * n + k]);
                    if r <= config.max_radius {
                        push_simplex(
                            &mut simplices,
                            simplex([i, j, k, 0], 3, r),
                            config.max_simplices,
                        )?;
                    }
                }
            }
        }
    }
    if config.max_homology_dim >= 2 {
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    for l in k + 1..n {
                        let r = distances[i * n + j]
                            .max(distances[i * n + k])
                            .max(distances[i * n + l])
                            .max(distances[j * n + k])
                            .max(distances[j * n + l])
                            .max(distances[k * n + l]);
                        if r <= config.max_radius {
                            push_simplex(
                                &mut simplices,
                                simplex([i, j, k, l], 4, r),
                                config.max_simplices,
                            )?;
                        }
                    }
                }
            }
        }
    }

    simplices.sort_by(compare_simplex);
    Ok(simplices)
}

fn reduce_z2(
    simplices: &[Simplex],
    max_homology_dim: usize,
) -> Result<PersistenceDiagram, TopomlError> {
    let mut reduced_columns: Vec<Vec<usize>> = Vec::with_capacity(simplices.len());
    let mut low_owner: Vec<Option<usize>> = vec![None; simplices.len()];
    let mut paired_birth = vec![false; simplices.len()];
    let mut pairs = Vec::new();

    for j in 0..simplices.len() {
        let mut column = boundary_indices(simplices, j);
        while let Some(&low) = column.last() {
            let Some(owner) = low_owner[low] else { break };
            column = xor_sorted(&column, &reduced_columns[owner]);
        }

        if let Some(&low) = column.last() {
            low_owner[low] = Some(j);
            paired_birth[low] = true;
            let dimension = simplices[low].dimension;
            if dimension <= max_homology_dim {
                pairs.push(PersistencePair {
                    dimension,
                    birth: simplices[low].filtration,
                    death: Some(simplices[j].filtration),
                });
            }
        }
        reduced_columns.push(column);
    }

    for (idx, column) in reduced_columns.iter().enumerate() {
        if column.is_empty() && !paired_birth[idx] && simplices[idx].dimension <= max_homology_dim {
            pairs.push(PersistencePair {
                dimension: simplices[idx].dimension,
                birth: simplices[idx].filtration,
                death: None,
            });
        }
    }

    pairs.sort_by(|a, b| {
        a.dimension
            .cmp(&b.dimension)
            .then(a.birth.total_cmp(&b.birth))
            .then(match (a.death, b.death) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => core::cmp::Ordering::Less,
                (None, Some(_)) => core::cmp::Ordering::Greater,
                (None, None) => core::cmp::Ordering::Equal,
            })
    });
    Ok(PersistenceDiagram::new(pairs))
}

fn simplex(vertices: [usize; SIMPLEX_VERTICES], len: usize, filtration: f64) -> Simplex {
    Simplex {
        vertices,
        len,
        dimension: len - 1,
        filtration,
    }
}

fn push_simplex(
    simplices: &mut Vec<Simplex>,
    simplex: Simplex,
    max_simplices: usize,
) -> Result<(), TopomlError> {
    if simplices.len() >= max_simplices {
        return Err(TopomlError::TooManySimplices { max: max_simplices });
    }
    simplices.push(simplex);
    Ok(())
}

fn compare_simplex(left: &Simplex, right: &Simplex) -> core::cmp::Ordering {
    left.filtration
        .total_cmp(&right.filtration)
        .then(left.dimension.cmp(&right.dimension))
        .then(left.vertices[..left.len].cmp(&right.vertices[..right.len]))
}

fn boundary_indices(simplices: &[Simplex], simplex_idx: usize) -> Vec<usize> {
    let simplex = &simplices[simplex_idx];
    if simplex.dimension == 0 {
        return Vec::new();
    }

    let mut boundary = Vec::with_capacity(simplex.len);
    for remove_idx in 0..simplex.len {
        let mut face = [0usize; SIMPLEX_VERTICES];
        let mut face_len = 0;
        for i in 0..simplex.len {
            if i != remove_idx {
                face[face_len] = simplex.vertices[i];
                face_len += 1;
            }
        }
        if let Some(idx) = find_simplex(simplices, simplex_idx, &face, face_len) {
            boundary.push(idx);
        }
    }
    boundary.sort_unstable();
    boundary
}

fn find_simplex(
    simplices: &[Simplex],
    before: usize,
    vertices: &[usize; SIMPLEX_VERTICES],
    len: usize,
) -> Option<usize> {
    simplices[..before]
        .iter()
        .position(|simplex| simplex.len == len && simplex.vertices[..len] == vertices[..len])
}

fn xor_sorted(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;
    while i < left.len() || j < right.len() {
        if j == right.len() || (i < left.len() && left[i] < right[j]) {
            out.push(left[i]);
            i += 1;
        } else if i == left.len() || right[j] < left[i] {
            out.push(right[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

fn euclidean<const D: usize>(left: &[f64; D], right: &[f64; D]) -> f64 {
    let squared = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>();
    libm::sqrt(squared)
}
