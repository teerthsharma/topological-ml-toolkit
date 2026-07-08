// CUDA reference kernels for topology preprocessing.
//
// These kernels are not wired into the Python package yet. They are checked in
// as the CUDA backend source contract: distance tiles and threshold edge masks
// are the first hot paths that must match the Safe Rust / Python reference
// behavior before any CUDA speed claim is published.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void topoml_pairwise_l2_f32(
    const float* __restrict__ points,
    float* __restrict__ out,
    int64_t n_points,
    int64_t dim) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_points || col >= n_points) {
    return;
  }

  float acc = 0.0f;
  for (int64_t k = 0; k < dim; ++k) {
    const float delta = points[row * dim + k] - points[col * dim + k];
    acc += delta * delta;
  }
  out[row * n_points + col] = sqrtf(acc);
}

extern "C" __global__ void topoml_threshold_edges_u8(
    const float* __restrict__ distances,
    uint8_t* __restrict__ edges,
    int64_t n_points,
    float radius) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_points || col >= n_points) {
    return;
  }
  const float value = distances[row * n_points + col];
  edges[row * n_points + col] = (row != col && value <= radius) ? 1 : 0;
}
