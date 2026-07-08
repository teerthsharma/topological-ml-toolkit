// CUDA warp/block reductions for topology feature preprocessing.
//
// These kernels provide real CUDA source for reductions used by future
// persistence-image and distance-summary GPU paths. They are deliberately
// standalone so they can be compiled by nvcc or folded into a future extension
// without pulling CUDA into the Python package import path.

#include <cuda_runtime.h>
#include <stdint.h>

__inline__ __device__ float topoml_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
  value += __shfl_down_sync(mask, value, 16);
  value += __shfl_down_sync(mask, value, 8);
  value += __shfl_down_sync(mask, value, 4);
  value += __shfl_down_sync(mask, value, 2);
  value += __shfl_down_sync(mask, value, 1);
  return value;
}

extern "C" __global__ void topoml_row_sum_f32(
    const float* __restrict__ matrix,
    float* __restrict__ row_sums,
    int64_t rows,
    int64_t cols) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ float warp_sums[32];

  if (row >= rows) {
    return;
  }

  float partial = 0.0f;
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    partial += matrix[row * cols + col];
  }
  partial = topoml_warp_sum(partial);
  if (lane == 0) {
    warp_sums[warp] = partial;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp == 0) {
    const int warp_count = (blockDim.x + 31) >> 5;
    block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
    block_sum = topoml_warp_sum(block_sum);
    if (lane == 0) {
      row_sums[row] = block_sum;
    }
  }
}

extern "C" __global__ void topoml_persistence_image_accumulate_f32(
    const float* __restrict__ births,
    const float* __restrict__ persistences,
    float* __restrict__ image,
    int64_t n_pairs,
    int64_t width,
    int64_t height,
    float birth_min,
    float birth_max,
    float persistence_min,
    float persistence_max,
    float sigma) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = width * height;
  if (pixel >= total) {
    return;
  }

  const int x_idx = pixel % width;
  const int y_idx = pixel / width;
  const float bx = birth_min + (birth_max - birth_min) * x_idx / max(1, (int)width - 1);
  const float py = persistence_min + (persistence_max - persistence_min) * y_idx / max(1, (int)height - 1);
  const float denom = 2.0f * sigma * sigma;

  float value = 0.0f;
  for (int64_t pair = 0; pair < n_pairs; ++pair) {
    const float db = bx - births[pair];
    const float dp = py - persistences[pair];
    const float weight = persistences[pair];
    value += weight * expf(-(db * db + dp * dp) / denom);
  }
  image[pixel] = value;
}
