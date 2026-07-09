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

extern "C" int topoml_cuda_pairwise_l2_f32_host(
    const float* points,
    float* out,
    int64_t n_points,
    int64_t dim) {
  if (points == nullptr || out == nullptr || n_points <= 0 || dim <= 0) {
    return (int)cudaErrorInvalidValue;
  }

  float* device_points = nullptr;
  float* device_out = nullptr;
  const size_t points_bytes = (size_t)n_points * (size_t)dim * sizeof(float);
  const size_t out_bytes = (size_t)n_points * (size_t)n_points * sizeof(float);

  cudaError_t err = cudaMalloc((void**)&device_points, points_bytes);
  if (err != cudaSuccess) {
    return (int)err;
  }
  err = cudaMalloc((void**)&device_out, out_bytes);
  if (err != cudaSuccess) {
    cudaFree(device_points);
    return (int)err;
  }
  err = cudaMemcpy(device_points, points, points_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    const dim3 block(16, 16);
    const dim3 grid((unsigned)((n_points + block.x - 1) / block.x),
                    (unsigned)((n_points + block.y - 1) / block.y));
    topoml_pairwise_l2_f32<<<grid, block>>>(device_points, device_out, n_points, dim);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, device_out, out_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(device_out);
  cudaFree(device_points);
  return (int)err;
}

extern "C" int topoml_cuda_threshold_edges_u8_host(
    const float* distances,
    uint8_t* edges,
    int64_t n_points,
    float radius) {
  if (distances == nullptr || edges == nullptr || n_points <= 0 || radius < 0.0f) {
    return (int)cudaErrorInvalidValue;
  }

  float* device_distances = nullptr;
  uint8_t* device_edges = nullptr;
  const size_t matrix_bytes = (size_t)n_points * (size_t)n_points * sizeof(float);
  const size_t edge_bytes = (size_t)n_points * (size_t)n_points * sizeof(uint8_t);

  cudaError_t err = cudaMalloc((void**)&device_distances, matrix_bytes);
  if (err != cudaSuccess) {
    return (int)err;
  }
  err = cudaMalloc((void**)&device_edges, edge_bytes);
  if (err != cudaSuccess) {
    cudaFree(device_distances);
    return (int)err;
  }
  err = cudaMemcpy(device_distances, distances, matrix_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    const dim3 block(16, 16);
    const dim3 grid((unsigned)((n_points + block.x - 1) / block.x),
                    (unsigned)((n_points + block.y - 1) / block.y));
    topoml_threshold_edges_u8<<<grid, block>>>(device_distances, device_edges, n_points, radius);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(edges, device_edges, edge_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(device_edges);
  cudaFree(device_distances);
  return (int)err;
}
