// Portable C++ native backend contract for Topological ML Toolkit.
//
// This source is intentionally dependency-free. It gives the planned C++ backend
// a concrete ABI target for distance matrices before packaging it as a Python
// extension or Rust FFI target.

#include <cmath>
#include <cstdint>

extern "C" int topoml_pairwise_l2_f64(
    const double* points,
    double* out,
    int64_t n_points,
    int64_t dim) {
  if (points == nullptr || out == nullptr || n_points <= 0 || dim <= 0) {
    return 1;
  }

  for (int64_t row = 0; row < n_points; ++row) {
    for (int64_t col = 0; col < n_points; ++col) {
      double acc = 0.0;
      for (int64_t k = 0; k < dim; ++k) {
        const double delta = points[row * dim + k] - points[col * dim + k];
        acc += delta * delta;
      }
      out[row * n_points + col] = std::sqrt(acc);
    }
  }
  return 0;
}

extern "C" int topoml_threshold_edges_u8(
    const double* distances,
    uint8_t* edges,
    int64_t n_points,
    double radius) {
  if (distances == nullptr || edges == nullptr || n_points <= 0 || radius < 0.0) {
    return 1;
  }

  for (int64_t row = 0; row < n_points; ++row) {
    for (int64_t col = 0; col < n_points; ++col) {
      const double value = distances[row * n_points + col];
      edges[row * n_points + col] = (row != col && value <= radius) ? 1 : 0;
    }
  }
  return 0;
}
