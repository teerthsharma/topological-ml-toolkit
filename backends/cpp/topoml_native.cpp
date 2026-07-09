// Portable C++ native backend contract for Topological ML Toolkit.
//
// This source is intentionally dependency-free. It gives the planned C++ backend
// a concrete ABI target for distance matrices before packaging it as a Python
// extension or Rust FFI target.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <vector>

namespace {

struct Edge {
  int64_t left;
  int64_t right;
  double distance;
};

class DisjointSet {
 public:
  explicit DisjointSet(int64_t n) : parent_(static_cast<size_t>(n)), rank_(static_cast<size_t>(n), 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int64_t find(int64_t value) {
    int64_t root = value;
    while (parent_[static_cast<size_t>(root)] != root) {
      root = parent_[static_cast<size_t>(root)];
    }
    while (parent_[static_cast<size_t>(value)] != value) {
      const int64_t next = parent_[static_cast<size_t>(value)];
      parent_[static_cast<size_t>(value)] = root;
      value = next;
    }
    return root;
  }

  bool unite(int64_t left, int64_t right) {
    int64_t a = find(left);
    int64_t b = find(right);
    if (a == b) {
      return false;
    }
    if (rank_[static_cast<size_t>(a)] < rank_[static_cast<size_t>(b)]) {
      std::swap(a, b);
    }
    parent_[static_cast<size_t>(b)] = a;
    if (rank_[static_cast<size_t>(a)] == rank_[static_cast<size_t>(b)]) {
      rank_[static_cast<size_t>(a)] += 1;
    }
    return true;
  }

 private:
  std::vector<int64_t> parent_;
  std::vector<int> rank_;
};

double l2_distance(const double* points, int64_t dim, int64_t row, int64_t col) {
  double acc = 0.0;
  for (int64_t k = 0; k < dim; ++k) {
    const double delta = points[row * dim + k] - points[col * dim + k];
    acc += delta * delta;
  }
  return std::sqrt(acc);
}

}  // namespace

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

extern "C" int topoml_h0_barcode_f64(
    const double* points,
    double* births,
    double* deaths,
    int64_t* out_count,
    int64_t n_points,
    int64_t dim,
    double max_radius) {
  if (points == nullptr || births == nullptr || deaths == nullptr || out_count == nullptr ||
      n_points <= 0 || dim <= 0 || max_radius < 0.0 || std::isnan(max_radius)) {
    return 1;
  }

  std::vector<Edge> edges;
  edges.reserve(static_cast<size_t>((n_points * (n_points - 1)) / 2));
  for (int64_t row = 0; row < n_points; ++row) {
    for (int64_t col = row + 1; col < n_points; ++col) {
      const double distance = l2_distance(points, dim, row, col);
      if (distance <= max_radius) {
        edges.push_back(Edge{row, col, distance});
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    if (a.left != b.left) {
      return a.left < b.left;
    }
    return a.right < b.right;
  });

  DisjointSet sets(n_points);
  int64_t count = 0;
  for (const Edge& edge : edges) {
    if (sets.unite(edge.left, edge.right)) {
      births[count] = 0.0;
      deaths[count] = edge.distance;
      ++count;
    }
  }

  std::vector<int64_t> roots;
  roots.reserve(static_cast<size_t>(n_points));
  for (int64_t point = 0; point < n_points; ++point) {
    const int64_t root = sets.find(point);
    if (std::find(roots.begin(), roots.end(), root) == roots.end()) {
      roots.push_back(root);
    }
  }
  for (int64_t ignored : roots) {
    (void)ignored;
    births[count] = 0.0;
    deaths[count] = -1.0;
    ++count;
  }

  *out_count = count;
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
