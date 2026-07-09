from __future__ import annotations

import numpy as np

import topoml


def test_noisy_circle_dataset_is_deterministic_and_has_h1_signal() -> None:
    left = topoml.make_noisy_circle(n_samples=32, noise=0.0, random_state=7)
    right = topoml.make_noisy_circle(n_samples=32, noise=0.0, random_state=7)

    assert left.name == "noisy_circle"
    assert left.points.shape == (32, 2)
    assert np.allclose(left.points, right.points)
    assert left.expected_betti == {"beta0": 1, "beta1": 1}

    diagram = topoml.persistent_homology(left.points, max_dim=1, max_radius=2.0)
    assert diagram.betti_at(0.45).beta1 == 1


def test_cluster_dataset_has_known_h0_merge_fixture() -> None:
    fixture = topoml.make_cluster_bridge(random_state=0)

    assert fixture.name == "cluster_bridge"
    assert fixture.points.shape == (3, 2)
    assert fixture.expected_betti == {"beta0@0.1": 3, "beta0@0.3": 2, "beta0@6": 1}
    assert fixture.labels.tolist() == ["near", "near", "far"]


def test_benchmark_dataset_registry_lists_and_loads_named_fixtures() -> None:
    names = topoml.list_benchmark_datasets()

    assert {"noisy_circle", "cluster_bridge", "two_circles"}.issubset(names)
    dataset = topoml.load_benchmark_dataset("two_circles", n_samples=16, noise=0.0, random_state=3)

    assert dataset.name == "two_circles"
    assert dataset.points.shape == (32, 2)
    assert dataset.labels.shape == (32,)
    assert dataset.expected_betti["components"] == 2


def test_benchmark_dataset_contracts_are_exported() -> None:
    exported = set(topoml.__all__)

    assert {
        "BenchmarkDataset",
        "list_benchmark_datasets",
        "load_benchmark_dataset",
        "make_cluster_bridge",
        "make_noisy_circle",
        "make_two_circles",
    }.issubset(exported)
