use topoml_core::{
    persistent_homology, time_delay_embedding, ComplexKind, PersistenceConfig, PointCloud,
};

fn circle_points(n: usize) -> PointCloud<2> {
    let points = (0..n)
        .map(|i| {
            let t = (i as f64) * core::f64::consts::TAU / (n as f64);
            [t.cos(), t.sin()]
        })
        .collect();
    PointCloud::new(points).expect("circle point cloud should be valid")
}

#[test]
fn h0_tracks_cluster_merges() {
    let cloud = PointCloud::<2>::new(vec![[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]])
        .expect("cluster cloud should be valid");
    let diagram = persistent_homology(
        &cloud,
        PersistenceConfig::builder()
            .max_homology_dim(0)
            .max_radius(10.0)
            .build(),
    )
    .expect("diagram should compute");

    assert_eq!(diagram.betti_at(0.1).beta0, 3);
    assert_eq!(diagram.betti_at(0.3).beta0, 2);
    assert_eq!(diagram.betti_at(6.0).beta0, 1);
}

#[test]
fn circle_exposes_one_persistent_loop() {
    let cloud = circle_points(12);
    let diagram = persistent_homology(
        &cloud,
        PersistenceConfig::builder()
            .max_homology_dim(1)
            .max_radius(2.0)
            .complex_kind(ComplexKind::VietorisRips)
            .build(),
    )
    .expect("diagram should compute");

    let beta = diagram.betti_at(1.0);
    assert!(beta.beta1 >= 1, "circle should expose a persistent H1 loop");
}

#[test]
fn time_delay_embedding_turns_series_into_point_cloud() {
    let samples = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0];
    let cloud = time_delay_embedding::<3>(&samples, 1).expect("embedding should compute");

    assert_eq!(cloud.dimension(), 3);
    assert_eq!(cloud.len(), 4);
    assert_eq!(cloud.points()[0], [0.0, 1.0, 0.0]);
}
