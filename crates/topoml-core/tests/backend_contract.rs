use topoml_core::{backend_metadata, select_backend, BackendId, BackendWarning};

#[test]
fn safe_rust_backend_is_active_and_selectable() {
    let selected = select_backend(BackendId::SafeRust).expect("safe Rust backend should select");

    assert_eq!(selected.id, BackendId::SafeRust);
    assert!(selected.active);
    assert!(selected.available);
    assert!(!selected.planned);
}

#[test]
fn cpp_backend_metadata_is_active_after_native_h0_gate() {
    let selected =
        select_backend(BackendId::Cpp).expect("C++ backend should select after H0 native gate");

    assert_eq!(selected.id, BackendId::Cpp);
    assert!(selected.active);
    assert!(selected.available);
    assert!(!selected.planned);
}

#[test]
fn framework_adapters_are_active_optional_not_planned() {
    for id in [BackendId::PyTorch, BackendId::TensorFlow] {
        let metadata = backend_metadata(id).expect("framework metadata should exist");

        assert!(metadata.active);
        assert!(!metadata.available);
        assert!(!metadata.planned);
        assert!(metadata
            .warnings
            .contains(&BackendWarning::OptionalDependency));
        assert!(!metadata.warnings.contains(&BackendWarning::PlannedOnly));
        assert!(!metadata
            .warnings
            .contains(&BackendWarning::MissingImplementation));
        assert!(select_backend(id).is_none());
    }
}

#[test]
fn asm_avx512_metadata_declares_cpu_and_correctness_gates() {
    let metadata = backend_metadata(BackendId::AsmAvx512).expect("ASM metadata should exist");

    assert!(metadata.active);
    assert!(!metadata.available);
    assert!(!metadata.planned);
    assert!(metadata.warnings.contains(&BackendWarning::CpuidGate));
    assert!(metadata.warnings.contains(&BackendWarning::CorrectnessGate));
    assert!(select_backend(BackendId::AsmAvx512).is_none());
}

#[test]
fn planned_acceleration_backends_are_not_selected() {
    let id = BackendId::Triton;
    let metadata = backend_metadata(id).expect("planned backend metadata should exist");

    assert!(!metadata.active);
    assert!(!metadata.available);
    assert!(metadata.planned);
    assert!(!metadata.gates.is_empty());
    assert!(select_backend(id).is_none());
}
