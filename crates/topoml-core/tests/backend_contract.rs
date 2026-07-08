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
fn asm_avx512_metadata_declares_cpu_and_correctness_gates() {
    let metadata = backend_metadata(BackendId::AsmAvx512).expect("ASM metadata should exist");

    assert!(!metadata.active);
    assert!(metadata.planned);
    assert!(metadata.warnings.contains(&BackendWarning::CpuidGate));
    assert!(metadata.warnings.contains(&BackendWarning::CorrectnessGate));
    assert!(select_backend(BackendId::AsmAvx512).is_none());
}

#[test]
fn planned_native_and_framework_backends_are_not_selected() {
    for id in [
        BackendId::Cpp,
        BackendId::Triton,
        BackendId::PyTorch,
        BackendId::TensorFlow,
    ] {
        let metadata = backend_metadata(id).expect("planned backend metadata should exist");

        assert!(!metadata.active);
        assert!(!metadata.available);
        assert!(metadata.planned);
        assert!(!metadata.gates.is_empty());
        assert!(select_backend(id).is_none());
    }
}
