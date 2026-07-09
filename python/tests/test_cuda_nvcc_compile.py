from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.cuda_compile
@pytest.mark.parametrize(
    ("source_name", "expected_symbols"),
    [
        (
            "topology_distance.cu",
            (
                "topoml_pairwise_l2_f32",
                "topoml_threshold_edges_u8",
                "topoml_cuda_pairwise_l2_f32_host",
                "topoml_cuda_threshold_edges_u8_host",
            ),
        ),
        ("warp_reductions.cu", ("topoml_row_sum_f32", "topoml_persistence_image_accumulate_f32")),
    ],
)
def test_cuda_sources_compile_to_objects_with_nvcc_when_available(tmp_path, source_name, expected_symbols):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    if sys.platform == "win32" and shutil.which("cl.exe") is None:
        pytest.skip("nvcc is available, but the MSVC host compiler cl.exe is not on PATH")

    source = ROOT / "backends" / "cuda" / source_name
    assert source.exists()
    text = source.read_text(encoding="utf-8")
    for symbol in expected_symbols:
        assert symbol in text

    output = tmp_path / f"{source.stem}.o"
    command = [
        nvcc,
        "-std=c++17",
        "-c",
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "nvcc failed to compile "
            f"{source_name}\ncommand: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert output.exists()
    assert output.stat().st_size > 0
