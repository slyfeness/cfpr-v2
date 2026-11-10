import pytest
import torch
import ast
from pathlib import Path

from scripts.chain_tests.audio_edit import apply_audio_shift
from scripts.chain_tests.lip_offset_config import build_registered_tasks
from scripts.chain_tests.generate_lip_offset import task_state_record


def test_audio_shift_changes_only_the_registered_window():
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    edited = apply_audio_shift(audio, window=(4, 8), k=2)

    assert torch.equal(edited[:, :4], audio[:, :4])
    assert torch.equal(edited[:, 8:], audio[:, 8:])
    assert torch.equal(edited[:, 4:8], audio[:, 2:6])


def test_zero_shift_returns_the_original_tensor_object():
    audio = torch.randn(1, 12, 3)
    assert apply_audio_shift(audio, window=(4, 8), k=0) is audio


def test_audio_shift_rejects_a_source_index_outside_feature_tensor():
    audio = torch.zeros(1, 12, 1)
    with pytest.raises(ValueError, match="outside"):
        apply_audio_shift(audio, window=(0, 4), k=1)


def test_acc_generate_exposes_an_optional_audio_edit_hook():
    source = Path("src/acc_core.py").read_text()
    module = ast.parse(source)
    generate = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    defaults = dict(zip([argument.arg for argument in generate.args.args[-len(generate.args.defaults):]], generate.args.defaults))
    assert "audio_edit" in defaults
    assert isinstance(defaults["audio_edit"], ast.Constant) and defaults["audio_edit"].value is None
    assert "audio_edit(audio)" in source


def test_registered_matrix_is_eight_people_times_fifteen_tasks():
    tasks = build_registered_tasks(["A", "B"], {"W1": [96, 144], "W2": [168, 216]}, seed=3407)
    assert len(tasks) == 30
    assert sum(task["actions"] == [] for task in tasks) == 4
    assert sum(bool(task["actions"]) and task["actions"][0]["k"] == 0 for task in tasks) == 2
    assert {task["actions"][0]["k"] for task in tasks if task["actions"]} == {-8, -4, -2, 0, 2, 4, 8}


def test_registered_matrix_accepts_the_explicit_regression_frame_count():
    tasks = build_registered_tasks(["A"], {"W1": [48, 96], "W2": [72, 120]}, seed=3407, frames=120)
    assert {task["frames"] for task in tasks} == {120}


def test_state_record_marks_only_the_registered_audio_shift_window():
    task = {
        "sample": "CRB100_001", "frames": 240, "seed": 3407,
        "actions": [{"type": "audio_shift", "window": [96, 144], "k": 4}],
    }
    state = task_state_record(task)

    assert [row["planned_audio_shift_24fps"] for row in state["segments"]] == [
        0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    assert {tuple(row["window"]) for row in state["segments"] if row["window"]} == {(96, 144)}
