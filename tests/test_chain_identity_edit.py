import numpy as np
import torch
from types import SimpleNamespace

from scripts.chain_tests.identity_config import build_tasks
from scripts.chain_tests.identity_edit import expanded_face_box, context_weight, face_token_mask, identity_state_record, make_identity_ctx_edit, mix_tokens


def test_expanded_face_box_uses_confident_points_and_quarter_padding():
    points = np.array([[.2, .3], [.4, .5], [.8, .7]])
    scores = np.array([.4, .2, .9])
    assert np.allclose(expanded_face_box(points, scores), (.05, .2, .95, .8))


def test_context_weight_is_fraction_of_registered_window():
    assert context_weight(list(range(84, 96)), (96, 144)) == 0.0
    assert context_weight(list(range(90, 102)), (96, 144)) == 0.5
    assert context_weight(list(range(96, 108)), (96, 144)) == 1.0


def test_zero_strength_returns_the_original_tensor_object_and_face_mix_is_local():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    mask = torch.tensor([False, True, True, False])
    assert mix_tokens(own, donor, mask, 0.0) is own
    actual = mix_tokens(own, donor, mask, 0.25)
    assert torch.equal(actual[:, ~mask], own[:, ~mask])
    assert torch.equal(actual[:, mask], torch.full((1, 2, 1), .5, dtype=torch.float16))


def test_face_token_mask_selects_centres_inside_registered_box():
    tokens = torch.zeros((1, 16, 2), dtype=torch.float16)
    mask = face_token_mask(tokens, (0.2, 0.2, 0.8, 0.8))
    assert mask.tolist() == [False, False, False, False, False, True, True, False,
                             False, True, True, False, False, False, False, False]


def test_registered_matrix_is_frozen_at_eighty_eight_tasks():
    tasks = build_tasks()
    assert len(tasks) == 88
    assert sum(task["kind"] == "clean_a" for task in tasks) == 8
    assert sum(task["kind"] == "clean_b" for task in tasks) == 8
    assert sum(task["kind"] == "zero_action" for task in tasks) == 8
    assert {tuple(task["actions"][0]["window"]) for task in tasks if task["actions"]} == {(96, 144), (168, 216)}


def test_context_editor_restores_pristine_bank_at_zero_strength_and_audits_formula():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    reader = SimpleNamespace(bank=[own])
    edit, state = make_identity_ctx_edit([reader], [donor], (.2, .2, .3, .3), {"alpha": .5, "window": [2, 4]})
    edit(0, [0, 1, 2, 3], None)
    assert torch.equal(reader.bank[0][:, 1:], own[:, 1:])
    assert state["checks"]["outside_face_identical"]
    edit(0, [0, 1], None)
    assert reader.bank[0] is own
    assert state["checks"]["zero_strength_original_object"]
    assert state["step0_contexts"][0]["w"] == .5


def test_state_record_uses_recorded_effective_strength():
    task = {"seed": 3407, "actions": [{"donor": "CRB100_002"}]}
    contexts = [{"frames": list(range(12)), "effective_strength": .2}]
    record = identity_state_record(task, contexts, segment_count=2)
    assert np.isclose(record["segments"][0]["planned_identity_mix_strength"], .2)
    assert record["segments"][1]["planned_identity_mix_strength"] == 0.0
