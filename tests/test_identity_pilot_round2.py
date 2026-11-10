import torch
from types import SimpleNamespace

from scripts.chain_tests.identity_pilot_round2 import make_round2_ctx_edit, round2_specs


def test_round2_registry_is_exactly_the_four_registered_variants():
    assert [x["id"] for x in round2_specs()] == ["P1b", "D75", "D90", "H50"]


def test_p1b_uses_discrete_transition_not_linear_mixing():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    reader = SimpleNamespace(bank=[own])
    edit, state = make_round2_ctx_edit([reader], [donor], (.2, .2, .8, .3), {"id": "P1b", "window": [2, 4], "p": 1.0})
    edit(0, [0, 2], None)  # w=.5: replace
    assert torch.equal(reader.bank[0][:, :2], donor[:, :2])
    edit(0, [0, 1, 2, 4], None)  # w=.25: restore pristine object
    assert reader.bank[0] is own
    assert state["contexts"][-1]["operation"] == "restore_original_below_threshold"


def test_d75_selects_a_fixed_per_layer_subset():
    own = torch.zeros((1, 16, 1), dtype=torch.float16)
    donor = torch.ones((1, 16, 1), dtype=torch.float16)
    reader = SimpleNamespace(bank=[own])
    edit, state = make_round2_ctx_edit([reader], [donor], (0, 0, 1, 1), {"id": "D75", "window": [2, 4], "p": .75})
    edit(0, [2, 3], None)
    assert int((reader.bank[0] == 1).sum()) == 12
    assert state["selected_face_token_counts"] == [12]
