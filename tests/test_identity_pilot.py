import torch

from scripts.chain_tests.identity_pilot import make_pilot_ctx_edit, pilot_specs


def test_pilot_registry_contains_all_eight_frozen_variants():
    assert [spec["id"] for spec in pilot_specs()] == [f"P{i}" for i in range(8)]


def test_p1_replaces_only_the_selected_face_tokens():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    reader = type("Reader", (), {"bank": [own]})()
    edit, state = make_pilot_ctx_edit(
        [reader], [donor], [donor], (.2, .2, .8, .3), {"id": "P1", "window": [2, 4]}
    )
    edit(0, [2, 3], None)
    assert torch.equal(reader.bank[0][:, [2, 3]], own[:, [2, 3]])
    assert torch.equal(reader.bank[0][:, [0, 1]], donor[:, [0, 1]])
    assert state["contexts"][0]["operation"] == "replace_face"


def test_p5_appends_donor_face_as_an_additional_bank_item():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    reader = type("Reader", (), {"bank": [own]})()
    edit, _ = make_pilot_ctx_edit(
        [reader], [donor], [donor], (.2, .2, .8, .3), {"id": "P5", "window": [2, 4]}
    )
    edit(0, [2, 3], None)
    assert len(reader.bank) == 2
    assert torch.equal(reader.bank[0], own)
    assert torch.equal(reader.bank[1], donor[:, [0, 1]])


def test_p6_has_the_same_token_values_as_p1_but_a_different_bank_partition():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    r1, r6 = type("Reader", (), {"bank": [own]})(), type("Reader", (), {"bank": [own]})()
    e1, _ = make_pilot_ctx_edit([r1], [donor], [donor], (.2, .2, .8, .3), {"id": "P1", "window": [2, 4]})
    e6, _ = make_pilot_ctx_edit([r6], [donor], [donor], (.2, .2, .8, .3), {"id": "P6", "window": [2, 4]})
    e1(0, [2, 3], None)
    e6(0, [2, 3], None)
    assert len(r1.bank) == 1
    assert len(r6.bank) == 2
    assert sorted(r1.bank[0].flatten().tolist()) == sorted(torch.cat(r6.bank, dim=1).flatten().tolist())


def test_no_overlap_restores_the_pristine_bank_object():
    own = torch.zeros((1, 4, 1), dtype=torch.float16)
    donor = torch.full((1, 4, 1), 2.0, dtype=torch.float16)
    reader = type("Reader", (), {"bank": [own]})()
    edit, state = make_pilot_ctx_edit(
        [reader], [donor], [donor], (.2, .2, .8, .3), {"id": "P0", "window": [2, 4]}
    )
    edit(0, [0, 1], None)
    assert reader.bank[0] is own
    assert state["contexts"][0]["operation"] == "restore_original"
