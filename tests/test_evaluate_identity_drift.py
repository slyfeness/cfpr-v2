from scripts.chain_tests.evaluate_identity_drift import arcface_command


def test_arcface_command_uses_original_named_reference_and_generated_frames(tmp_path):
    task = {"sample": "CRB100_001", "out_id": "CRB100_001/W1/a_0.20"}
    cmd = arcface_command(tmp_path, task, "CRB100_002", "donor")
    assert cmd[1].endswith("eval_identity_arcface.py")
    assert cmd[2].endswith("CRB100_002/reference.png")
    assert cmd[-1].endswith("frames_uint8.npy")
    assert "/arcface_donor" in cmd[-2]
