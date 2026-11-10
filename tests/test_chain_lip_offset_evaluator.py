from pathlib import Path

from scripts.chain_tests.evaluate_lip_offset import lipsync_command


def test_lipsync_command_uses_original_audio_and_registered_gen_name():
    root = Path("/outputs/chain")
    task = {"sample": "CRB100_025", "out_id": "CRB100_025/W2/k_+4"}

    command = lipsync_command(root, task, Path("/project/eval_lipsync.py"), Path("/env/syncnet/python"))

    assert command == [
        "/env/syncnet/python",
        "/project/eval_lipsync.py",
        "gen",
        "/outputs/chain/CRB100_025/W2/k_+4/frames_uint8.npy",
        "/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1/CRB100_025/audio.wav",
        "/outputs/chain/CRB100_025/W2/k_+4/evaluators/lipsync",
    ]
