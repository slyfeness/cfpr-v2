from pathlib import Path


def test_audit_recognizes_complete_posthoc_retry(tmp_path):
    from scripts.evaluator_unit_tests.audit_conditions import condition_status
    required = ["run_meta.json", "face.json"]
    normal = tmp_path / "evaluators"
    retry = tmp_path / "evaluators_posthoc_retry_v1"
    retry.mkdir()
    for name in required:
        (retry / name).write_text("ok")
    result = condition_status(normal, required)
    assert result["status"] == "retried"
    assert result["retry_path"] == str(retry)