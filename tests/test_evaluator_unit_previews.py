from scripts.evaluator_unit_tests.make_previews import preview_frame_ids


def test_preview_frame_ids_adds_registered_context_and_stays_in_video_bounds():
    assert preview_frame_ids(240, 96, 144, context=24) == list(range(72, 168))
    assert preview_frame_ids(100, 5, 10, context=24) == list(range(0, 34))


def test_resolve_ffmpeg_prefers_explicit_existing_binary(tmp_path):
    from scripts.evaluator_unit_tests.make_previews import resolve_ffmpeg
    binary=tmp_path / "ffmpeg";binary.write_text("");binary.chmod(0o755)
    assert resolve_ffmpeg(str(binary)) == str(binary)
