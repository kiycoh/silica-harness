from silica.sources.convert import _mineru_error


def test_extracts_error_field_from_json_blob():
    # The real failure: mineru wrote a JSON task blob; the old [-300:] slice
    # started mid-token ("2:25:16…") and buried the message. The recorded blob
    # carried a missing `six`, which now routes to its own message
    # (test_missing_six_is_named_with_its_fix), so the parse itself is proved
    # here with a cause that has nothing to add.
    blob = (
        '{"started_at": "2026-07-21T22:25:16.251918+00:00", '
        '"error": "CUDA out of memory", "queued_ahead": 0}'
    )
    assert _mineru_error(blob) == "CUDA out of memory"


def test_raw_text_head_truncated_not_tail():
    assert _mineru_error("boom: " + "x" * 500) == ("boom: " + "x" * 294)


def test_plain_short_message_stripped():
    assert _mineru_error("  segfault  ") == "segfault"


def test_json_without_error_field_falls_back_to_head():
    assert _mineru_error('{"status": "queued"}') == '{"status": "queued"}'


def test_skips_server_startup_noise_and_surfaces_real_error():
    # The exact symptom: mineru's internal mineru-api logs fill the head, so the
    # old [:300] slice returned "Started local mineru-api ..." and hid the cause.
    stderr = (
        "2026-07-22 00:36:13.800 | INFO     | mineru.cli.client:run_orchestrated_cli:953"
        " - Started local mineru-api at http://127.0.0.1:49077\n"
        "INFO:     Started server process [1041512]\n"
        "INFO:     Waiting for application startup.\n"
        "Layout Predict:  50%|#####     | 20/40 [00:01<00:01, 18.5it/s]\n"
        "2026-07-22 00:36:20.1 | ERROR    | mineru.backend.pipeline:run:99 - CUDA out of memory\n"
    )
    out = _mineru_error(stderr)
    assert "out of memory" in out.lower()
    assert "mineru-api" not in out


def test_extracts_error_field_from_embedded_task_blob():
    # mineru 3.4.4's real failure shape: startup noise, then a final line with
    # the task JSON embedded — its "error" field sits past any 300-char window.
    stderr = (
        "2026-07-22 01:04:05.607 | INFO     | mineru.cli.client:run_orchestrated_cli:953"
        " - Started local mineru-api at http://127.0.0.1:52983\n"
        "INFO:     Started server process [1097256]\n"
        "Error: 1 task(s) failed while processing documents:\n"
        '- task#1 (l.Spark-SQL): Task f07cd05c failed for task#1 [l.Spark-SQL]: '
        '{"task_id": "f07cd05c", "status": "failed", "backend": "pipeline", '
        '"file_names": ["l.Spark-SQL"], "created_at": "2026-07-21T23:07:03+00:00", '
        '"started_at": "2026-07-21T23:07:03+00:00", "completed_at": '
        '"2026-07-21T23:07:06+00:00", "error": "CUDA out of memory", '
        '"queued_ahead": 0}\n'
    )
    assert _mineru_error(stderr) == "CUDA out of memory"


def test_missing_six_is_named_with_its_fix():
    """The one cause relayed as an instruction instead of a quote.

    mineru 3.4.4 imports `six` from its vendored pytorchocr without declaring
    it, so the message names a dependency of a dependency and the reader has no
    way to know it is not their fault. The `[pdf]` extra used to patch it in at
    install time and the user never saw it; since 2026-09-02 mineru is their own
    install, so the error is the only place the fix can live. Both real shapes
    it arrives in are pinned: the bare traceback and the task blob.
    """
    traceback = (
        "2026-09-02 | INFO     | mineru.cli:main:41 - Started local mineru-api\n"
        '  File "/x/pytorchocr/data/imaug/operators.py", line 7, in <module>\n'
        "ModuleNotFoundError: No module named 'six'\n"
    )
    blob = '{"status": "failed", "error": "No module named \'six\'"}'
    for stderr in (traceback, blob):
        out = _mineru_error(stderr)
        assert "six" in out
        assert "pip install six" in out


def test_last_meaningful_line_when_no_explicit_error_keyword():
    # No ERROR line (e.g. killed mid-run) → last non-noise line beats head noise.
    stderr = (
        "2026-07-22 | INFO | mineru.cli.client - Started local mineru-api\n"
        "OCR-rec Predict:  10%|#         | 60/627 [00:00<00:01, 544it/s]\n"
        "Killed\n"
    )
    assert _mineru_error(stderr) == "Killed"
