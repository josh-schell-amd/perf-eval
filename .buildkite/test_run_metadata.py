#!/usr/bin/env python3
"""Stdlib-only regression tests for lib/write_run_metadata.py."""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITER = os.path.join(ROOT, "lib", "write_run_metadata.py")
sys.path.insert(0, os.path.join(ROOT, "lib"))

import write_run_metadata  # noqa: E402


def write(env):
    """Run the writer with only `env` (plus PATH) set; return (record, stdout)."""
    base = {"PATH": os.environ["PATH"]}
    if "SYSTEMROOT" in os.environ:  # Python on Windows needs it to start
        base["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run([sys.executable, WRITER, tmp], env={**base, **env},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        with open(os.path.join(tmp, "run_metadata.json")) as fh:
            return json.load(fh), result.stdout


def test_credentials_are_redacted_by_whole_word():
    for name in ("HF_TOKEN", "hf_token", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
                 "AWS_ACCESS_KEY_ID", "DB_PASSWORD", "GOOGLE_CREDENTIALS", "MY_APIKEY"):
        assert write_run_metadata.is_secret_name(name), name
    for name in ("TOKENIZERS_PARALLELISM", "VLLM_KEY_CACHE_DTYPE", "HF_HOME",
                 "MONKEY_PATCH", "VLLM_ROCM_USE_AITER"):
        assert not write_run_metadata.is_secret_name(name), name


def test_env_block_redacts_values_and_keeps_the_rest():
    env = write_run_metadata.parse_env_block(
        "HF_TOKEN=hf_abc\nTOKENIZERS_PARALLELISM=false\n\nmalformed\nA = 1 \n")
    assert env == {"HF_TOKEN": "<redacted>", "TOKENIZERS_PARALLELISM": "false", "A": "1"}


def test_prelaunch_record_is_marked_and_quiet_about_missing_version():
    record, stdout = write({"WORKLOAD_IMAGE": "vllm/vllm-openai-rocm:nightly",
                            "WORKLOAD_SERVE_COMMAND": "vllm serve m --port 1"})
    assert record["server_healthy"] is False
    assert record["vllm_version"] is None and record["vllm_version_commit"] is None
    assert "not captured" not in stdout


def test_healthy_record_carries_version_commit_separately():
    record, stdout = write({
        "WORKLOAD_SERVER_HEALTHY": "true",
        "WORKLOAD_VLLM_COMMIT": "a" * 40,
        "WORKLOAD_VLLM_VERSION": "0.11.1.dev42+g1a2b3c4d5.rocm700",
        "WORKLOAD_VLLM_VERSION_COMMIT": "1a2b3c4d5",
    })
    assert record["server_healthy"] is True
    assert record["vllm_commit"] == "a" * 40
    assert record["vllm_version_commit"] == "1a2b3c4d5"
    assert "not captured: image, image_digest, serve_command" in stdout


def test_run_type_defaults_to_adhoc_and_is_independent_of_nightly():
    record, _ = write({"NIGHTLY": "1"})
    assert (record["run_type"], record["nightly"]) == ("adhoc", True)
    record, _ = write({"PERF_EVAL_RUN_TYPE": " aiter_nightly "})
    assert (record["run_type"], record["nightly"]) == ("aiter_nightly", False)


def test_unwritable_results_dir_does_not_fail_the_run():
    with tempfile.NamedTemporaryFile() as blocker:
        # A path under a regular file can never be created as a directory.
        result = subprocess.run([sys.executable, WRITER, os.path.join(blocker.name, "x")],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "could not write" in result.stderr


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
