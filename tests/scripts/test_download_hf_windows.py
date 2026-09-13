import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/download_hf_windows.py"
SPEC = importlib.util.spec_from_file_location("download_hf_windows", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
REVISION = "179b7cda25486efbaaf8637d696759d9a791d8bd"
REPO = "s-zaizen/DeepSeek-V4.1-Flash-NVFP4"
STORAGE = "models--s-zaizen--DeepSeek-V4.1-Flash-NVFP4"


def stage_file(stage, name, content, *, git=False, revision=REVISION):
    source = stage / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    etag = (hashlib.sha1(f"blob {len(content)}\0".encode() + content) if git
            else hashlib.sha256(content)).hexdigest()
    metadata = stage / ".cache/huggingface/download" / f"{name}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{revision}\n{etag}\n1.0\n")
    return source, etag


def test_imports_sha256_shards_and_git_files_and_preserves_existing(tmp_path):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    shard, etag = stage_file(stage, "model-00004.safetensors", b"complete tiny shard")
    config, config_etag = stage_file(stage, "nested/config.json", b'{"ok":true}', git=True)
    snapshot = cache / STORAGE / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    existing = snapshot / "model-00001.safetensors"
    existing.write_bytes(b"already complete on Linux")
    assert helper.import_completed(stage, cache, REPO, REVISION) == {"imported": 2, "reused": 0}
    for name, expected, digest in [(shard.name, shard.read_bytes(), etag),
                                   ("nested/config.json", config.read_bytes(), config_etag)]:
        target = snapshot / name
        assert target.is_symlink()
        assert target.read_bytes() == expected
        assert target.resolve() == cache / STORAGE / "blobs" / digest
    assert existing.read_bytes() == b"already complete on Linux"
    assert not (snapshot / ".cache").exists()
    assert helper.import_completed(stage, cache, REPO, REVISION) == {"imported": 0, "reused": 2}
    assert shard.read_bytes() == b"complete tiny shard"


def test_only_final_files_with_metadata_are_imported(tmp_path):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    source, _ = stage_file(stage, "complete.safetensors", b"ready")
    (stage / "unknown.safetensors").write_bytes(b"no metadata")
    stage_file(stage, "unready.incomplete", b"partial")
    stage_file(stage, "unready.lock", b"")
    missing, _ = stage_file(stage, "missing.safetensors", b"not renamed yet")
    missing.unlink()
    (stage / ".cache/huggingface/download/complete.safetensors.lock").touch()
    helper.import_completed(stage, cache, REPO, REVISION)
    assert [p.name for p in (cache / STORAGE / "snapshots" / REVISION).iterdir()] == [source.name]


def test_truncated_or_interrupted_copy_never_publishes(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    source, etag = stage_file(stage, "model.safetensors", b"the complete payload")
    source.write_bytes(b"the complete")
    with pytest.raises(ValueError, match="checksum mismatch"):
        helper.import_completed(stage, cache, REPO, REVISION)
    root = cache / STORAGE
    assert not (root / "blobs" / etag).exists()
    assert not (root / "snapshots" / REVISION / source.name).exists()
    assert list((root / "blobs").iterdir()) == []

    source.write_bytes(b"the complete payload")
    def broken_copy(path, expected_etag, output=None):
        assert not (root / "blobs" / expected_etag).exists()
        assert not (root / "snapshots" / REVISION / source.name).exists()
        output.write(b"partial")
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(helper, "hash_file", broken_copy)
    with pytest.raises(OSError, match="interrupted"):
        helper.import_completed(stage, cache, REPO, REVISION)
    assert list((root / "blobs").iterdir()) == []
    assert not (root / "snapshots" / REVISION / source.name).exists()


@pytest.mark.parametrize("where", ["blobs", "snapshots"])
def test_conflicting_cache_file_is_preserved(tmp_path, where):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    source, etag = stage_file(stage, "model.safetensors", b"correct")
    conflict = cache / STORAGE / (f"blobs/{etag}" if where == "blobs"
                                  else f"snapshots/{REVISION}/{source.name}")
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"existing content")
    with pytest.raises(ValueError, match="conflicts|invalid checksum"):
        helper.import_completed(stage, cache, REPO, REVISION)
    assert conflict.read_bytes() == b"existing content"


@pytest.mark.parametrize("repo,revision", [("../escape", REVISION), ("a/b/c", REVISION),
                                          (REPO, "main"), (REPO, "../escape")])
def test_target_paths_are_validated(repo, revision):
    with pytest.raises(ValueError):
        helper.validate_target(repo, revision)


def test_wrong_revision_or_non_sha256_shard_is_not_published(tmp_path):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    stage_file(stage, "model.safetensors", b"payload", revision="a" * 40)
    with pytest.raises(ValueError, match="revision differs"):
        helper.import_completed(stage, cache, REPO, REVISION)
    stage_file(stage, "model.safetensors", b"payload", git=True)
    with pytest.raises(ValueError, match="SHA256"):
        helper.import_completed(stage, cache, REPO, REVISION)
    assert list((cache / STORAGE / "blobs").iterdir()) == []


def test_source_symlink_cannot_import_outside_stage(tmp_path):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    source, _ = stage_file(stage, "model.safetensors", b"payload")
    outside = tmp_path / "outside"
    outside.write_bytes(b"private data")
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        helper.import_completed(stage, cache, REPO, REVISION)


def test_repeated_import_uses_immutable_snapshot_without_rehashing(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    stage_file(stage, "model.safetensors", b"payload")
    helper.import_completed(stage, cache, REPO, REVISION)
    monkeypatch.setattr(helper, "hash_file", lambda *_: pytest.fail("rehashed completed import"))
    assert helper.import_completed(stage, cache, REPO, REVISION) == {"imported": 0, "reused": 1}


def test_source_change_after_copy_is_detected_before_publication(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    source, _ = stage_file(stage, "model.safetensors", b"payload")
    original = helper.hash_file
    def mutate_source(path, etag, output=None):
        result = original(path, etag, output)
        path.write_bytes(b"modified after copy")
        return result
    monkeypatch.setattr(helper, "hash_file", mutate_source)
    with pytest.raises(ValueError, match="changed during import"):
        helper.import_completed(stage, cache, REPO, REVISION)
    root = cache / STORAGE
    assert list((root / "blobs").iterdir()) == []
    assert not (root / "snapshots" / REVISION / source.name).exists()


def test_downloader_reads_only_explicit_token_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **kw: calls.append((a, kw))))
    monkeypatch.setenv("HF_TOKEN", "ignored environment token")
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    helper.download(tmp_path, REPO, REVISION, 32)
    assert calls[-1][1]["token"] is False
    token = tmp_path / "credential"
    token.write_text("test-token\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token))
    helper.download(tmp_path, REPO, REVISION, 32)
    assert calls[-1][1] == {"revision": REVISION, "local_dir": tmp_path,
                             "max_workers": 32, "token": "test-token"}


def test_import_only_uses_readonly_bind_and_no_network_or_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(helper, "download", lambda *_: pytest.fail("import-only downloaded"))
    monkeypatch.setattr(helper.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    helper.main(["--import-only", "--stage-dir", str(tmp_path), "--repo-id", REPO,
                 "--revision", REVISION, "--volume", "freetoken_hf-cache"])
    command, kwargs = calls[0]
    assert "--network=none" in command and "--read-only" in command
    assert f"type=bind,src={tmp_path},dst=/stage,readonly" in command
    assert "type=volume,src=freetoken_hf-cache,dst=/cache" in command
    assert "--container-import" in command
    assert command[command.index("--import-workers") + 1] == "4"
    assert kwargs == {"check": True}
    assert not any("TOKEN" in part or part in {"--env", "-e"} for part in command)


def test_serial_import_prioritizes_larger_files_then_relative_path(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    for name, content in [("small", b"s"), ("b", b"b" * 5), ("a", b"a" * 5), ("large", b"l" * 20)]:
        stage_file(stage, name, content)
    order = []
    original = helper.hash_file
    def record(path, etag, output=None):
        order.append(path.name)
        return original(path, etag, output)
    monkeypatch.setattr(helper, "hash_file", record)
    assert helper.import_completed(stage, cache, REPO, REVISION, import_workers=1) == {"imported": 4, "reused": 0}
    assert order == ["large", "a", "b", "small"]


def test_default_import_runs_four_verified_copies_concurrently(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    for index in range(4):
        stage_file(stage, f"model-{index}.safetensors", bytes([index]) * (index + 1))
    barrier = threading.Barrier(4, timeout=5)
    original = helper.hash_file
    def parallel_copy(path, etag, output=None):
        assert output is not None
        barrier.wait()
        return original(path, etag, output)
    monkeypatch.setattr(helper, "hash_file", parallel_copy)
    assert helper.import_completed(stage, cache, REPO, REVISION) == {"imported": 4, "reused": 0}
    for source in stage.glob("*.safetensors"):
        assert (cache / STORAGE / "snapshots" / REVISION / source.name).read_bytes() == source.read_bytes()


def test_parallel_files_with_same_etag_copy_once(tmp_path, monkeypatch):
    stage, cache = tmp_path / "stage", tmp_path / "cache"
    for name in ("one.safetensors", "two.safetensors"):
        stage_file(stage, name, b"identical content")
    original = helper.hash_file
    copies = []
    def count_copy(path, etag, output=None):
        if output is not None:
            copies.append(path.name)
        return original(path, etag, output)
    monkeypatch.setattr(helper, "hash_file", count_copy)
    assert helper.import_completed(stage, cache, REPO, REVISION) == {"imported": 1, "reused": 1}
    assert len(copies) == 1
    snapshot = cache / STORAGE / "snapshots" / REVISION
    assert (snapshot / "one.safetensors").resolve() == (snapshot / "two.safetensors").resolve()


def test_failed_parallel_copy_waits_for_active_cleanup_and_skips_unstarted(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    stage, cache = tmp_path / "stage", tmp_path / "cache"
    bad, bad_etag = stage_file(stage, "bad.safetensors", b"b" * 30)
    active, active_etag = stage_file(stage, "active.safetensors", b"a" * 20)
    waiting, _ = stage_file(stage, "unstarted.safetensors", b"w" * 10)
    active_started, release_active, shutdown_started = (threading.Event() for _ in range(3))
    original = helper.hash_file
    copied = []

    class ObservedExecutor(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            if cancel_futures:
                shutdown_started.set()
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def staged_copy(path, etag, output=None):
        copied.append(path.name)
        if path == bad:
            output.write(b"partial")
            assert active_started.wait(5)
            raise OSError("simulated copy failure")
        if path == active:
            active_started.set()
            assert release_active.wait(5)
        return original(path, etag, output)

    monkeypatch.setattr(helper, "ThreadPoolExecutor", ObservedExecutor)
    monkeypatch.setattr(helper, "hash_file", staged_copy)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(helper.import_completed, stage, cache, REPO, REVISION, import_workers=2)
        try:
            assert shutdown_started.wait(5)
            assert not future.done()
        finally:
            release_active.set()
        with pytest.raises(OSError, match="simulated copy failure"):
            future.result(timeout=5)
    blobs = cache / STORAGE / "blobs"
    snapshot = cache / STORAGE / "snapshots" / REVISION
    assert waiting.name not in copied
    assert not (blobs / bad_etag).exists() and not (snapshot / bad.name).exists()
    assert (blobs / active_etag).read_bytes() == active.read_bytes()
    assert not list(blobs.glob(".import-*"))


@pytest.mark.parametrize("workers", [0, -1])
def test_import_workers_must_be_positive(tmp_path, workers):
    with pytest.raises(ValueError, match="import-workers must be positive"):
        helper.import_completed(tmp_path, tmp_path / "cache", REPO, REVISION, import_workers=workers)
    with pytest.raises(SystemExit) as error:
        helper.main(["--import-only", "--stage-dir", str(tmp_path), "--repo-id", REPO,
                     "--revision", REVISION, "--import-workers", str(workers)])
    assert error.value.code == 2


def test_import_workers_reaches_docker_and_container(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(helper.subprocess, "run", lambda command, **kwargs: calls.append(command))
    options = ["--stage-dir", str(tmp_path), "--repo-id", REPO, "--revision", REVISION,
               "--import-workers", "3"]
    helper.main(["--import-only", *options])
    assert calls[0][calls[0].index("--import-workers") + 1] == "3"
    imports = []
    monkeypatch.setattr(helper, "import_completed", lambda *args, **kwargs: imports.append((args, kwargs)))
    helper.main(["--container-import", *options])
    assert imports == [((tmp_path, Path("/cache/hub"), REPO, REVISION), {"import_workers": 3})]
