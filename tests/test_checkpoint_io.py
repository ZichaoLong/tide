import errno
import stat
import pytest
from tidegraph import checkpoint_io
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent
from cases import ring


@pytest.mark.parametrize("failure", ["serialize", "file_fsync"])
def test_failed_checkpoint_write_is_not_published_and_can_retry(dtype, tmp_path, monkeypatch, failure):
    g, m, q, *_ = ring(dtype)
    target = tmp_path/"state.pt"
    with monkeypatch.context() as patch:
        if failure == "serialize":
            def fail(record, output):
                output.write(b"partial checkpoint")
                raise OSError(errno.ENOSPC, "injected full disk")
            patch.setattr(checkpoint_io.torch, "save", fail)
        else:
            def fail(fd):
                raise OSError(errno.ENOSPC, "injected fsync failure")
            patch.setattr(checkpoint_io.os, "fsync", fail)
        with pytest.raises(OSError, match="injected"):
            save(target, g, m, q)
    assert not target.exists() and not list(tmp_path.iterdir())
    save(target, g, m, q)
    equivalent(load(target, g, m), q)
    assert list(tmp_path.iterdir()) == [target]


def test_existing_checkpoint_and_racing_writer_are_preserved(dtype, tmp_path, monkeypatch):
    g, m, q, *_ = ring(dtype)
    target = tmp_path/"state.pt"
    save(target, g, m, q)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        save(target, g, m, q)
    assert target.read_bytes() == original
    target.unlink()
    real_save = checkpoint_io.torch.save
    def racing_save(record, output):
        real_save(record, output)
        target.write_bytes(original)
    monkeypatch.setattr(checkpoint_io.torch, "save", racing_save)
    with pytest.raises(FileExistsError):
        save(target, g, m, q)
    assert target.read_bytes() == original and list(tmp_path.iterdir()) == [target]
    equivalent(load(target, g, m), q)


def test_checkpoint_target_is_absent_during_serialization(dtype, tmp_path, monkeypatch):
    g, m, q, *_ = ring(dtype)
    target = tmp_path/"state.pt"
    real_save = checkpoint_io.torch.save
    def inspect(record, output):
        assert not target.exists()
        real_save(record, output)
        assert not target.exists()
    monkeypatch.setattr(checkpoint_io.torch, "save", inspect)
    save(target, g, m, q)
    equivalent(load(target, g, m), q)


def test_directory_fsync_error_reports_failure_after_complete_publication(dtype, tmp_path, monkeypatch):
    g, m, q, *_ = ring(dtype)
    target = tmp_path/"state.pt"
    real_fsync = checkpoint_io.os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(checkpoint_io.os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "injected directory fsync failure")
        return real_fsync(fd)
    monkeypatch.setattr(checkpoint_io.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory fsync"):
        save(target, g, m, q)
    assert list(tmp_path.iterdir()) == [target]
    equivalent(load(target, g, m), q)
    with pytest.raises(FileExistsError):
        save(target, g, m, q)
