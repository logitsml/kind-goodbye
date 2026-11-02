"""Write-once files: reports and schedules are created once and never silently rewritten."""


import os


from pathlib import Path


def freeze(path, data):
    """Exclusive, atomic create of a write-once file: write a temporary file in the same directory, fsync, os.link it to `path` (fails if
    `path` exists), remove the temporary. If `path` exists with exactly these bytes, return False (a rerun reuses it); with other bytes,
    raise FileExistsError (a saved file is never rewritten; --force keeps the old file under a new name). Returns True on create."""
    p = Path(path)
    b = data if isinstance(data, bytes) else data.encode()
    if p.exists():
        if p.read_bytes() == b:
            return False
        raise FileExistsError(f"{p} exists with other content")
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(b)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(tmp, p)
    except FileExistsError:
        if p.read_bytes() != b:
            raise
        return False
    finally:
        tmp.unlink()
    return True

