import json
from pathlib import Path

from run_artifacts import (
    create_run_dir,
    subdir,
    update_latest_pointer,
    write_manifest,
    write_run_readme,
)


def test_create_run_dir_and_manifest(tmp_path: Path):
    run_dir = create_run_dir(components=["offline"], runs_root=tmp_path)
    assert run_dir.is_dir()
    subdir(run_dir, "offline")
    manifest = {"components": ["offline"], "offline": {"models": ["XGB"]}}
    write_manifest(run_dir, manifest)
    write_run_readme(run_dir, manifest)
    update_latest_pointer(run_dir, manifest, runs_root=tmp_path)

    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert data["components"] == ["offline"]
    assert (run_dir / "README.md").is_file()
