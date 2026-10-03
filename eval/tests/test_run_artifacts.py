import copy
import json
from pathlib import Path

import pytest

import run_artifacts as ra
from promote_report_run import PromotionError, check_promotable, promote
from run_artifacts import (
    create_run_dir,
    metric_row,
    subdir,
    update_latest_pointer,
    validate_manifest,
    write_manifest,
    write_run_readme,
)


def _component() -> dict:
    return {
        "dataset": {"source": "swiftborder.causeway.travel_times", "cache_path": "eval/data/x.csv"},
        "window": {"start": "2026-09-22 13:25:00", "end": "2026-09-26 15:20:00"},
        "models": ["XGB"],
        "horizon_minutes": 60,
        "metrics": [metric_row("XGB", "holdout", 10, 2.0, 3.0), metric_row("Persistence", "holdout", 10, 3.5, 5.0)],
        "significance": [
            {
                "label_challenger": "XGB",
                "label_reference": "Persistence",
                "n": 10,
                "mean_ae_diff_min": -1.5,
                "bootstrap_ci_low_min": -1.7,
                "bootstrap_ci_high_min": -1.4,
                "dm_pvalue": 1e-6,
                "holm_adjusted_p": 1e-6,
                "decision": "challenger",
            }
        ],
        "artifacts": ["offline/plot.png"],
    }


def _manifest(dirty: bool = False) -> dict:
    return {
        "schema_version": 2,
        "run_id": "20260930T000000Z_offline",
        "created_at_utc": "2026-09-30T00:00:00Z",
        "components": ["offline"],
        "provenance": {"git_sha": "abc", "git_dirty": dirty, "code_sha256": "f" * 64},
        "offline": _component(),
    }


@pytest.fixture(autouse=True)
def _tree_matches_fixture(monkeypatch):
    """Promotion compares the run's code hash with the tree; fixtures use 'f' * 64."""
    import promote_report_run

    monkeypatch.setattr(promote_report_run, "code_fingerprint", lambda: "f" * 64)


def test_promote_refuses_when_code_changed_since_the_run(tmp_path, monkeypatch):
    import promote_report_run

    monkeypatch.setattr(promote_report_run, "code_fingerprint", lambda: "0" * 64)
    run_dir = _run_dir(tmp_path, _manifest())
    assert any("code changed since the run" in p for p in check_promotable(run_dir, allow_dirty=True))


def _run_dir(tmp_path: Path, manifest: dict) -> Path:
    run_dir = tmp_path / manifest["run_id"]
    (run_dir / "offline").mkdir(parents=True)
    (run_dir / "offline" / "plot.png").write_bytes(b"png")
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_run_readme(run_dir, manifest)
    return run_dir


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
    assert data["schema_version"] == 2
    assert "code_sha256" in data["provenance"]
    assert (run_dir / "README.md").is_file()
    assert json.loads((tmp_path / "LATEST.json").read_text())["path"] == run_dir.name


def test_artifact_that_escapes_the_run_dir_is_rejected(tmp_path: Path):
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "plot.png").write_bytes(b"png")
    m = _manifest()
    run_dir = _run_dir(tmp_path, m)
    m["offline"]["artifacts"] = ["../other/plot.png"]
    problems = validate_manifest(m, run_dir)
    assert any("escapes the run directory" in p for p in problems), problems
    assert not any("not found" in p for p in problems)


def test_valid_manifest_has_no_problems(tmp_path: Path):
    m = _manifest()
    assert validate_manifest(m, _run_dir(tmp_path, m)) == []


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda m: m.update(schema_version=1), "schema_version"),
        (lambda m: m["provenance"].pop("git_sha"), "provenance.git_sha"),
        (lambda m: m["offline"].pop("significance"), "offline.significance missing"),
        (lambda m: m["offline"]["window"].pop("end"), "window needs start and end"),
        (lambda m: m["offline"].update(metrics=[]), "metrics is empty"),
        (lambda m: m["offline"]["metrics"][0].pop("rmse_min"), "missing ['rmse_min']"),
        (lambda m: m["offline"]["significance"][0].pop("decision"), "missing ['decision']"),
        (lambda m: m["components"].append("bqml"), "'bqml' listed but missing"),
        (lambda m: m["components"].append("mystery"), "unknown component"),
        (lambda m: m["offline"]["dataset"].update(cache_path="D:/Git Repositories/x.csv"), "absolute path"),
        (lambda m: m["offline"].update(artifacts=["offline/missing.png"]), "not found"),
    ],
)
def test_invalid_manifests_are_reported(tmp_path: Path, mutate, needle):
    base = _manifest()
    run_dir = _run_dir(tmp_path, base)
    m = copy.deepcopy(base)
    mutate(m)
    problems = validate_manifest(m, run_dir)
    assert any(needle in p for p in problems), problems


def test_repo_relpath_and_fingerprint(tmp_path: Path):
    f = ra.EVAL_ROOT / "run_artifacts.py"
    fp = ra.file_fingerprint(f)
    assert fp["path"] == "eval/run_artifacts.py"
    assert len(fp["sha256"]) == 64 and fp["bytes"] > 0
    with pytest.raises(ValueError):
        ra.repo_relpath(tmp_path / "outside.csv")


def test_code_fingerprint_changes_with_content(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ra, "REPO_ROOT", tmp_path)
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    first = ra.code_fingerprint([a])
    a.write_bytes(b"x = 2\n")
    lf = ra.code_fingerprint([a])
    assert lf != first
    a.write_bytes(b"x = 2\r\n")  # a Windows checkout gives the same fingerprint
    assert ra.code_fingerprint([a]) == lf


def test_promote_copies_a_valid_run(tmp_path: Path):
    m = _manifest()
    run_dir = _run_dir(tmp_path, m)
    report = tmp_path / "report"
    promote(run_dir, report, runs_root=tmp_path)
    assert (report / "offline" / "plot.png").is_file()
    source = json.loads((report / "SOURCE_RUN.json").read_text())
    assert source["promoted_from"] == m["run_id"]
    assert source["git_dirty"] is False
    assert (report / "README.md").read_text(encoding="utf-8").startswith("> **Committed report snapshot.**")


def test_promote_refuses_overlapping_source_and_report_directories(tmp_path: Path):
    m = _manifest()
    run_dir = _run_dir(tmp_path, m)

    with pytest.raises(PromotionError, match="must not overlap"):
        promote(run_dir, run_dir, runs_root=tmp_path)

    assert (run_dir / "run.json").is_file()
    assert (run_dir / "offline" / "plot.png").is_file()


def test_promote_refuses_invalid_run_and_leaves_report_alone(tmp_path: Path):
    m = _manifest()
    m["offline"]["metrics"] = []
    run_dir = _run_dir(tmp_path, m)
    report = tmp_path / "report"
    report.mkdir()
    (report / "keep.txt").write_text("old")
    with pytest.raises(PromotionError):
        promote(run_dir, report, runs_root=tmp_path)
    assert (report / "keep.txt").is_file()


def test_promote_refuses_dirty_unless_allowed(tmp_path: Path):
    m = _manifest(dirty=True)
    run_dir = _run_dir(tmp_path, m)
    assert any("dirty" in p for p in check_promotable(run_dir))
    with pytest.raises(PromotionError):
        promote(run_dir, tmp_path / "report", runs_root=tmp_path)
    promote(run_dir, tmp_path / "report", allow_dirty=True, runs_root=tmp_path)
    readme = (tmp_path / "report" / "README.md").read_text(encoding="utf-8")
    assert "uncommitted code" in readme
    assert json.loads((tmp_path / "report" / "SOURCE_RUN.json").read_text())["git_dirty"] is True


def _sig(label, p, diff, lo, hi, decision, family="f"):
    return {
        "label_challenger": label, "label_reference": "ref", "n": 100, "mean_ae_diff_min": diff,
        "bootstrap_ci_low_min": lo, "bootstrap_ci_high_min": hi, "dm_pvalue": p / 2,
        "dm_pvalue_two_sided": p, "holm_adjusted_p": p, "alpha": 0.05, "decision": decision, "family": family,
    }


def test_multiplicity_summary_recomputes_decisions_over_the_whole_run():
    manifest = {
        "components": ["a", "b"],
        "a": {"significance": [_sig("strong", 1e-6, -0.3, -0.4, -0.2, "challenger"), _sig("weak", 0.02, -0.1, -0.2, -0.01, "challenger")]},
        "b": {"significance": [_sig("few", 0.001, -0.5, -0.9, -0.1, "insufficient data")] + [_sig(f"n{i}", 0.9, 0.0, -0.1, 0.1, "not significant") for i in range(4)]},
    }
    out = ra.multiplicity_summary(manifest)
    assert out["n_comparisons"] == 7
    changed = {c["challenger"]: c for c in out["changed"]}
    assert set(changed) == {"weak"}  # 0.02 x 6 > 0.05 after run-wide Holm
    assert changed["weak"]["decision_global"] == "not significant"
    assert out["transitions"]["insufficient data -> insufficient data"] == 1


def test_provenance_records_platform_and_full_environment():
    prov = ra.provenance()
    assert prov["platform"]["system"]
    env = prov["environment"]
    names = [d.split("==")[0].lower() for d in env["distributions"]]
    assert names == sorted(names) and len(names) == len(set(names))
    assert any(d.lower().startswith("numpy==") for d in env["distributions"])
    assert len(env["sha256"]) == 64
