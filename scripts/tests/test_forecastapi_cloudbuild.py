"""Static checks on forecastapi/cloudbuild.yaml (the CD path for Cloud Run forecast-api)."""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "forecastapi" / "cloudbuild.yaml").read_text(encoding="utf-8"))


def _step(step_id):
    return next(s for s in CFG["steps"] if s["id"] == step_id)


def test_step_order_gates_traffic_on_the_smoke_test():
    assert [s["id"] for s in CFG["steps"]] == ["Test", "Buildpack", "Deploy", "Smoke", "Promote"]


def test_images_are_pinned_by_digest():
    for s in CFG["steps"]:
        assert "@sha256:" in s["name"], s["id"]
    assert any(re.match(r"--builder=gcr\.io/buildpacks/builder(:[\w.-]+)?@sha256:", a) for a in _step("Buildpack")["args"])


def test_tests_include_the_harness_equivalence_check():
    script = _step("Test")["args"][-1]
    assert "pytest -q forecastapi" in script and "tests/test_forecastapi_models.py" in script


def test_deploy_is_public_and_candidate_first():
    script = _step("Deploy")["args"][-1]
    # public by decision (invoker check disabled live on 2026-10-03); bounded instances
    assert "--no-invoker-iam-check" in script and "--no-allow-unauthenticated" not in script
    assert "--max-instances=$_MAX_INSTANCES" in script
    assert "--no-traffic --tag=candidate" in script
    assert "COMMIT_SHA=$COMMIT_SHA" in script
    assert "swiftbackend" not in script


def test_shell_variables_are_escaped_from_substitution():
    # Cloud Build substitutes $NAME / ${NAME}; bash variables must be written $$name.
    allowed = set(CFG["substitutions"]) | {"PROJECT_ID", "BUILD_ID", "COMMIT_SHA", "REPO_NAME", "SHORT_SHA", "BRANCH_NAME"}
    for s in CFG["steps"]:
        text = "\n".join(str(a) for a in s.get("args", []))
        for name in re.findall(r"(?<!\$)\$\{?([A-Za-z_][A-Za-z0-9_]*)", text):
            assert name in allowed, f"{s['id']}: unescaped ${name}"


def test_smoke_calls_like_the_browser_catalog_served_and_undeployed():
    script = _step("Smoke")["args"][-1]
    for needle in (
        "run.googleapis.com/invoker-iam-disabled", "access-control-allow-origin", "?list=models",
        "?model=served", "?model=lstm", '"source": "local model"', '"lead_min"',
    ):
        assert needle in script
    # anonymous on purpose: no identity token, so no 403 expectation either
    assert "Authorization: Bearer" not in script and 'test "$$code" = "403"' not in script


def test_builder_ships_the_python_the_tests_use():
    # builder:latest is google-24 (Python 3.13 / 3.14 only); the suites and the harness run 3.11.
    step = _step("Buildpack")
    assert any(a.startswith("--builder=gcr.io/buildpacks/builder:google-22@sha256:") for a in step["args"])
    assert "GOOGLE_PYTHON_VERSION=3.11.x" in step["env"]
