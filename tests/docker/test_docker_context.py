from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKER_ROOT = ROOT / "docker"


def test_dockerfile_is_pinned_and_runs_only_n03_entrypoint() -> None:
    dockerfile = (DOCKER_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@"
        "sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126"
    ) in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "n03_docker.cli"]' in dockerfile
    assert "COPY assets /opt/n03/assets" not in dockerfile
    assert dockerfile.count("/fold_") == 30
    assert "COPY assets/learned_models /opt/n03/assets/learned_models" in dockerfile
    assert "COPY . " not in dockerfile
    assert "N03_FINAL_UTILITY_V4" in dockerfile
    assert "/input" not in "\n".join(
        line for line in dockerfile.splitlines() if line.lstrip().startswith("COPY")
    )


def test_runtime_requirements_pin_numerical_and_gate_dependencies() -> None:
    rows = {
        line.strip()
        for line in (DOCKER_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "numpy==1.26.4" in rows
    assert "scipy==1.13.1" in rows
    assert "lightgbm==4.6.0" in rows
    assert "scikit-learn==1.7.0" in rows
    assert "nibabel==5.3.3" in rows
    assert "SimpleITK==2.5.3" in rows
    assert not any(row.startswith("torch") for row in rows)


def test_dockerignore_excludes_logs_smoke_assets_and_historical_predictions() -> None:
    rules = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "logs" not in "\n".join(
        line for line in rules.splitlines() if line.startswith("!")
    )
    assert "smoke_assets" not in "\n".join(
        line for line in rules.splitlines() if line.startswith("!")
    )
    assert "!assets/**" in rules
    assert "!inference/src/**" in rules
    assert "!inference/vendor/**" in rules


def test_build_script_exports_a_docker_load_compatible_archive() -> None:
    script = (ROOT / "scripts" / "build_docker_archive.sh").read_text(
        encoding="utf-8"
    )
    assert "unshare --propagation unchanged -Urm" in script
    assert "--oci-worker-no-process-sandbox" in script
    assert "type=docker,name=" in script
    assert ".docker.tar" in script
    assert "sha256sum" in script


def test_kaniko_build_script_matches_the_restricted_host_workflow() -> None:
    script = (
        ROOT / "scripts" / "build_docker_archive_kaniko.sh"
    ).read_text(encoding="utf-8")
    assert "KANIKO_ROOTFS_TAR" in script
    assert "REGISTRY_BINARY" in script
    assert "REGISTRY_CONFIG" in script
    assert "CRANE" in script
    assert 'chroot "${KANIKO_ROOT}" /kaniko/executor' in script
    assert "--force" in script
    assert "--format=legacy" in script
    assert "crane validate" not in script
    assert '"${CRANE}" validate --insecure --remote "${IMAGE_REF}"' in script
    assert "sha256sum" in script
    private_prefix = "/data/" + "coding/challenge"
    assert private_prefix not in script


def test_ordered_image_build_can_select_kaniko() -> None:
    script = (ROOT / "scripts" / "05_build_final_image.sh").read_text(
        encoding="utf-8"
    )
    assert "N03_IMAGE_BUILDER" in script
    assert "build_docker_archive_kaniko.sh" in script
