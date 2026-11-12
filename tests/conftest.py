from pathlib import Path

import pytest

from steward.config import Settings
from steward.index import Index, build_index
from steward.manifest import Manifest, load_manifest
from steward.tools import Toolbox

HEADLINE = (
    "Why did Norris get five seconds in Austin in 2024, while Verstappen got ten seconds "
    "at Turn 4 in Mexico the next week?"
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def manifest(settings: Settings) -> Manifest:
    return load_manifest(settings.manifest_path)


@pytest.fixture(scope="session")
def index_path(
    settings: Settings, manifest: Manifest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    missing = [
        s.id
        for s in manifest.sources
        if s.cache_file and not (settings.cache_dir / s.cache_file).exists()
    ]
    if missing:
        pytest.skip(f"PDF cache incomplete ({len(missing)} missing). Run `steward fetch` first.")
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(manifest, settings.cache_dir, path)
    return path


@pytest.fixture()
def toolbox(index_path: Path, manifest: Manifest, settings: Settings) -> Toolbox:
    return Toolbox(Index(index_path), manifest, settings)
