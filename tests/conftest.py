from pathlib import Path

import pytest

from hemahier.ontology import Ontology

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ont():
    return Ontology.from_yaml(str(ROOT / "configs/ontology/mll.yaml"))
