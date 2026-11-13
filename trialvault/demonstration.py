import json
import tempfile
from pathlib import Path
from .service import Vault
from .validation import canonical
from .releases.exporter import verify_directory

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def demonstrate_response_surface():
    """Walk an interior-peak response-surface study through to export."""
    with tempfile.TemporaryDirectory() as directory:
        with Vault(Path(directory) / "vault") as vault:
            schema = json.loads((FIXTURES / "rsm_schema.json").read_text())
            vault.registry.add_schema(schema)
            vault.registry.register(
                "trials", "Synthetic response-surface trials", "response_surface", "owner"
            )
            content = (FIXTURES / "rsm_interior_peak.jsonl").read_bytes()
            ingested = vault.ingestion.ingest(
                "ingest", "trials", 1, content, "jsonl", "internal", ["research"]
            )
            vault.releases.plan(
                "release", [{"dataset": "trials", "version": 1}], "research", "internal"
            )
            vault.releases.approve("release", 1, "independent-reviewer", "Synthetic evidence")
            study = vault.studies.register(
                "surface-study",
                "release",
                "trials",
                1,
                factors=[
                    {"field": "temperature", "low": "100", "high": "200", "center": "150"},
                    {"field": "pressure", "low": "1", "high": "5", "center": "3"},
                ],
                response="strength",
                batch_field="batch",
                center_repeat_field="center_repeat",
                folds=3,
                bootstrap=50,
                seed=1,
            )
            exported = vault.studies.export("surface-study")
            analysis = study["analysis"]
            assert analysis["fitted"]
            assert analysis["recommendation"]["within_registered_factor_box"]
            return {
                "ingestion": ingested,
                "study": {
                    "selected_model": analysis["selected_model"]["model"],
                    "recommendation": analysis["recommendation"]["natural"],
                    "convex_hull": analysis["recommendation"][
                        "within_design_convex_hull"
                    ],
                },
                "export": exported["digest"],
            }


def demonstrate():
    with tempfile.TemporaryDirectory() as directory:
        with Vault(Path(directory) / "vault") as vault:
            schema = {
                "id": "measurements",
                "version": 1,
                "fields": [
                    {"name": "id", "type": "text"},
                    {"name": "pressure", "type": "decimal", "minimum": "0"},
                ],
                "primary_key": ["id"],
            }
            vault.registry.add_schema(schema)
            vault.registry.register(
                "trial", "Synthetic trial measurements", "measurements", "operator"
            )
            content = (
                canonical({"id": "sample-A", "pressure": "12.5"})
                + "\n"
                + canonical({"id": "sample-B", "pressure": "13"})
                + "\n"
            ).encode()
            ingested = vault.ingestion.ingest(
                "request", "trial", 1, content, "jsonl", "internal", ["research"]
            )
            vault.releases.plan(
                "release", [{"dataset": "trial", "version": 1}], "research", "internal"
            )
            vault.releases.approve(
                "release", 1, "independent-reviewer", "合成数据集 verified"
            )
            exported = vault.exporter.export(
                "export", "release", Path(directory) / "release"
            )
            verified = verify_directory(Path(directory) / "release")
            assert ingested["rows"] == 2
            assert verified["rows"] == 2
            assert exported["state"] == "complete"
            return {
                "ingestion": ingested,
                "export": exported,
                "verification": verified,
                "integrity": vault.checks.inspect(),
            }
