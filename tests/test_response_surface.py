import json
import tempfile
import unittest
from pathlib import Path

from trialvault.service import Vault
from trialvault.validation import VaultError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_schema():
    return json.loads((FIXTURES / "rsm_schema.json").read_text())


class StudyHarness(unittest.TestCase):
    dataset = "rsm"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vault = Vault(self.root / "vault")
        self.addCleanup(self.vault.close)
        self.vault.registry.add_schema(load_schema())
        self.vault.registry.register(
            self.dataset, "Synthetic response-surface trials", "response_surface", "owner"
        )

    def ingest(
        self, fixture, request="ingest", classification="internal", scopes=None,
        expected_version=0,
    ):
        content = (FIXTURES / fixture).read_bytes()
        return self.vault.ingestion.ingest(
            request,
            self.dataset,
            1,
            content,
            "jsonl",
            classification,
            scopes or ["research"],
            expected_version=expected_version,
        )

    def release(self, release_id="release", version=1):
        self.vault.releases.plan(
            release_id,
            [{"dataset": self.dataset, "version": version}],
            "research",
            "internal",
        )
        return self.vault.releases.approve(
            release_id, 1, "reviewer", "Synthetic evidence reviewed"
        )

    def register(self, study_id="study", **changes):
        config = {
            "factors": [
                {"field": "temperature", "low": "100", "high": "200", "center": "150"},
                {"field": "pressure", "low": "1", "high": "5", "center": "3"},
            ],
            "response": "strength",
            "batch_field": "batch",
            "center_repeat_field": "center_repeat",
            "factor_box": None,
            "process_constraints": [],
            "folds": 3,
            "bootstrap": 100,
            "seed": 1,
        }
        config.update(changes)
        return self.vault.studies.register(
            study_id, "release", self.dataset, 1, **config
        )


class ResponseSurfaceTests(StudyHarness):
    def setUp(self):
        super().setUp()
        self.ingest("rsm_interior_peak.jsonl")
        self.release()

    def test_interior_peak_finds_stationary_maximum(self):
        study = self.register()
        analysis = study["analysis"]
        self.assertTrue(analysis["fitted"])
        self.assertEqual(analysis["selected_model"]["model"], "quadratic")
        diagnostics = analysis["design_diagnostics"]["full_quadratic"]
        self.assertEqual(diagnostics["rank"], 6)
        self.assertFalse(diagnostics["deficient"])
        self.assertIsNotNone(diagnostics["condition_number"])
        stationary = analysis["recommendation"]["model"]["interior_stationary"]
        self.assertEqual(stationary["nature"], "maximum")
        recommendation = analysis["recommendation"]
        self.assertEqual(recommendation["source"], "unconstrained_stationary")
        self.assertEqual(recommendation["active_constraints"], [])
        self.assertTrue(recommendation["within_registered_factor_box"])
        self.assertTrue(recommendation["within_design_convex_hull"])
        self.assertFalse(recommendation["extrapolation_warning"])
        natural = recommendation["natural"]
        self.assertTrue(140 < natural["temperature"] < 170)
        self.assertTrue(2 < natural["pressure"] < 4)
        self.assertTrue(recommendation["model"]["global_proof"])
        self.assertFalse(recommendation["model"]["heuristic"])

    def test_coefficients_and_lack_of_fit(self):
        study = self.register()
        quadratic = study["analysis"]["models"]["quadratic"]
        estimates = quadratic["coefficient_map"]
        # Synthetic surface: 90 + 2*x0 + 1*x1 + 3*x0*x1 - 8*x0^2 - 5*x1^2
        self.assertAlmostEqual(estimates["intercept"], 90.0, places=1)
        self.assertAlmostEqual(estimates["x0*x1"], 3.0, places=1)
        self.assertAlmostEqual(estimates["x0^2"], -8.0, places=1)
        self.assertEqual(len(quadratic["coefficients"]), 6)
        for row in quadratic["coefficients"]:
            self.assertIn("estimate", row)
            self.assertIn("t_statistic", row)
            self.assertIn("p_value", row)
            self.assertIn("confidence_interval_95", row)
        lack = study["analysis"]["lack_of_fit"]
        self.assertTrue(lack["testable"])
        self.assertEqual(lack["degrees_of_freedom_pure_error"], 18)
        self.assertEqual(lack["sum_squares_pure_error"], 0.18)
        self.assertLess(lack["f_statistic"], 1e-9)
        self.assertAlmostEqual(lack["p_value"], 1.0)

    def test_center_repeats_estimate_pure_error(self):
        center = self.register()["analysis"]["center_points"]
        self.assertTrue(center["repeats_identified"])
        self.assertEqual(center["flagged_center_observations"], 3)
        self.assertEqual(center["flagged_batches"], ["batch-0", "batch-1", "batch-2"])
        self.assertTrue(center["pure_error"]["testable"])
        self.assertEqual(center["pure_error"]["degrees_of_freedom"], 2)
        self.assertEqual(center["pure_error"]["mean_square"], 0.01)

    def test_batch_isolated_cross_validation(self):
        selection = self.register()["analysis"]["selected_model"]
        self.assertEqual(selection["status"], "valid")
        self.assertEqual(selection["folds"], 3)
        candidates = {item["model"]: item for item in selection["candidates"]}
        self.assertEqual(candidates["quadratic"]["status"], "valid")
        self.assertLess(candidates["quadratic"]["rmse"], candidates["linear"]["rmse"])
        for fold in candidates["quadratic"]["folds"]:
            self.assertTrue(fold["validation_batches"])
            self.assertFalse(
                set(fold["validation_batches"]) & set(fold["training_batches"]),
                "a batch leaked across the validation boundary",
            )
        self.assertFalse(selection["selection_uses_final_validation"])

    def test_stability_resamples_whole_batches(self):
        stability = self.register()["analysis"]["stability"]
        self.assertTrue(stability["computed"])
        self.assertEqual(stability["repetitions"], 100)
        self.assertEqual(stability["successful"], 100)
        self.assertEqual(stability["optimum_predicted_mean"]["count"], 100)
        self.assertFalse(stability["optimum_jumped"])
        self.assertGreaterEqual(stability["optimum_predicted_mean"]["sd"], 0)
        self.assertEqual(
            stability["position_per_factor"]["temperature"]["count"], 100
        )

    def test_study_identity_and_determinism(self):
        first = self.register()
        again = self.register()
        self.assertEqual(first["digest"], again["digest"])
        with self.assertRaises(VaultError):
            self.register(seed=2)
        changed = self.vault.studies.evaluate("study")
        self.assertEqual(changed["digest"], first["digest"])

    def test_export_requires_document_and_blocks_after_revoke(self):
        self.register()
        self.assertTrue(self.vault.studies.status("study")["deliverable"])
        envelope = self.vault.studies.export("study", "delivery")
        self.assertEqual(envelope["kind"], "response-surface-study")
        self.assertEqual(len(envelope["digest"]), 64)
        self.assertFalse(envelope["reused"])
        again = self.vault.studies.export("study", "delivery")
        self.assertTrue(again["reused"])
        self.assertEqual(again["digest"], envelope["digest"])
        delivered = self.vault.studies.verify_export("delivery")
        self.assertTrue(delivered["valid_delivery"])
        self.vault.releases.revoke("release", "Distribution halted")
        status = self.vault.studies.status("study")
        self.assertFalse(status["deliverable"])
        self.assertIn("must not be treated", status["message"])
        self.assertFalse(self.vault.studies.verify_export("delivery")["valid_delivery"])
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.export("study", "delivery")
        self.assertEqual(caught.exception.code, "state")
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.export("study", "second-delivery")
        self.assertEqual(caught.exception.code, "state")
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.evaluate("study")
        self.assertEqual(caught.exception.code, "state")
        # The historical record remains readable for audit purposes.
        self.assertTrue(self.vault.studies.result("study")["analysis"]["fitted"])

    def test_source_is_pinned_to_frozen_release(self):
        study = self.register()
        source = study["source"]
        self.assertEqual(source["dataset"], "rsm")
        self.assertEqual(source["version"], 1)
        self.assertEqual(source["release_digest"], self.vault.releases.get("release")["digest"])
        self.assertEqual(source["rows"], 27)
        self.assertEqual(
            source["retention"]["action"], "study_pins_frozen_release_versions"
        )
        self.assertEqual(len(source["retention"]["used"]), 2)
        self.assertEqual(
            [item["version"] for item in source["pinned_versions"]], [1]
        )
        # A second dataset version must not silently change the frozen input.
        self.ingest("rsm_boundary_peak.jsonl", request="second", expected_version=1)
        self.assertEqual(
            self.vault.studies.evaluate("study")["source"]["version_digest"],
            source["version_digest"],
        )

    def test_schema_and_release_must_match(self):
        with self.assertRaises(VaultError):
            self.register(factors=[
                {"field": "missing", "low": "1", "high": "2"},
            ])
        self.assertEqual(self.vault.studies.list(), [])
        with self.assertRaises(VaultError):
            self.vault.studies.register(
                "unapproved", "missing-release", self.dataset, 1,
                factors=[
                    {"field": "temperature", "low": "100", "high": "200"},
                    {"field": "pressure", "low": "1", "high": "5"},
                ],
                response="strength",
                batch_field="batch",
                folds=2,
                bootstrap=0,
            )

    def test_cli_dispatch(self):
        from trialvault.app import dispatch

        self.register()
        self.assertEqual(
            dispatch(
                self.vault,
                {
                    "action": "study_evaluate",
                    "study_id": "study",
                },
            )["digest"],
            self.vault.studies.result("study")["digest"],
        )
        self.assertEqual(self.vault.checks.inspect()["versions"], 1)
        self.assertEqual(
            dispatch(self.vault, {"action": "study", "study_id": "study"})["id"],
            "study",
        )
        self.assertEqual(
            dispatch(
                self.vault, {"action": "study_status", "study_id": "study"}
            )["deliverable"],
            True,
        )
        listing = dispatch(self.vault, {"action": "studies"})[0]
        self.assertEqual(listing["id"], "study")
        self.assertTrue(listing["deliverable"])


class ResponseSurfaceBoundaryTests(StudyHarness):
    def test_narrow_factor_box_can_move_optimum_to_the_boundary(self):
        self.ingest("rsm_interior_peak.jsonl")
        self.release()
        # Unconstrained peak is near 157.6/3.29; clipping the box moves the
        # recommendation onto the temperature and pressure high limits.
        study = self.register(
            factor_box=[
                {"field": "temperature", "low": "100", "high": "150"},
                {"field": "pressure", "low": "1", "high": "3"},
            ]
        )
        recommendation = study["analysis"]["recommendation"]
        self.assertEqual(
            recommendation["natural"], {"temperature": 150.0, "pressure": 3.0}
        )
        self.assertEqual(recommendation["source"], "vertex")
        self.assertEqual(
            recommendation["active_constraints"],
            ["box:pressure<=high", "box:temperature<=high"],
        )
        self.assertTrue(recommendation["within_registered_factor_box"])
        self.assertTrue(recommendation["optimality_evidence"]["multipliers_nonnegative"])

    def test_box_that_keeps_peak_inside_preserves_interior_solution(self):
        self.ingest("rsm_interior_peak.jsonl")
        self.release()
        study = self.register(
            factor_box=[
                {"field": "temperature", "low": "140", "high": "160"},
                {"field": "pressure", "low": "2", "high": "4"},
            ]
        )
        recommendation = study["analysis"]["recommendation"]
        self.assertEqual(recommendation["source"], "unconstrained_stationary")
        self.assertEqual(recommendation["active_constraints"], [])

    def test_boundary_peak_reports_active_box_and_linear_constraints(self):
        self.ingest("rsm_boundary_peak.jsonl")
        self.release()
        study = self.register()
        analysis = study["analysis"]
        self.assertTrue(analysis["fitted"])
        recommendation = analysis["recommendation"]
        self.assertEqual(recommendation["model"]["interior_stationary"]["nature"], "maximum")
        self.assertFalse(
            recommendation["model"]["interior_stationary"]["within_design_bounds"]
        )
        self.assertEqual(recommendation["source"], "vertex")
        self.assertEqual(
            recommendation["natural"], {"temperature": 200.0, "pressure": 5.0}
        )
        self.assertEqual(
            recommendation["active_constraints"],
            ["box:pressure<=high", "box:temperature<=high"],
        )
        evidence = recommendation["optimality_evidence"]
        self.assertTrue(evidence["multipliers_nonnegative"])
        self.assertTrue(all(value >= 0 for value in evidence["multipliers"].values()))
        self.assertTrue(recommendation["model"]["global_proof"])

    def test_linear_process_constraint_binds_and_is_reported(self):
        self.ingest("rsm_boundary_peak.jsonl")
        self.release()
        study = self.register(
            process_constraints=[
                {
                    "id": "temperature-pressure",
                    "coefficients": {"temperature": 1, "pressure": 20},
                    "bound": "210",
                }
            ]
        )
        recommendation = study["analysis"]["recommendation"]
        self.assertEqual(recommendation["source"], "face_stationary")
        self.assertEqual(recommendation["active_constraints"], ["process:temperature-pressure"])
        self.assertTrue(recommendation["optimality_evidence"]["multipliers_nonnegative"])
        natural = recommendation["natural"]
        self.assertAlmostEqual(
            natural["temperature"] + 20 * natural["pressure"], 210, places=4
        )

    def test_extrapolation_inside_box_outside_hull(self):
        self.ingest("rsm_extrapolation.jsonl")
        self.release()
        study = self.register()
        recommendation = study["analysis"]["recommendation"]
        self.assertTrue(recommendation["within_registered_factor_box"])
        self.assertFalse(recommendation["within_design_convex_hull"])
        self.assertTrue(recommendation["extrapolation_warning"])
        self.assertIn("outside the convex hull", recommendation["extrapolation_note"])
        for observation in study["design"]["unique_design_points"]:
            self.assertNotEqual(
                observation["natural"],
                ["200", "5"],
            )
        self.assertLessEqual(
            max(
                abs(value)
                for observation in study["design"]["unique_design_points"]
                for value in observation["coded"]
            ),
            1.0,
        )

    def test_boundary_study_exports_with_its_evidence(self):
        self.ingest("rsm_boundary_peak.jsonl")
        self.release()
        self.register()
        envelope = self.vault.studies.export("study", "boundary-delivery")
        analysis = envelope["study"]["analysis"]
        self.assertEqual(
            analysis["recommendation"]["active_constraints"],
            ["box:pressure<=high", "box:temperature<=high"],
        )
        self.assertIn("retention", envelope["study"]["source"])
        self.assertEqual(
            envelope["study"]["configuration"]["seed"],
            envelope["study"]["seed"],
        )
        self.assertTrue(self.vault.studies.verify_export("boundary-delivery")["valid_delivery"])

    def test_collinear_diagnostic_study_still_exports(self):
        self.ingest("rsm_collinear.jsonl")
        self.release()
        study = self.register()
        envelope = self.vault.studies.export("study", "collinear-delivery")
        analysis = envelope["study"]["analysis"]
        self.assertFalse(analysis["fitted"])
        self.assertIsNone(analysis["recommendation"])
        self.assertTrue(analysis["design_diagnostics"]["full_quadratic"]["deficient"])
        self.assertEqual(
            envelope["study"]["configuration"]["factor_box"],
            {
                "temperature": {"low": "100", "high": "200"},
                "pressure": {"low": "1", "high": "5"},
            },
        )
        self.assertEqual(self.vault.studies.status("study")["state"], "diagnostic_only")

    def test_box_must_stay_within_registered_range(self):
        self.ingest("rsm_interior_peak.jsonl")
        self.release()
        with self.assertRaises(VaultError) as caught:
            self.register(
                factor_box=[
                    {"field": "temperature", "low": "50", "high": "250"},
                    {"field": "pressure", "low": "1", "high": "5"},
                ]
            )
        self.assertEqual(caught.exception.code, "study")


class ResponseSurfaceFailureTests(StudyHarness):
    def test_collinear_design_is_diagnosed_without_pseudo_inverse(self):
        self.ingest("rsm_collinear.jsonl")
        self.release()
        study = self.register()
        analysis = study["analysis"]
        self.assertFalse(analysis["fitted"])
        diagnostics = analysis["design_diagnostics"]["full_quadratic"]
        self.assertTrue(diagnostics["deficient"])
        self.assertEqual(diagnostics["nullity"], 3)
        self.assertIsNone(diagnostics["condition_number"])
        self.assertEqual(analysis["models"]["quadratic"]["status"], "rank_deficient")
        self.assertIsNone(analysis["recommendation"])
        self.assertIn("pseudo-inverse", analysis["reason"])
        candidate_states = {
            item["model"]: item["status"]
            for item in analysis["selected_model"]["candidates"]
        }
        self.assertEqual(candidate_states["quadratic"], "unavailable")
        self.assertEqual(study["revision"] >= 1, True)

    def test_missing_replication_cannot_judge_lack_of_fit(self):
        self.ingest("rsm_no_replication.jsonl")
        self.release()
        study = self.register()
        analysis = study["analysis"]
        self.assertFalse(analysis["fitted"])
        lack = analysis["lack_of_fit"]
        self.assertFalse(lack["testable"])
        self.assertEqual(lack["reason"], "no_independent_replicates")
        self.assertIn("cannot be judged", lack["message"])
        center = analysis["center_points"]["pure_error"]
        self.assertFalse(center["testable"])
        self.assertEqual(center["distinct_flagged_batches"], 1)
        for item in analysis["selected_model"]["candidates"]:
            self.assertEqual(item["status"], "unavailable")
            self.assertIn("at least two", item["reason"])
        self.assertIsNone(analysis["recommendation"])

    def test_repeats_within_one_batch_do_not_count_as_pure_error(self):
        # Two specimens of batch-0 share the 100/1 condition; only the 200/5
        # condition spans batches 0 and 1 and therefore supplies pure error.
        rows = [
            {"id": "a1", "batch": "batch-0", "temperature": "100", "pressure": "1",
             "strength": "10", "center_repeat": True},
            {"id": "a2", "batch": "batch-0", "temperature": "100", "pressure": "1",
             "strength": "12", "center_repeat": True},
            {"id": "b1", "batch": "batch-0", "temperature": "200", "pressure": "5",
             "strength": "20", "center_repeat": False},
            {"id": "c1", "batch": "batch-1", "temperature": "200", "pressure": "5",
             "strength": "22", "center_repeat": False},
        ]
        content = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
        self.vault.ingestion.ingest(
            "single", self.dataset, 1, content, "jsonl", "internal", ["research"]
        )
        self.release()
        study = self.register(folds=2)
        analysis = study["analysis"]
        groups = analysis["residual_diagnostics"]["pure_error_by_condition"]
        by_label = {group["condition"][0] + "/" + group["condition"][1]: group for group in groups}
        center = by_label["100/1"]
        self.assertEqual(center["independent_batches"], 1)
        self.assertFalse(center["contributes_pure_error"])
        # The 200/5 condition does span two batches and supplies pure error, but
        # the design cannot fit any model here, so lack of fit stays untestable.
        self.assertEqual(analysis["lack_of_fit"]["reason"], "no_model_fitted")
        self.assertEqual(analysis["lack_of_fit"]["degrees_of_freedom_pure_error"], 1)
        self.assertEqual(analysis["lack_of_fit"]["pure_error_sum_of_squares"], 2.0)
        center_points = analysis["center_points"]["pure_error"]
        self.assertFalse(center_points["testable"])

    def test_insufficient_observations_for_full_quadratic(self):
        rows = [
            {"id": "p1", "batch": "batch-0", "temperature": "100", "pressure": "1",
             "strength": "10", "center_repeat": False},
            {"id": "p2", "batch": "batch-1", "temperature": "200", "pressure": "5",
             "strength": "20", "center_repeat": False},
        ]
        content = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
        self.vault.ingestion.ingest(
            "tiny", self.dataset, 1, content, "jsonl", "internal", ["research"]
        )
        self.release()
        study = self.register(folds=2, bootstrap=0)
        analysis = study["analysis"]
        self.assertEqual(analysis["models"]["quadratic"]["status"], "insufficient")
        self.assertEqual(
            analysis["design_diagnostics"]["full_quadratic"]["minimum_observations"], 7
        )
        self.assertIsNone(analysis["recommendation"])

    def test_release_must_pin_the_analysed_version(self):
        self.ingest("rsm_interior_peak.jsonl", request="first")
        self.ingest("rsm_boundary_peak.jsonl", request="second", expected_version=1)
        self.vault.releases.plan(
            "release", [{"dataset": "rsm", "version": 1}], "research", "internal"
        )
        self.vault.releases.approve("release", 1, "reviewer", "Synthetic evidence")
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.register(
                "study", "release", self.dataset, 2,
                factors=[
                    {"field": "temperature", "low": "100", "high": "200"},
                    {"field": "pressure", "low": "1", "high": "5"},
                ],
                response="strength",
                batch_field="batch",
                folds=2,
                bootstrap=0,
            )
        self.assertEqual(caught.exception.code, "study")


class ResponseSurfaceThreeFactorTests(StudyHarness):
    def test_three_factor_quadratic_model(self):
        self.ingest("rsm_three_factor.jsonl")
        self.release()
        study = self.register(
            factors=[
                {"field": "temperature", "low": "130", "high": "170", "center": "150"},
                {"field": "pressure", "low": "4", "high": "6", "center": "5"},
                {"field": "ratio", "low": "0.8", "high": "1.2", "center": "1"},
            ],
            bootstrap=50,
        )
        analysis = study["analysis"]
        self.assertTrue(analysis["fitted"])
        self.assertEqual(analysis["selected_model"]["model"], "quadratic")
        diagnostics = analysis["design_diagnostics"]["full_quadratic"]
        self.assertEqual(diagnostics["rank"], 10)
        self.assertEqual(diagnostics["terms"][-3:], ["x0^2", "x1^2", "x2^2"])
        self.assertIn("x0*x1", diagnostics["terms"])
        self.assertIn("x0*x2", diagnostics["terms"])
        self.assertIn("x1*x2", diagnostics["terms"])
        recommendation = analysis["recommendation"]
        self.assertEqual(recommendation["source"], "unconstrained_stationary")
        self.assertTrue(recommendation["within_design_convex_hull"])
        self.assertEqual(
            set(recommendation["natural"]), {"temperature", "pressure", "ratio"}
        )
        self.assertTrue(recommendation["model"]["global_proof"])
        self.assertTrue(analysis["stability"]["computed"])

    def test_one_factor_is_rejected(self):
        self.ingest("rsm_three_factor.jsonl")
        self.release()
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.register(
                "study", "release", self.dataset, 1,
                factors=[{"field": "temperature", "low": "130", "high": "170"}],
                response="strength",
                batch_field="batch",
            )
        self.assertEqual(caught.exception.code, "study")

    def test_four_factors_are_rejected(self):
        self.ingest("rsm_three_factor.jsonl")
        self.release()
        factors = [
            {"field": "temperature", "low": "130", "high": "170"},
            {"field": "pressure", "low": "4", "high": "6"},
            {"field": "ratio", "low": "0.8", "high": "1.2"},
            {"field": "strength", "low": "1", "high": "2"},
        ]
        with self.assertRaises(VaultError) as caught:
            self.vault.studies.register(
                "study", "release", self.dataset, 1,
                factors=factors,
                response="strength",
                batch_field="batch",
            )
        self.assertEqual(caught.exception.code, "study")


class ResponseSurfaceSelectionTests(StudyHarness):
    def test_linear_model_selected_for_planar_surface(self):
        # A planar response with balanced batches should favour the simpler model.
        self.vault.registry.add_schema(json.loads((FIXTURES / "rsm_schema.json").read_text()))
        from trialvault.validation import canonical

        rows = []
        for temperature, pressure in [
            (100, 1), (100, 5), (200, 1), (200, 5), (150, 3),
            (150, 1), (100, 3), (200, 3), (150, 5),
        ]:
            x0 = (temperature - 150) / 50
            x1 = (pressure - 3) / 2
            for batch in range(4):
                rows.append(
                    {
                        "id": "lin-%s-%s-%s" % (temperature, pressure, batch),
                        "batch": "batch-%d" % batch,
                        "temperature": str(temperature),
                        "pressure": str(pressure),
                        "strength": "%.4f" % (10 + 3 * x0 - 2 * x1 + (batch - 1.5) * 0.01),
                        "center_repeat": temperature == 150 and pressure == 3,
                    }
                )
        content = ("\n".join(canonical(row) for row in rows) + "\n").encode()
        self.vault.ingestion.ingest(
            "linear", self.dataset, 1, content, "jsonl", "internal", ["research"]
        )
        self.release()
        study = self.register(folds=3, bootstrap=10)
        self.assertEqual(study["analysis"]["selected_model"]["model"], "linear")
        recommendation = study["analysis"]["recommendation"]
        self.assertEqual(recommendation["natural"], {"temperature": 200.0, "pressure": 1.0})
        self.assertEqual(recommendation["source"], "vertex")
        # A planar objective has no stationary point; the maximum sits on a vertex.
        self.assertIsNone(recommendation["model"]["interior_stationary"])
        self.assertTrue(recommendation["model"]["planar_objective"])


class ResponseSurfaceStabilityTests(StudyHarness):
    def test_boundary_switch_is_reported_when_resampling(self):
        self.ingest("rsm_ambiguous_peak.jsonl")
        self.release()
        study = self.register(bootstrap=300, seed=3)
        recommendation = study["analysis"]["recommendation"]
        self.assertEqual(recommendation["source"], "unconstrained_stationary")
        self.assertEqual(recommendation["model"]["interior_stationary"]["nature"], "maximum")
        stability = study["analysis"]["stability"]
        self.assertTrue(stability["computed"])
        self.assertTrue(stability["optimum_jumped"])
        self.assertFalse(stability["optimum_jumped_to_different_constraints"])
        self.assertTrue(stability["optimum_jumped_to_different_boundary"])
        self.assertEqual(stability["reference_boundary_pattern"], ["interior", "interior"])
        patterns = {tuple(item["pattern"]): item["count"] for item in stability["boundary_patterns"]}
        self.assertGreater(patterns[("interior", "interior")], 0)
        self.assertTrue(
            ("low", "interior") in patterns or ("high", "interior") in patterns
        )
        self.assertEqual(sum(patterns.values()), stability["successful"])

    def test_infeasible_process_constraints_emit_no_recommendation(self):
        self.ingest("rsm_interior_peak.jsonl")
        self.release()
        study = self.register(
            process_constraints=[
                {
                    "id": "below-range",
                    "coefficients": {"temperature": 1},
                    "bound": "90",
                }
            ]
        )
        analysis = study["analysis"]
        self.assertTrue(analysis["fitted"])
        self.assertIsNone(analysis["recommendation"])
        self.assertIn("no feasible point", analysis["unavailable_recommendation_reason"])
        self.assertFalse(study["analysis"]["stability"]["computed"])


class ResponseSurfaceDesignRulesTests(unittest.TestCase):
    def test_optimiser_finds_global_maximum_on_a_face(self):
        from trialvault.studies import optimizer

        constraints = []
        for index in range(2):
            for sign in (1.0, -1.0):
                row = [0.0, 0.0]
                row[index] = sign
                constraints.append(
                    {"id": "c%d_%s" % (index, sign), "row": row, "bound": 1.0, "kind": "box"}
                )
        # Maximiser at z=(0.4,-0.6), strictly interior.
        Q = [[-3.0, 0.0], [0.0, -2.0]]
        linear = [2.4, -2.4]
        outcome = optimizer.maximise(10.0, linear, Q, constraints)
        self.assertEqual(outcome["status"], "optimal")
        best = outcome["best"]
        self.assertAlmostEqual(best["coded"][0], 0.4, places=9)
        self.assertAlmostEqual(best["coded"][1], -0.6, places=9)
        self.assertEqual(best["active_constraints"], [])
        # Constrain the sum so the interior optimum becomes infeasible.
        constraints.append(
            {"id": "sum", "row": [1.0, 1.0], "bound": -0.5, "kind": "linear"}
        )
        outcome = optimizer.maximise(10.0, linear, Q, constraints)
        best = outcome["best"]
        self.assertAlmostEqual(best["coded"][0] + best["coded"][1], -0.5, places=9)
        self.assertIn("sum", best["active_constraints"])
        self.assertTrue(outcome["global_proof"])

    def test_optimiser_reports_unbounded_direction_without_bounding(self):
        from trialvault.studies import optimizer

        constraints = []
        for index in range(2):
            for sign in (1.0, -1.0):
                row = [0.0, 0.0]
                row[index] = sign
                constraints.append(
                    {"id": "c%d_%s" % (index, sign), "row": row, "bound": 1.0, "kind": "box"}
                )
        # Convex objective: the maximum is on the boundary, never a saddle.
        outcome = optimizer.maximise(0.0, [0.0, 0.0], [[2.0, 0.0], [0.0, 2.0]], constraints)
        best = outcome["best"]
        self.assertEqual(best["source"], "vertex")
        self.assertTrue(all(abs(value) == 1.0 for value in best["coded"]))
