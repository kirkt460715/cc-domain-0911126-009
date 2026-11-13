"""Response-surface studies over frozen, approved release data.

A study registers its design once, reads the exact versions pinned by an
approved release, fits linear and full quadratic models, diagnoses the design,
selects a model with batch-isolated cross-validation, optimises within the
registered factor box and linear process constraints, and records a stability
summary from batch resampling.  The record keeps its configuration and seed so
the same study can be recomputed and compared.
"""

import hashlib
import math
import random

from ..validation import (
    decimal,
    decimal_text,
    digest,
    integer,
    list_value,
    name,
    object_value,
    parse_json,
    reject,
    VaultError,
)
from . import design, numeric, optimizer

MODELS = ("linear", "quadratic")
MIN_FACTORS = 2
MAX_FACTORS = 3
_MAX_RESIDUAL_ROWS = 1000


class ResponseSurfaceStudies:
    def __init__(self, database, registry, ingestion, releases):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        self.releases = releases
        with database.transaction() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY,request TEXT NOT NULL,"
                "result TEXT NOT NULL,state TEXT NOT NULL,digest TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS study_exports(id TEXT PRIMARY KEY,study TEXT NOT NULL,"
                "payload TEXT NOT NULL,digest TEXT NOT NULL,state TEXT NOT NULL)"
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        study_id,
        release_id,
        dataset,
        version,
        factors,
        response,
        batch_field,
        center_repeat_field=None,
        factor_box=None,
        process_constraints=None,
        folds=3,
        bootstrap=200,
        seed=1,
    ):
        name(study_id)
        integer(version, 1)
        if (
            not isinstance(factors, list)
            or not MIN_FACTORS <= len(factors) <= MAX_FACTORS
        ):
            reject(
                "study",
                "response-surface study supports two to three continuous factors",
                minimum_factors=MIN_FACTORS,
                maximum_factors=MAX_FACTORS,
            )
        parsed_factors = self._parse_factors(factors)
        response_field = name(response)
        batch = name(batch_field)
        center = name(center_repeat_field) if center_repeat_field is not None else None
        box = self._parse_box(factor_box, parsed_factors)
        constraints = self._parse_constraints(process_constraints or [], parsed_factors)
        integer(folds, 2, 10)
        integer(bootstrap, 0, 2000)
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            reject("study", "seed must be an integer or bounded text")
        if isinstance(seed, str) and len(seed) > 256:
            reject("study", "seed must be bounded text")
        request = {
            "release": name(release_id),
            "dataset": name(dataset),
            "version": version,
            "factors": parsed_factors,
            "response": response_field,
            "batch_field": batch,
            "center_repeat_field": center,
            "factor_box": box,
            "process_constraints": constraints,
            "folds": folds,
            "bootstrap": bootstrap,
            "seed": seed,
        }
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM studies WHERE id=?", (study_id,))
            if old:
                if old["request"] != _canonical(request):
                    reject("conflict", "study identity differs")
                return self.result(study_id)
            # Validate the frozen source before recording the study so a bad
            # release reference does not leave an unevaluated registration.
            self._check_source(request)
            self.db.audit("study_registered", study_id, request)
            con.execute(
                "INSERT INTO studies VALUES(?,?,?,'registered','')",
                (study_id, _canonical(request), ""),
            )
        return self.evaluate(study_id)

    def _check_source(self, request):
        release = self.releases.get(request["release"])
        eligibility = self.releases.eligibility(request["release"])
        if release["state"] != "approved" or not eligibility["eligible"]:
            reject(
                "state",
                "study source release is not approved or no longer eligible",
                release=request["release"],
                problems=eligibility["problems"],
            )
        pinned = {
            (item["dataset"], item["version"])
            for item in release["manifest"]["versions"]
        }
        if (request["dataset"], request["version"]) not in pinned:
            reject("study", "release does not pin the requested dataset version")
        record = self.registry.version(request["dataset"], request["version"])
        if record["state"] != "active":
            reject("state", "study dataset version is withdrawn")
        schema = self.registry.schema(
            self.registry.dataset(request["dataset"])["schema_id"],
            record["schema_version"],
        )
        required = [factor["field"] for factor in request["factors"]] + [
            request["response"],
            request["batch_field"],
        ]
        if request["center_repeat_field"]:
            required.append(request["center_repeat_field"])
        for field in required:
            if field not in schema.by_name:
                reject("study", "study field absent from dataset contract", field=field)
        return release, record


    def _parse_factors(self, factors):
        parsed = []
        seen = set()
        for factor in list_value(factors, 1, MAX_FACTORS):
            object_value(
                factor, {"field", "low", "high", "center"}, {"field", "low", "high"}
            )
            field = name(factor["field"])
            if field in seen:
                reject("study", "duplicate factor field")
            seen.add(field)
            low = decimal(factor["low"])
            high = decimal(factor["high"])
            if not low < high:
                reject("study", "factor bounds must increase")
            center = decimal(factor.get("center", (low + high) / 2))
            if not low < center < high:
                reject("study", "factor center must lie strictly inside its bounds")
            parsed.append(
                {
                    "field": field,
                    "low": decimal_text(low),
                    "high": decimal_text(high),
                    "center": decimal_text(center),
                }
            )
        return parsed

    def _parse_box(self, factor_box, factors):
        declared = {}
        for entry in list_value(factor_box or [], 0, MAX_FACTORS):
            object_value(entry, {"field", "low", "high"}, {"field", "low", "high"})
            field = name(entry["field"])
            if field in declared:
                reject("study", "duplicate factor box entry")
            declared[field] = entry
        known = {factor["field"] for factor in factors}
        if declared and set(declared) != known:
            reject("study", "factor box must declare every factor exactly once")
        result = {}
        for factor in factors:
            low = decimal(factor["low"])
            high = decimal(factor["high"])
            if factor["field"] in declared:
                low = decimal(declared[factor["field"]]["low"])
                high = decimal(declared[factor["field"]]["high"])
                if not low < high:
                    reject("study", "box bounds must increase")
                if low < decimal(factor["low"]) or high > decimal(factor["high"]):
                    reject(
                        "study",
                        "study box cannot exceed the registered experiment range",
                        field=factor["field"],
                    )
            result[factor["field"]] = {
                "low": decimal_text(low),
                "high": decimal_text(high),
            }
        return result

    def _parse_constraints(self, constraints, factors):
        fields = [factor["field"] for factor in factors]
        parsed = []
        seen = set()
        for index, constraint in enumerate(list_value(constraints, 0, 12)):
            object_value(
                constraint,
                {"id", "coefficients", "bound"},
                {"coefficients", "bound"},
            )
            constraint_id = name(constraint.get("id", "c%d" % index))
            if constraint_id in seen:
                reject("study", "duplicate process constraint id")
            seen.add(constraint_id)
            coefficients = object_value(constraint["coefficients"])
            if set(coefficients) - set(fields):
                reject("study", "process constraint references unknown factor")
            if not coefficients:
                reject("study", "process constraint must reference a factor")
            encoded = [0.0] * len(fields)
            for field, value in coefficients.items():
                encoded[fields.index(field)] = float(decimal(value))
            parsed.append(
                {
                    "id": constraint_id,
                    "coefficients": {
                        field: decimal_text(decimal(value))
                        for field, value in coefficients.items()
                    },
                    "bound": decimal_text(decimal(constraint["bound"])),
                    "natural_row": encoded,
                }
            )
        return parsed

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _request(self, study_id):
        row = self.db.one("SELECT request FROM studies WHERE id=?", (study_id,))
        if row is None:
            reject("missing", "study not found")
        return parse_json(row["request"])

    def evaluate(self, study_id):
        request = self._request(study_id)
        release, record = self._check_source(request)
        rows = self.ingestion.rows(request["dataset"], request["version"])
        data = self._extract(request, record, rows)
        analysis = self._analyse(request, data)
        payload = {
            "id": study_id,
            "source": {
                "release": request["release"],
                "release_digest": release["digest"],
                "release_scope": release["manifest"]["scope"],
                "release_clearance": release["manifest"]["clearance"],
                "dataset": request["dataset"],
                "version": request["version"],
                "version_digest": record["digest"],
                "artifact": record["artifact"],
                "rows": len(rows),
                "pinned_versions": [
                    {
                        "dataset": item["dataset"],
                        "version": item["version"],
                        "role": "study_input",
                    }
                    for item in release["manifest"]["versions"]
                ],
                "retention": {
                    "action": "study_pins_frozen_release_versions",
                    "note": "study never mutates, patches or deletes the source versions",
                    "used": {
                        "dataset": request["dataset"],
                        "version": request["version"],
                    },
                    "other_pinned": [
                        {"dataset": item["dataset"], "version": item["version"]}
                        for item in release["manifest"]["versions"]
                        if (item["dataset"], item["version"])
                        != (request["dataset"], request["version"])
                    ],
                },
            },
            "design": data["public_design"],
            "configuration": {
                "factors": request["factors"],
                "response": request["response"],
                "batch_field": request["batch_field"],
                "center_repeat_field": request["center_repeat_field"],
                "factor_box": request["factor_box"],
                "process_constraints": [
                    {
                        "id": item["id"],
                        "coefficients": item["coefficients"],
                        "bound": item["bound"],
                    }
                    for item in request["process_constraints"]
                ],
                "folds": request["folds"],
                "bootstrap": request["bootstrap"],
                "seed": request["seed"],
            },
            "seed": request["seed"],
            "analysis": analysis,
        }
        content_digest = digest(payload)
        previous = None
        revision = 1
        old = self.db.one("SELECT result FROM studies WHERE id=?", (study_id,))
        if old and old["result"]:
            previous_result = parse_json(old["result"])
            previous = previous_result.get("digest")
            revision = previous_result.get("revision", 1)
            if previous is not None and previous != content_digest:
                revision += 1
                payload["previous_digest"] = previous
        result = dict(payload)
        result["digest"] = content_digest
        result["revision"] = revision
        with self.db.transaction() as con:
            con.execute(
                "UPDATE studies SET result=?,state=?,digest=? WHERE id=?",
                (
                    _canonical(result),
                    "computed" if analysis.get("fitted") else "diagnostic_only",
                    content_digest,
                    study_id,
                ),
            )
            self.db.audit(
                "study_computed",
                study_id,
                {
                    "digest": result["digest"],
                    "release": request["release"],
                    "version": request["version"],
                },
            )
        return result

    def result(self, study_id):
        row = self.db.one("SELECT result,state,digest FROM studies WHERE id=?", (study_id,))
        if row is None:
            reject("missing", "study not found")
        if not row["result"]:
            reject("state", "study has not been evaluated")
        result = parse_json(row["result"])
        payload = {key: value for key, value in result.items() if key in _PAYLOAD_FIELDS}
        if result.get("digest") != row["digest"] or digest(payload) != row["digest"]:
            reject("integrity", "study result digest mismatch", study=study_id)
        return result

    def _extract(self, request, record, rows):
        factor_fields = [factor["field"] for factor in request["factors"]]
        needed = factor_fields + [request["response"], request["batch_field"]]
        if request["center_repeat_field"]:
            needed.append(request["center_repeat_field"])
        schema = self.registry.schema(
            self.registry.dataset(request["dataset"])["schema_id"],
            record["schema_version"],
        )
        for field in needed:
            if field not in schema.by_name:
                reject("study", "study field absent from dataset contract", field=field)
        observations = []
        for index, row in enumerate(rows, 1):
            values = []
            texts = []
            for field in factor_fields:
                value = row.get(field)
                if value is None:
                    reject("study", "factor value is null", row=index, field=field)
                values.append(float(decimal(value)))
                texts.append(decimal_text(decimal(value)))
            response = row.get(request["response"])
            if response is None:
                reject("study", "response value is null", row=index, field=request["response"])
            batch = row.get(request["batch_field"])
            if batch is None:
                reject("study", "batch value is null", row=index, field=request["batch_field"])
            center = (
                row.get(request["center_repeat_field"])
                if request["center_repeat_field"]
                else None
            )
            observations.append(
                {
                    "row": index,
                    "values": values,
                    "response": float(decimal(response)),
                    "batch": str(batch),
                    "center_repeat": bool(center) if center is not None else False,
                    "natural": texts,
                }
            )
        if not observations:
            reject("study", "dataset version has no observations")
        encoded = []
        for observation in observations:
            coded = []
            for index, factor in enumerate(request["factors"]):
                low = float(decimal(factor["low"]))
                high = float(decimal(factor["high"]))
                center = float(decimal(factor["center"]))
                coded.append(
                    (observation["values"][index] - center) / ((high - low) / 2.0)
                )
            observation["coded"] = coded
            encoded.append(coded)
        unique = {}
        for observation in observations:
            key = tuple(round(value, 9) for value in observation["values"])
            unique.setdefault(key, observation)
        unique_points = [
            {
                "natural": unique[key]["natural"],
                "coded": [numeric.clean(value, 10) for value in unique[key]["coded"]],
            }
            for key in sorted(unique)
        ]
        return {
            "observations": observations,
            "encoded": encoded,
            "response": [observation["response"] for observation in observations],
            "batches": [observation["batch"] for observation in observations],
            "conditions": [
                tuple(round(value, 9) for value in observation["values"])
                for observation in observations
            ],
            "labels": [observation["natural"] for observation in observations],
            "public_design": {
                "factors": request["factors"],
                "response": request["response"],
                "observations": len(observations),
                "batches": sorted({observation["batch"] for observation in observations}),
                "design_points": len(unique),
                "unique_design_points": unique_points[:_MAX_RESIDUAL_ROWS],
                "unique_design_points_truncated": max(
                    0, len(unique_points) - _MAX_RESIDUAL_ROWS
                ),
                "center_repeat_observations": sum(
                    observation["center_repeat"] for observation in observations
                ),
            },
        }

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _analyse(self, request, data):
        models = {
            model: design.fit(data["encoded"], data["response"], model)
            for model in MODELS
        }
        selected = self._select(request, data, models)
        analysis = {
            "design_diagnostics": self._design_diagnostics(request, data, models),
            "models": models,
            "selected_model": selected,
            "center_points": self._center_points(request, data),
        }
        if len(data["observations"]) > _MAX_RESIDUAL_ROWS:
            for model in analysis["models"].values():
                rows = model.get("residual_rows")
                if rows and len(rows) > _MAX_RESIDUAL_ROWS:
                    model["residual_rows"] = rows[:_MAX_RESIDUAL_ROWS]
                    model["residual_rows_truncated"] = len(rows)
        chosen = models.get(selected.get("model"))
        if (
            selected.get("status") != "valid"
            or chosen is None
            or chosen.get("status") != "fitted"
        ):
            diagnostics = self._unfitted_diagnostics(request, data, models)
            analysis.update(
                fitted=False,
                recommendation=None,
                residual_diagnostics=diagnostics,
                lack_of_fit=diagnostics["fitted_model_lack_of_fit"]
                or {
                    "testable": False,
                    "reason": "no_model_fitted",
                    "message": "no model could be fitted, so lack of fit cannot be judged; pure-error information remains visible",
                    "pure_error_sum_of_squares": diagnostics["pure_error"][
                        "sum_squares"
                    ],
                    "degrees_of_freedom_pure_error": diagnostics["pure_error"][
                        "degrees_of_freedom"
                    ],
                },
                reason=(
                    "no model passed batch-isolated selection; a recommendation is "
                    "withheld rather than produced from an unsupported or "
                    "pseudo-inverse solution"
                ),
                diagnostic_note=(
                    "design rank, conditioning, pure error and lack-of-fit are still "
                    "reported so the failure is inspectable"
                ),
            )
            return analysis
        analysis["fitted"] = True
        residuals = self._residual_diagnostics(request, data, chosen)
        analysis["residual_diagnostics"] = residuals
        analysis["lack_of_fit"] = residuals["lack_of_fit"]
        recommendation = self._optimise(request, data, chosen)
        analysis["recommendation"] = recommendation
        if recommendation is None:
            analysis["unavailable_recommendation_reason"] = (
                "no feasible point exists inside the registered factor box and "
                "linear process constraints, or the optimiser could not certify a candidate"
            )
        analysis["stability"] = self._stability(request, data, chosen, recommendation)
        return analysis

    def _design_diagnostics(self, request, data, models):
        factors = [factor["field"] for factor in request["factors"]]
        terms = design.term_names(len(factors), "quadratic")
        full = numeric.design_rank(design.model_matrix(data["encoded"], "quadratic"))
        linear = numeric.design_rank(design.model_matrix(data["encoded"], "linear"))
        replicated = {}
        for condition, batch in zip(data["conditions"], data["batches"]):
            replicated.setdefault(condition, set()).add(batch)
        return {
            "observations": len(data["encoded"]),
            "independent_batches": len(set(data["batches"])),
            "design_points": data["public_design"]["design_points"],
            "replicated_conditions": sum(
                1 for batches in replicated.values() if len(batches) >= 2
            ),
            "full_quadratic": {
                "terms": terms,
                "parameters": len(terms),
                "minimum_observations": len(terms) + 1,
                "rank": full["rank"],
                "deficient": full["deficient"],
                "nullity": full["nullity"],
                "condition_number": full["condition_number"],
                "singular_values": full["singular_values"],
                "aliased_basis": full["aliased_terms"],
            },
            "linear": {
                "terms": design.term_names(len(factors), "linear"),
                "parameters": len(design.term_names(len(factors), "linear")),
                "rank": linear["rank"],
                "deficient": linear["deficient"],
                "nullity": linear["nullity"],
                "condition_number": linear["condition_number"],
                "singular_values": linear["singular_values"],
            },
            "fit_information": {
                model_name: {
                    "terms": model["terms"],
                    "parameters": model["parameters"],
                    "rows": model["rows"],
                    "residual_df": model["residual_df"],
                    "rank": model["design_rank"]["rank"],
                    "condition_number": model["design_rank"]["condition_number"],
                    "status": model["status"],
                }
                for model_name, model in models.items()
            },
            "note": (
                "columns are scaled to unit length before rank and condition "
                "analysis; a deficient design is reported and never solved by "
                "a pseudo-inverse"
            ),
        }

    def _unfitted_diagnostics(self, request, data, models):
        groups, pure_ss, pure_df = design.replicate_groups(
            data["conditions"],
            data["batches"],
            data["response"],
            data["labels"],
        )
        by_model = {}
        for model_name, model in models.items():
            if model.get("status") != "fitted":
                by_model[model_name] = {
                    "status": "unavailable",
                    "reason": model.get("reason", "model did not fit"),
                    "lack_of_fit": {
                        "testable": False,
                        "reason": "model_unavailable_pure_error_only",
                        "degrees_of_freedom_pure_error": pure_df,
                        "pure_error_sum_of_squares": numeric.clean(pure_ss)
                        if pure_df
                        else None,
                        "message": (
                            "lack of fit cannot be judged because the model did not fit"
                        ),
                    },
                }
                continue
            lack = design.lack_of_fit(
                model["anova"]["sum_squares_residual"],
                model["residual_df"],
                pure_ss,
                pure_df,
            )
            if pure_df > 0:
                lack["sum_squares_residual"] = numeric.clean(
                    model["anova"]["sum_squares_residual"]
                )
            by_model[model_name] = {
                "status": "fitted",
                "r_squared": model["anova"]["r_squared"],
                "lack_of_fit": lack,
            }
        return {
            "pure_error_by_condition": groups,
            "pure_error": {
                "sum_squares": numeric.clean(pure_ss) if pure_df else None,
                "degrees_of_freedom": pure_df,
                "conditions_contributing": sum(
                    1 for group in groups if group["contributes_pure_error"]
                ),
                "rule": "only distinct independent batches at the same condition contribute pure error",
            },
            "lack_of_fit_by_model": by_model,
            "fitted_model_lack_of_fit": next(
                (
                    item["lack_of_fit"]
                    for name, item in by_model.items()
                    if item["status"] == "fitted"
                ),
                None,
            ),
        }

    def _residual_diagnostics(self, request, data, model):
        groups, pure_ss, pure_df = design.replicate_groups(
            data["conditions"],
            data["batches"],
            data["response"],
            data["labels"],
        )
        residual_sse = model["anova"]["sum_squares_residual"]
        lack = design.lack_of_fit(
            residual_sse,
            model["residual_df"],
            pure_ss,
            pure_df,
        )
        if pure_df > 0:
            lack["sum_squares_residual"] = numeric.clean(residual_sse)
            lack["note"] = (
                "lack-of-fit sum of squares is residual sum of squares minus the "
                "pure-error contribution of independent batches at identical conditions"
            )
        return {
            "residual_summary": model["residual_summary"],
            "durbin_watson": model["anova"]["durbin_watson"],
            "r_squared": model["anova"]["r_squared"],
            "adjusted_r_squared": model["anova"]["adjusted_r_squared"],
            "pure_error_by_condition": groups,
            "pure_error": {
                "sum_squares": numeric.clean(pure_ss) if pure_df else None,
                "degrees_of_freedom": pure_df,
                "conditions_contributing": sum(
                    1 for group in groups if group["contributes_pure_error"]
                ),
                "rule": "only distinct independent batches at the same condition contribute pure error",
            },
            "lack_of_fit": lack,
            "residuals": model.get("residual_rows", []),
        }

    def _center_points(self, request, data):
        if not request["center_repeat_field"]:
            return {
                "repeats_identified": False,
                "message": "no center-repeat field was registered; lack of fit is untestable",
            }
        center_observations = [
            observation
            for observation in data["observations"]
            if all(abs(value) <= 1e-9 for value in observation["coded"])
        ]
        flagged = [
            observation for observation in center_observations if observation["center_repeat"]
        ]
        batches = sorted({observation["batch"] for observation in flagged})
        per_batch = {}
        for observation in flagged:
            per_batch.setdefault(observation["batch"], []).append(observation["response"])
        means = [sum(values) / len(values) for values in per_batch.values()]
        pure = None
        if len(means) >= 2:
            overall = sum(means) / len(means)
            ss = sum((value - overall) ** 2 for value in means)
            pure = {
                "testable": True,
                "independent_batches": len(means),
                "degrees_of_freedom": len(means) - 1,
                "sum_squares": numeric.clean(ss),
                "mean_square": numeric.clean(ss / (len(means) - 1)),
            }
        values = [observation["response"] for observation in flagged]
        return {
            "repeats_identified": True,
            "center_observations": len(center_observations),
            "flagged_center_observations": len(flagged),
            "flagged_batches": batches,
            "flagged_response_mean": numeric.clean(sum(values) / len(values))
            if values
            else None,
            "pure_error": pure
            or {
                "testable": False,
                "reason": "flagged center repeats do not span distinct independent batches",
                "distinct_flagged_batches": len(batches),
            },
            "message": (
                "center repeats span independent batches and estimate pure error"
                if pure
                else "center repeats within one batch are not independent repeats and do not estimate pure error"
            ),
        }

    # ------------------------------------------------------------------
    # Model comparison with batch-isolated cross-validation
    # ------------------------------------------------------------------
    def _select(self, request, data, models):
        batches = sorted(set(data["batches"]))
        split_count = min(request["folds"], len(batches))
        candidates = []
        for model_name, model in models.items():
            if model.get("status") != "fitted":
                candidates.append(
                    {
                        "model": model_name,
                        "status": "unavailable",
                        "reason": model.get("reason", "model did not fit"),
                    }
                )
                continue
            if split_count < 2:
                candidates.append(
                    {
                        "model": model_name,
                        "status": "unavailable",
                        "reason": "batch-isolated cross-validation needs at least two independent batches",
                    }
                )
                continue
            candidates.append(
                self._batch_cv(request, data, model_name, split_count)
            )
        valid = [item for item in candidates if item["status"] == "valid"]
        if not valid:
            return {
                "status": "unavailable",
                "candidates": candidates,
                "reason": "no model could be validated on held-out batches",
            }
        ranked = sorted(
            valid, key=lambda item: (item["rmse"], item["model"] == "quadratic")
        )
        best = ranked[0]
        return {
            "status": "valid",
            "model": best["model"],
            "criterion": "lowest batch-isolated cross-validated RMSE",
            "candidates": candidates,
            "folds": split_count,
            "isolation": (
                "each batch is assigned to exactly one fold, so specimens from the "
                "same independent batch never enter both training and validation"
            ),
            "selection_uses_final_validation": False,
        }

    def _batch_cv(self, request, data, model_name, split_count):
        batches = data["batches"]
        names = sorted(set(batches))
        assignment = {
            batch: index % split_count
            for index, batch in enumerate(
                sorted(
                    names,
                    key=lambda batch: hashlib.sha256(
                        ("%s/%s" % (request["seed"], batch)).encode()
                    ).hexdigest(),
                )
            )
        }
        predictions = [None] * len(batches)
        fold_rows = []
        for fold in range(split_count):
            training = [
                index
                for index, batch in enumerate(batches)
                if assignment[batch] != fold
            ]
            validation = [
                index
                for index, batch in enumerate(batches)
                if assignment[batch] == fold
            ]
            if not training or not validation:
                return {
                    "model": model_name,
                    "status": "unavailable",
                    "reason": "a fold would have no training or validation rows",
                }
            fit = design.fit(
                [data["encoded"][index] for index in training],
                [data["response"][index] for index in training],
                model_name,
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    "validation_batches": sorted({batches[index] for index in validation}),
                    "training_batches": sorted({batches[index] for index in training}),
                    "status": fit["status"],
                }
            )
            if fit["status"] != "fitted":
                return {
                    "model": model_name,
                    "status": "unavailable",
                    "reason": "a training fold could not be fitted without a pseudo-inverse",
                    "folds": fold_rows,
                    "rmse": None,
                }
            for index in validation:
                predictions[index] = design.predict(
                    fit["coefficient_map"], data["encoded"][index], model_name
                )
        errors = [
            data["response"][index] - predictions[index]
            for index in range(len(batches))
            if predictions[index] is not None
        ]
        if not errors:
            return {
                "model": model_name,
                "status": "unavailable",
                "reason": "cross-validation produced no held-out predictions",
                "folds": fold_rows,
                "rmse": None,
            }
        rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
        return {
            "model": model_name,
            "status": "valid",
            "rmse": numeric.clean(rmse, 10),
            "mean_absolute_error": numeric.clean(
                sum(abs(value) for value in errors) / len(errors), 10
            ),
            "folds": fold_rows,
        }

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def _constraints(self, request):
        dimension = len(request["factors"])
        constraints = []
        for index, factor in enumerate(request["factors"]):
            limits = request["factor_box"][factor["field"]]
            center = float(decimal(factor["center"]))
            half = (float(decimal(factor["high"])) - float(decimal(factor["low"]))) / 2.0
            low = (float(decimal(limits["low"])) - center) / half
            high = (float(decimal(limits["high"])) - center) / half
            row = [0.0] * dimension
            row[index] = 1.0
            constraints.append(
                {
                    "id": "box:%s<=high" % factor["field"],
                    "row": list(row),
                    "bound": high,
                    "kind": "box",
                }
            )
            row = [0.0] * dimension
            row[index] = -1.0
            constraints.append(
                {
                    "id": "box:%s>=low" % factor["field"],
                    "row": row,
                    "bound": -low,
                    "kind": "box",
                }
            )
        for constraint in request["process_constraints"]:
            row = [float(value) for value in constraint["natural_row"]]
            bound = float(decimal(constraint["bound"]))
            for index, factor in enumerate(request["factors"]):
                center = float(decimal(factor["center"]))
                half = (
                    float(decimal(factor["high"])) - float(decimal(factor["low"]))
                ) / 2.0
                bound -= row[index] * center
                row[index] *= half
            constraints.append(
                {
                    "id": "process:" + constraint["id"],
                    "row": row,
                    "bound": bound,
                    "kind": "linear",
                }
            )
        return constraints

    def _natural(self, request, coded):
        values = []
        for index, factor in enumerate(request["factors"]):
            center = float(decimal(factor["center"]))
            half = (float(decimal(factor["high"])) - float(decimal(factor["low"]))) / 2.0
            values.append(center + coded[index] * half)
        return values

    def _box_limits_coded(self, request):
        limits = []
        for factor in request["factors"]:
            center = float(decimal(factor["center"]))
            half = (float(decimal(factor["high"])) - float(decimal(factor["low"]))) / 2.0
            registered = request["factor_box"][factor["field"]]
            limits.append(
                {
                    "coded_low": (float(decimal(registered["low"])) - center) / half,
                    "coded_high": (float(decimal(registered["high"])) - center) / half,
                }
            )
        return limits

    def _boundary_pattern(self, coded, limits):
        pattern = []
        for index, value in enumerate(coded):
            low = limits[index]["coded_low"]
            high = limits[index]["coded_high"]
            tolerance = 1e-3 * max(1.0, abs(high - low))
            if value <= low + tolerance:
                pattern.append("low")
            elif value >= high - tolerance:
                pattern.append("high")
            else:
                pattern.append("interior")
        return tuple(pattern)

    def _variance_function(self, model_name, covariance):
        if covariance is None:
            return None

        def variance(z):
            row = design.model_row(z, model_name)
            return sum(
                row[i] * row[j] * covariance[i][j]
                for i in range(len(row))
                for j in range(len(row))
            )

        return variance

    def _optimise(self, request, data, model):
        intercept, linear, quadratic = design.polynomial_coefficients(
            model["coefficient_map"]
        )
        constraints = self._constraints(request)
        outcome = optimizer.maximise(
            intercept,
            linear,
            quadratic,
            constraints,
            self._variance_function(model["model"], model.get("covariance")),
        )
        if outcome["status"] != "optimal" or outcome["best"] is None:
            return None
        best = outcome["best"]
        natural = self._natural(request, best["coded"])
        within_registered = True
        for index, factor in enumerate(request["factors"]):
            limits = request["factor_box"][factor["field"]]
            if not (
                float(decimal(limits["low"])) - 1e-9
                <= natural[index]
                <= float(decimal(limits["high"])) + 1e-9
            ):
                within_registered = False
        if not within_registered:
            reject("study", "optimiser produced a point outside the registered experiment range")
        hull = numeric.convex_combination(
            [observation["values"] for observation in data["observations"]],
            natural,
        )
        return {
            "natural": {
                factor["field"]: numeric.clean(natural[index], 10)
                for index, factor in enumerate(request["factors"])
            },
            "coded": best["coded"],
            "predicted_mean": best["predicted_mean"],
            "standard_error": best["standard_error"],
            "source": best["source"],
            "active_constraints": best["active_constraints"],
            "optimality_evidence": best["optimality"],
            "model": {
                "form": outcome["method"],
                "global_proof": outcome["global_proof"],
                "global_evidence": outcome["global_evidence"],
                "heuristic": outcome["heuristic"],
                "objective_concave": outcome["objective_concave"],
                "planar_objective": outcome["planar_objective"],
                "constant_objective": outcome["constant_objective"],
                "candidates_examined": outcome["candidates_examined"],
                "interior_stationary": outcome["interior_stationary"],
                "boundary_candidates": outcome["boundary_candidates"],
                "top_candidates": outcome["candidates"],
            },
            "within_registered_factor_box": within_registered,
            "within_design_convex_hull": hull["inside"],
            "convex_hull_test": hull,
            "extrapolation_warning": not hull["inside"],
            "extrapolation_note": (
                "point is inside the registered experiment box but outside the "
                "convex hull of actual design points; treat it as extrapolation"
                if not hull["inside"]
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Stability under batch resampling
    # ------------------------------------------------------------------
    def _stability(self, request, data, model, recommendation):
        if recommendation is None:
            return {"computed": False, "reason": "no recommendation to resample"}
        repetitions = request["bootstrap"]
        if repetitions == 0:
            return {"computed": False, "reason": "bootstrap repetitions set to zero"}
        batches = data["batches"]
        independent = sorted(set(batches))
        rows_by_batch = {
            batch: [index for index, value in enumerate(batches) if value == batch]
            for batch in independent
        }
        rng = random.Random("%s/bootstrap" % (request["seed"],))
        reference = [
            recommendation["natural"][factor["field"]] for factor in request["factors"]
        ]

        box_limits = self._box_limits_coded(request)
        reference_coded = [
            (reference[index] - float(decimal(factor["center"])))
            / ((float(decimal(factor["high"])) - float(decimal(factor["low"]))) / 2.0)
            for index, factor in enumerate(request["factors"])
        ]
        reference_pattern = self._boundary_pattern(reference_coded, box_limits)
        optimum_values = []
        positions = []
        failures = {"fit_failures": 0, "optimiser_failures": 0}
        constraint_sets = {}
        boundary_patterns = {}
        for _ in range(repetitions):
            draw = [rng.choice(independent) for _ in independent]
            indices = [index for batch in draw for index in rows_by_batch[batch]]
            fit = design.fit(
                [data["encoded"][index] for index in indices],
                [data["response"][index] for index in indices],
                model["model"],
            )
            if fit["status"] != "fitted":
                failures["fit_failures"] += 1
                continue
            intercept, linear, quadratic = design.polynomial_coefficients(
                fit["coefficient_map"]
            )
            outcome = optimizer.maximise(
                intercept,
                linear,
                quadratic,
                self._constraints(request),
                self._variance_function(model["model"], fit.get("covariance")),
            )
            if outcome["status"] != "optimal" or outcome["best"] is None:
                failures["optimiser_failures"] += 1
                continue
            best = outcome["best"]
            optimum_values.append(best["predicted_mean"])
            positions.append(self._natural(request, best["coded"]))
            key = tuple(sorted(best["active_constraints"]))
            constraint_sets[key] = constraint_sets.get(key, 0) + 1
            pattern = self._boundary_pattern(best["coded"], box_limits)
            boundary_patterns[pattern] = boundary_patterns.get(pattern, 0) + 1
        if not optimum_values:
            return {
                "computed": False,
                "reason": "every resample failed to fit or optimise",
                "repetitions": repetitions,
                "failures": failures,
            }
        displacements = [
            math.sqrt(
                sum((position[index] - reference[index]) ** 2 for index in range(len(reference)))
            )
            for position in positions
        ]
        reference_set = tuple(sorted(recommendation["active_constraints"]))
        differing = [key for key in constraint_sets if key != reference_set]
        differing_patterns = [
            key for key in boundary_patterns if key != reference_pattern
        ]
        return {
            "computed": True,
            "repetitions": repetitions,
            "successful": len(optimum_values),
            "failures": failures,
            "fit_failures_observed": failures["fit_failures"] > 0,
            "optimum_predicted_mean": numeric.summary(optimum_values),
            "recommended_position": reference,
            "position_displacement_euclidean": numeric.summary(displacements),
            "position_per_factor": {
                factor["field"]: numeric.summary(
                    [position[index] for position in positions]
                )
                for index, factor in enumerate(request["factors"])
            },
            "active_constraint_sets": [
                {"constraints": list(key), "count": value}
                for key, value in sorted(
                    constraint_sets.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "optimum_jumped_to_different_constraints": bool(differing),
            "optimum_jumped_to_different_boundary": bool(differing_patterns),
            "optimum_jumped": bool(differing or differing_patterns),
            "reference_boundary_pattern": list(reference_pattern),
            "boundary_patterns": [
                {"pattern": list(key), "count": value}
                for key, value in sorted(
                    boundary_patterns.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "method": (
                "batches are resampled with replacement as whole units; "
                "every specimen of a sampled batch is included together"
            ),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def list(self):
        rows = self.db.rows("SELECT id,state,digest,request FROM studies ORDER BY id")
        result = []
        for row in rows:
            request = parse_json(row["request"])
            status = self.status(row["id"])
            result.append(
                {
                    "id": row["id"],
                    "state": row["state"],
                    "digest": row["digest"],
                    "release": request["release"],
                    "release_state": status["release_state"],
                    "deliverable": status["deliverable"],
                    "dataset": request["dataset"],
                    "version": request["version"],
                }
            )
        return result

    def status(self, study_id):
        row = self.db.one("SELECT * FROM studies WHERE id=?", (study_id,))
        if row is None:
            reject("missing", "study not found")
        request = parse_json(row["request"])
        try:
            release = self.releases.get(request["release"])
            eligibility = self.releases.eligibility(request["release"])
        except VaultError:
            release = None
            eligibility = {"eligible": False, "problems": [{"code": "release_missing"}]}
        deliverable = bool(
            release
            and release["state"] == "approved"
            and eligibility["eligible"]
        )
        return {
            "id": study_id,
            "state": row["state"],
            "digest": row["digest"],
            "release": request["release"],
            "release_state": release["state"] if release else None,
            "release_eligible": eligibility["eligible"],
            "problems": eligibility["problems"],
            "deliverable": deliverable,
            "message": (
                "source release remains approved and eligible"
                if deliverable
                else "study stays traceable but must not be treated as a valid deliverable"
            ),
        }

    def export(self, study_id, export_id=None):
        result = self.result(study_id)
        status = self.status(study_id)
        if not status["deliverable"]:
            reject(
                "state",
                "study cannot be exported while its source release is revoked or ineligible",
                study=study_id,
            )
        envelope = {
            "kind": "response-surface-study",
            "schema": 1,
            "study": result,
            "status": status,
        }
        envelope["digest"] = _envelope_digest(envelope)
        if export_id is None:
            return envelope
        name(export_id)
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT * FROM study_exports WHERE id=?", (export_id,)
            )
            if old:
                if old["study"] != study_id:
                    reject("conflict", "study export identity differs")
                payload = parse_json(old["payload"])
                if _envelope_digest(payload) != old["digest"]:
                    reject("integrity", "stored study export is corrupted")
                return {**payload, "reused": True}
            con.execute(
                "INSERT INTO study_exports VALUES(?,?,?,?, 'complete')",
                (export_id, study_id, _canonical(envelope), envelope["digest"]),
            )
            self.db.audit(
                "study_exported",
                export_id,
                {"study": study_id, "digest": envelope["digest"]},
            )
        return {**envelope, "reused": False}

    def verify_export(self, export_id):
        row = self.db.one("SELECT * FROM study_exports WHERE id=?", (export_id,))
        if row is None:
            reject("missing", "study export not found")
        payload = parse_json(row["payload"])
        if _envelope_digest(payload) != row["digest"]:
            reject("integrity", "stored study export is corrupted")
        status = self.status(row["study"])
        return {
            "id": export_id,
            "study": row["study"],
            "digest": row["digest"],
            "valid_delivery": status["deliverable"],
            "release_state": status["release_state"],
            "message": status["message"],
        }


_PAYLOAD_FIELDS = frozenset({"id", "source", "design", "configuration", "seed", "analysis"})


def _envelope_digest(envelope):
    return digest(
        {key: value for key, value in envelope.items() if key != "digest"}
    )


def _canonical(value):
    from ..validation import canonical

    return canonical(value)
