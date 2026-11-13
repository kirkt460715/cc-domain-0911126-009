"""Constrained maximisation of a fitted quadratic response surface.

The search is deterministic: for a two- or three-factor problem the maximum of
a quadratic over the polytope defined by the factor box and linear process
constraints is attained either at the unconstrained stationary point or on a
face of the polytope.  Every candidate face is enumerated, so the result is a
global maximum up to floating-point tolerance rather than a heuristic search.
When the model cannot be fitted the study reports that limitation instead of
building a recommendation.
"""

import itertools
import math

from . import numeric


def _independent(rows, tolerance=1e-9):
    """Whether the constraint normals are linearly independent.

    Row rank is tested through the ``count x count`` Gram matrix ``A A^T``
    after scaling each normal to unit length.
    """
    count = len(rows)
    if count == 0:
        return True
    scaled = []
    for row in rows:
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            return False
        scaled.append([value / norm for value in row])
    gram = [
        [sum(scaled[i][k] * scaled[j][k] for k in range(len(scaled[i]))) for j in range(count)]
        for i in range(count)
    ]
    eigenvalues, _ = numeric.jacobi_eigen(gram)
    return eigenvalues[-1] > tolerance * max(eigenvalues[0], 1.0)


def _solve_square(matrix, target):
    return numeric.solve_square(matrix, target)


def _kkt(Q, b, rows, bounds):
    """Solve ``2 Q z + b = A^T mu`` with ``A z = c`` for a candidate face.

    Returns ``(z, mu)`` or ``None`` when the KKT system is singular; then the
    maximum of that face lies on a smaller face that is enumerated separately.
    """
    dimension = len(b)
    count = len(rows)
    size = dimension + count
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(dimension):
        for j in range(dimension):
            matrix[i][j] = 2.0 * Q[i][j]
        for k in range(count):
            matrix[i][dimension + k] = -rows[k][i]
    for k in range(count):
        for j in range(dimension):
            matrix[dimension + k][j] = rows[k][j]
    target = [-b[i] for i in range(dimension)] + list(bounds)
    solution = _solve_square(matrix, target)
    if solution is None:
        return None
    return solution[:dimension], solution[dimension:]


def value(Q, b, intercept, z):
    return intercept + sum(b[i] * z[i] for i in range(len(z))) + sum(
        z[i] * Q[i][j] * z[j] for i in range(len(z)) for j in range(len(z))
    )


def _tangent_projector_null(Q, active_rows):
    """Eigenvalues of the Hessian restricted to the active-face tangent space."""
    if not active_rows:
        return list(numeric.jacobi_eigen(Q)[0])
    dimension = len(Q)
    gram = numeric.gram(active_rows)
    eigenvalues, vectors = numeric.jacobi_eigen(gram)
    threshold = max(eigenvalues[0], 1.0) * 1e-9
    basis = [
        vectors[index]
        for index, current in enumerate(eigenvalues)
        if current <= threshold
    ]
    if not basis:
        return []
    projected = [
        [
            sum(
                basis[a][i] * Q[i][j] * basis[b][j]
                for i in range(dimension)
                for j in range(dimension)
            )
            for b in range(len(basis))
        ]
        for a in range(len(basis))
    ]
    projected = [[(projected[i][j] + projected[j][i]) / 2.0 for j in range(len(basis))] for i in range(len(basis))]
    return list(numeric.jacobi_eigen(projected)[0])


def maximise(intercept, linear, quadratic, constraints, prediction_variance=None):
    """Maximise the fitted surface in coded coordinates.

    ``constraints`` is a list of ``{"id", "row", "bound", "kind"}`` with
    ``row . z <= bound``.  ``prediction_variance(z)`` may supply the variance
    of the predicted mean for a candidate.
    """
    dimension = len(linear)
    Q = [list(row) for row in quadratic]
    b = list(linear)
    names = [item["id"] for item in constraints]
    candidates = []
    seen = set()

    def feasible(z, tolerance=1e-7):
        return all(
            sum(item["row"][i] * z[i] for i in range(dimension))
            <= item["bound"] + tolerance
            for item in constraints
        )

    def evidence(z, active_indices):
        if not active_indices:
            return {
                "gradient": [numeric.clean(current, 10) for current in (
                    b[i] + 2.0 * sum(Q[i][j] * z[j] for j in range(dimension))
                    for i in range(dimension)
                )],
                "stationary": True,
                "multipliers_nonnegative": True,
                "multipliers": {},
                "tangent_hessian_eigenvalues": [
                    numeric.clean(current, 10)
                    for current in _tangent_projector_null(Q, [])
                ],
                "second_order_sufficient": bool(
                    all(
                        current <= 1e-7
                        for current in _tangent_projector_null(Q, [])
                    )
                ),
                "all_constraints_inactive": True,
            }
        rows = [constraints[index]["row"] for index in active_indices]
        bounds = [constraints[index]["bound"] for index in active_indices]
        solved = _kkt(Q, b, rows, bounds)
        multipliers = (
            {names[index]: numeric.clean(current, 10) for index, current in zip(active_indices, solved[1])}
            if solved is not None
            else None
        )
        eigenvalues = _tangent_projector_null(Q, rows)
        inactive_ok = all(
            sum(item["row"][i] * z[i] for i in range(dimension))
            <= item["bound"] + 1e-7
            for index, item in enumerate(constraints)
            if index not in active_indices
        )
        return {
            "stationary_in_tangent_space": solved is not None,
            "multipliers_nonnegative": bool(
                multipliers is not None
                and all(current >= -1e-7 for current in multipliers.values())
            ),
            "multipliers": multipliers or {},
            "tangent_hessian_eigenvalues": [
                numeric.clean(current, 10) for current in eigenvalues
            ],
            "second_order_sufficient": bool(
                all(current <= 1e-7 for current in eigenvalues)
            ),
            "all_constraints_inactive": bool(inactive_ok),
        }

    def add(z, source, active_indices):
        key = tuple(round(float(current), 9) for current in z)
        if key in seen:
            return
        seen.add(key)
        variance = prediction_variance(z) if prediction_variance is not None else None
        candidates.append(
            {
                "source": source,
                "coded": [numeric.clean(current, 10) for current in z],
                "predicted_mean": numeric.clean(value(Q, b, intercept, z), 10),
                "standard_error": numeric.clean(math.sqrt(max(variance, 0.0)), 10)
                if variance is not None
                else None,
                "active_constraints": sorted(names[index] for index in active_indices),
                "optimality": evidence(z, list(active_indices)),
            }
        )

    stationary = None
    hessian = [[2.0 * Q[i][j] for j in range(dimension)] for i in range(dimension)]
    solution = _solve_square(hessian, [-b[i] for i in range(dimension)])
    if solution is not None:
        stationary = solution
        if feasible(solution):
            add(solution, "unconstrained_stationary", [])

    for count in range(1, min(dimension, len(constraints)) + 1):
        for selection in itertools.combinations(range(len(constraints)), count):
            rows = [constraints[index]["row"] for index in selection]
            bounds = [constraints[index]["bound"] for index in selection]
            if not _independent(rows):
                continue
            if count == dimension:
                vertex = _solve_square(rows, bounds)
                if vertex is not None and feasible(vertex):
                    add(vertex, "vertex", list(selection))
            solved = _kkt(Q, b, rows, bounds)
            if solved is None:
                continue
            z = solved[0]
            if feasible(z):
                add(z, "face_stationary", list(selection))

    hessian_eigenvalues = numeric.jacobi_eigen(Q)[0]
    if not candidates:
        return {
            "status": "no_feasible_candidate",
            "method": "exhaustive active-set enumeration",
            "global_proof": False,
            "candidates_examined": 0,
            "interior_stationary": None,
            "best": None,
            "candidates": [],
            "boundary_candidates": [],
        }
    ranked = sorted(
        candidates,
        key=lambda item: (-item["predicted_mean"], item["coded"], item["source"]),
    )
    interior = None
    if stationary is not None:
        nature = (
            "maximum"
            if hessian_eigenvalues[0] < -1e-9
            else "minimum"
            if hessian_eigenvalues[-1] > 1e-9
            else "saddle_or_ridge"
        )
        interior = {
            "coded": [numeric.clean(current, 10) for current in stationary],
            "predicted_mean": numeric.clean(value(Q, b, intercept, stationary), 10),
            "within_design_bounds": bool(feasible(stationary)),
            "nature": nature,
            "hessian_eigenvalues": [
                numeric.clean(current, 10) for current in hessian_eigenvalues
            ],
        }
    planar_objective = all(
        abs(current) <= 1e-9 for row in Q for current in row
    )
    constant_objective = planar_objective and all(
        abs(current) <= 1e-9 for current in b
    )
    return {
        "status": "optimal",
        "method": "exhaustive active-set enumeration over factor box and linear process faces",
        "global_proof": True,
        "global_evidence": "the maximum of a quadratic over a polytope lies at its unconstrained stationary point or on an enumerated face",
        "heuristic": False,
        "objective_concave": bool(hessian_eigenvalues[0] <= 1e-9),
        "objective_hessian_eigenvalues": [
            numeric.clean(current, 10) for current in hessian_eigenvalues
        ],
        "planar_objective": bool(planar_objective),
        "constant_objective": bool(constant_objective),
        "candidates_examined": len(candidates),
        "candidate_count": len(ranked),
        "interior_stationary": interior,
        "best": ranked[0],
        "candidates": ranked[:12],
        "boundary_candidates": [
            item for item in ranked[:12] if item["active_constraints"]
        ],
    }
