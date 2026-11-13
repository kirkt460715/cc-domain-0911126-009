"""Design construction, least-squares fitting and diagnostic statistics."""

import math

from . import numeric


class DesignError(Exception):
    """Internal signal used when a model cannot be fitted."""


def term_names(factor_count, model):
    names = ["intercept"]
    names.extend("x%d" % index for index in range(factor_count))
    if model == "quadratic":
        for left in range(factor_count):
            for right in range(left + 1, factor_count):
                names.append("x%d*x%d" % (left, right))
        for index in range(factor_count):
            names.append("x%d^2" % index)
    if model not in {"linear", "quadratic"}:
        raise DesignError("unknown model form")
    return names


def model_row(values, model):
    row = [1.0]
    row.extend(values)
    if model == "quadratic":
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                row.append(values[left] * values[right])
        row.extend(value * value for value in values)
    return row


def model_matrix(encoded, model):
    return [model_row(row, model) for row in encoded]


def polynomial_coefficients(coefficients):
    """Translate fitted term estimates into intercept/linear/quadratic form."""
    intercept = coefficients["intercept"]
    linear = []
    squares = []
    interactions = {}
    for term, value in coefficients.items():
        if term == "intercept":
            continue
        if term.startswith("x") and "^2" in term:
            squares.append((int(term[1 : term.index("^2")]), value))
        elif "*" in term:
            left, right = term[1:].split("*x")
            interactions[(int(left), int(right))] = value
        elif term.startswith("x"):
            linear.append((int(term[1:]), value))
    factor_count = len(linear)
    linear.sort()
    squares.sort()
    vector = [0.0] * factor_count
    for index, value in linear:
        vector[index] = value
    q = [[0.0] * factor_count for _ in range(factor_count)]
    for index, value in squares:
        q[index][index] = value
    for (left, right), value in interactions.items():
        q[left][right] = value / 2.0
        q[right][left] = value / 2.0
    return intercept, vector, q


def fit(encoded, response, model, residuals=True):
    """Fit a linear or full quadratic model, refusing rank-deficient designs."""
    names = term_names(len(encoded[0]) if encoded else 0, model)
    matrix = model_matrix(encoded, model)
    rows = len(matrix)
    columns = len(names)
    rank = numeric.design_rank(matrix)
    result = {
        "model": model,
        "terms": names,
        "rows": rows,
        "parameters": columns,
        "residual_df": rows - columns,
        "design_rank": rank,
    }
    if rows < columns + 1:
        result.update(
            status="insufficient",
            reason="fewer observations than parameters plus one residual degree of freedom",
        )
        return result
    if rank["deficient"]:
        result.update(
            status="rank_deficient",
            reason="design matrix does not have full column rank",
        )
        return result
    coefficients = numeric.qr_solve(matrix, response)
    if coefficients is None:
        result.update(
            status="numerical_failure",
            reason="least-squares solve did not produce a finite solution",
        )
        return result
    fitted = [
        sum(matrix[row][column] * coefficients[column] for column in range(columns))
        for row in range(rows)
    ]
    errors = [response[index] - fitted[index] for index in range(rows)]
    sse = sum(value * value for value in errors)
    mean = sum(response) / rows
    sst = sum((value - mean) ** 2 for value in response)
    degrees = rows - columns
    mse = sse / degrees
    # ``(X^T X)`` may be moderately ill-conditioned even when the design has
    # full rank; if its inverse cannot be certified we report missing standard
    # errors rather than substituting a pseudo-inverse.
    gram = numeric.gram(matrix)
    inverse_columns = [
        numeric.solve_square(gram, [1.0 if i == column else 0.0 for i in range(columns)])
        for column in range(columns)
    ]
    inverse_matrix = None
    if all(item is not None for item in inverse_columns):
        inverse_matrix = [
            [inverse_columns[column][row] for column in range(columns)]
            for row in range(columns)
        ]
    coefficient_rows = []
    t_quantile = _t_multiplier(0.05, degrees) if degrees > 0 else None
    for index, name in enumerate(names):
        estimate = coefficients[index]
        variance = (
            mse * inverse_matrix[index][index]
            if inverse_matrix is not None and inverse_matrix[index][index] > 0
            else None
        )
        standard_error = math.sqrt(variance) if variance is not None else None
        statistic = None
        probability = None
        interval = None
        if standard_error is not None and standard_error > 1e-15:
            statistic = estimate / standard_error
            probability = numeric.t_two_sided(statistic, degrees)
        if standard_error is not None and t_quantile is not None:
            half_width = t_quantile * standard_error
            interval = [
                numeric.clean(estimate - half_width),
                numeric.clean(estimate + half_width),
            ]
        else:
            interval = None
        coefficient_rows.append(
            {
                "term": name,
                "estimate": numeric.clean(estimate),
                "standard_error": numeric.clean(standard_error)
                if standard_error is not None
                else None,
                "t_statistic": numeric.clean(statistic) if statistic is not None else None,
                "p_value": numeric.clean(probability, 15)
                if probability is not None
                else None,
                "confidence_interval_95": interval,
            }
        )
    covariance = None
    if inverse_matrix is not None and mse >= 0:
        covariance = [
            [numeric.clean(mse * inverse_matrix[i][j], 12) for j in range(columns)]
            for i in range(columns)
        ]
    residual_rows = []
    if residuals:
        for index in range(rows):
            leverage = None
            if inverse_matrix is not None:
                leverage = sum(
                    matrix[index][i] * inverse_matrix[i][j] * matrix[index][j]
                    for i in range(columns)
                    for j in range(columns)
                )
            standardized = None
            if mse > 0 and leverage is not None and 1.0 - leverage > 1e-12:
                standardized = errors[index] / math.sqrt(mse * (1.0 - leverage))
            residual_rows.append(
                {
                    "row": index,
                    "observed": numeric.clean(response[index]),
                    "fitted": numeric.clean(fitted[index]),
                    "residual": numeric.clean(errors[index]),
                    "leverage": numeric.clean(leverage)
                    if leverage is not None
                    else None,
                    "standardized_residual": numeric.clean(standardized)
                    if standardized is not None
                    else None,
                }
            )
    durbin_watson = None
    if len(errors) > 1 and sse > 0:
        numerator = sum((errors[i] - errors[i - 1]) ** 2 for i in range(1, len(errors)))
        durbin_watson = numerator / sse
    result.update(
        status="fitted",
        coefficients=coefficient_rows,
        coefficient_map={row["term"]: row["estimate"] for row in coefficient_rows},
        residual_rows=residual_rows,
        anova={
            "sum_squares_residual": numeric.clean(sse),
            "sum_squares_total": numeric.clean(sst),
            "degrees_of_freedom": degrees,
            "mean_square_residual": numeric.clean(mse),
            "root_mean_square_error": numeric.clean(math.sqrt(mse)),
            "r_squared": numeric.clean(1.0 - sse / sst) if sst > 0 else None,
            "adjusted_r_squared": numeric.clean(
                1.0 - (sse / degrees) / (sst / (rows - 1))
            )
            if sst > 0 and rows > 1
            else None,
            "durbin_watson": numeric.clean(durbin_watson)
            if durbin_watson is not None
            else None,
        },
        residual_summary={
            "minimum": numeric.clean(min(errors)) if errors else None,
            "maximum": numeric.clean(max(errors)) if errors else None,
            "mean": numeric.clean(sum(errors) / len(errors)) if errors else None,
            "largest_absolute": numeric.clean(max((abs(value) for value in errors), default=0.0)),
        },
        covariance=covariance,
    )
    return result


def predict(coefficient_map, values, model):
    row = model_row(values, model)
    names = term_names(len(values), model)
    return sum(row[index] * coefficient_map[name] for index, name in enumerate(names))


def replicate_groups(conditions, batches, response, labels):
    """Compute pure-error contributions from independent batch replicates.

    ``conditions`` is a list of condition keys, ``batches`` the batch identity
    of every observation and ``labels`` the human-readable factor values.  A
    condition contributes pure error only when at least two distinct batches
    occur there.  Multiple rows from one batch are collapsed to the batch mean
    so that they cannot masquerade as independent repeats.
    """
    grouped = {}
    for index, key in enumerate(conditions):
        grouped.setdefault(key, {"label": labels[index], "rows": []})["rows"].append(index)
    groups = []
    total_ss = 0.0
    total_df = 0
    for key in sorted(grouped):
        entry = grouped[key]
        by_batch = {}
        for index in entry["rows"]:
            by_batch.setdefault(batches[index], []).append(response[index])
        batch_means = {
            batch: sum(values) / len(values) for batch, values in by_batch.items()
        }
        independent = len(by_batch)
        df = max(0, independent - 1)
        ss = 0.0
        if independent >= 2:
            overall = sum(batch_means.values()) / independent
            ss = sum((value - overall) ** 2 for value in batch_means.values())
        contributes = independent >= 2
        if contributes:
            total_ss += ss
            total_df += df
        groups.append(
            {
                "condition": entry["label"],
                "observations": len(entry["rows"]),
                "batches": sorted(by_batch),
                "independent_batches": independent,
                "same_batch_repeats": len(entry["rows"]) - independent,
                "degrees_of_freedom": df if contributes else 0,
                "sum_squares": numeric.clean(ss) if contributes else None,
                "contributes_pure_error": contributes,
            }
        )
    return groups, total_ss, total_df


def lack_of_fit(residual_sse, residual_df, pure_ss, pure_df):
    if pure_df <= 0:
        return {
            "testable": False,
            "reason": "no_independent_replicates",
            "message": "only independent repeats at the same condition contribute pure error; without them lack of fit cannot be judged",
        }
    degrees = residual_df - pure_df
    if degrees <= 0:
        return {
            "testable": False,
            "reason": "no_lack_of_fit_degrees_of_freedom",
            "degrees_of_freedom": degrees,
        }
    lack_ss = max(0.0, residual_sse - pure_ss)
    lack_ms = lack_ss / degrees
    pure_ms = pure_ss / pure_df
    if pure_ms <= 0:
        if lack_ms <= 0:
            statistic = 0.0
            probability = 1.0
        else:
            statistic = None
            probability = 0.0
    else:
        statistic = lack_ms / pure_ms
        probability = numeric.f_survival(statistic, degrees, pure_df)
    return {
        "testable": True,
        "source": "independent_batches_at_identical_conditions",
        "sum_squares_lack_of_fit": numeric.clean(lack_ss),
        "degrees_of_freedom_lack_of_fit": degrees,
        "mean_square_lack_of_fit": numeric.clean(lack_ms),
        "sum_squares_pure_error": numeric.clean(pure_ss),
        "degrees_of_freedom_pure_error": pure_df,
        "mean_square_pure_error": numeric.clean(pure_ms),
        "f_statistic": numeric.clean(statistic) if statistic is not None else None,
        "p_value": numeric.clean(probability, 15)
        if probability is not None
        else None,
    }


def _t_multiplier(probability, degrees):
    """Two-sided Student-t multiplier for a target upper-tail probability.

    The tail probability decreases monotonically as the statistic grows, so a
    short bisection yields the critical value used for coefficient intervals.
    """
    if degrees <= 0:
        return None
    low, high = 0.0, 1000.0
    for _ in range(200):
        middle = (low + high) / 2.0
        tail = numeric.t_two_sided(middle, degrees)
        if tail is None:
            return None
        if tail < probability:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0
