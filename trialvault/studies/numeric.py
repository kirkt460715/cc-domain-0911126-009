"""Small, dependency-free numerical helpers for response-surface studies.

The project intentionally depends only on the Python standard library.  These
routines are deliberately limited to the sizes needed by a two- or three-factor
response-surface design.  They favour explicit failure over silently returning
a pseudo-inverse solution when a design is rank deficient.
"""

import math

TOLERANCE = 1e-10


def clean(value, digits=12):
    """Return a finite float suitable for canonical JSON output."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric result")
    rounded = round(value, digits)
    if rounded == 0:
        return 0.0
    return rounded


def jacobi_eigen(matrix, tolerance=1e-13, maximum_sweeps=100):
    """Eigen-decompose a real symmetric matrix with cyclic Jacobi rotations.

    Returns ``(values, vectors)`` where ``vectors`` is a list of rows such that
    ``vectors[j]`` is the eigenvector belonging to ``values[j]``.
    """
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("eigen decomposition requires a square matrix")
    work = [[float(value) for value in row] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    scale = max((abs(work[i][j]) for i in range(size) for j in range(size)), default=0.0)
    if scale == 0:
        return [0.0] * size, vectors
    for _ in range(maximum_sweeps):
        off = math.sqrt(
            sum(
                work[i][j] * work[i][j]
                for i in range(size)
                for j in range(i + 1, size)
            )
        )
        if off <= tolerance * scale:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(work[p][q]) <= tolerance * scale:
                    continue
                angle = (work[q][q] - work[p][p]) / (2.0 * work[p][q])
                if angle >= 0:
                    tangent = 1.0 / (angle + math.sqrt(1.0 + angle * angle))
                else:
                    tangent = -1.0 / (-angle + math.sqrt(1.0 + angle * angle))
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                for k in range(size):
                    a_pk = work[p][k]
                    a_qk = work[q][k]
                    work[p][k] = cosine * a_pk - sine * a_qk
                    work[q][k] = sine * a_pk + cosine * a_qk
                for k in range(size):
                    a_kp = work[k][p]
                    a_kq = work[k][q]
                    work[k][p] = cosine * a_kp - sine * a_kq
                    work[k][q] = sine * a_kp + cosine * a_kq
                for k in range(size):
                    v_kp = vectors[k][p]
                    v_kq = vectors[k][q]
                    vectors[k][p] = cosine * v_kp - sine * v_kq
                    vectors[k][q] = sine * v_kp + cosine * v_kq
    values = [work[i][i] for i in range(size)]
    order = sorted(range(size), key=lambda index: values[index], reverse=True)
    ordered_values = [values[index] for index in order]
    ordered_vectors = [
        [vectors[row][column] for row in range(size)] for column in order
    ]
    return ordered_values, ordered_vectors


def gram(matrix):
    columns = len(matrix[0]) if matrix else 0
    result = [[0.0] * columns for _ in range(columns)]
    for row in matrix:
        for i in range(columns):
            if row[i] == 0:
                continue
            for j in range(i, columns):
                product = row[i] * row[j]
                result[i][j] += product
                if i != j:
                    result[j][i] += product
    return result


def design_rank(matrix, rank_tolerance=1e-10):
    """Return rank and conditioning information for a design matrix.

    Columns are scaled to unit Euclidean norm before the Gram matrix is
    analysed so that the condition number measures collinearity rather than
    differences in factor units.
    """
    rows = len(matrix)
    columns = len(matrix[0]) if matrix else 0
    if rows == 0 or columns == 0:
        return {
            "rows": rows,
            "columns": columns,
            "rank": 0,
            "deficient": True,
            "nullity": columns,
            "condition_number": None,
            "singular_values": [],
            "scaled_eigenvalues": [],
            "aliased_terms": [],
        }
    norms = []
    for j in range(columns):
        norm = math.sqrt(sum(matrix[i][j] * matrix[i][j] for i in range(rows)))
        norms.append(norm)
        if norm == 0:
            return {
                "rows": rows,
                "columns": columns,
                "rank": 0,
                "deficient": True,
                "nullity": columns,
                "condition_number": None,
                "singular_values": [],
                "scaled_eigenvalues": [],
                "aliased_terms": [],
            }
    scaled = [
        [matrix[i][j] / norms[j] for j in range(columns)] for i in range(rows)
    ]
    values, vectors = jacobi_eigen(gram(scaled))
    largest = max(values[0], 0.0)
    threshold = max(largest * rank_tolerance, 1e-14)
    positive = [value for value in values if value > threshold]
    rank = len(positive)
    condition = None
    if rank == columns and positive:
        condition = math.sqrt(positive[0] / positive[-1])
    aliased = []
    for index, value in enumerate(values):
        if value <= threshold:
            weights = vectors[index]
            largest_weight = max(abs(weight) for weight in weights)
            if largest_weight == 0:
                continue
            aliased.append([round(weight / largest_weight, 8) for weight in weights])
            break
    return {
        "rows": rows,
        "columns": columns,
        "rank": rank,
        "deficient": rank < columns,
        "nullity": columns - rank,
        "condition_number": condition,
        "singular_values": [math.sqrt(max(value, 0.0)) for value in values],
        "scaled_eigenvalues": values,
        "aliased_terms": aliased,
    }


def solve_square(matrix, vector):
    """Solve a dense square system with partial pivoting.

    Returns ``None`` when the system is singular within tolerance; callers must
    treat that as a diagnosed failure, never substitute a pseudo-inverse.
    """
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or len(vector) != size:
        raise ValueError("square solve requires matching dimensions")
    work = [[float(value) for value in row] + [float(vector[i])] for i, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    tolerance = max(scale * 1e-12, 1e-15)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= tolerance:
            return None
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for row in range(column + 1, size):
            factor = work[row][column] / divisor
            if factor == 0:
                continue
            for item in range(column, size + 1):
                work[row][item] -= factor * work[column][item]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = work[row][size]
        for column in range(row + 1, size):
            total -= work[row][column] * result[column]
        result[row] = total / work[row][row]
    if any(not math.isfinite(value) for value in result):
        return None
    return result


def qr_solve(matrix, vector):
    """Least-squares solve using Householder QR without forming normal equations."""
    rows = len(matrix)
    columns = len(matrix[0]) if matrix else 0
    if rows < columns or len(vector) != rows:
        return None
    work = [[float(value) for value in row] for row in matrix]
    target = [float(value) for value in vector]
    for column in range(columns):
        norm = math.sqrt(sum(work[row][column] ** 2 for row in range(column, rows)))
        if norm <= 1e-14 * max(
            1.0,
            max(
                math.sqrt(sum(work[row][j] ** 2 for row in range(rows)))
                for j in range(columns)
            ),
        ):
            return None
        alpha = -norm if work[column][column] >= 0 else norm
        vector_norm = 0.0
        reflector = [0.0] * rows
        for row in range(column, rows):
            reflector[row] = work[row][column]
        reflector[column] -= alpha
        vector_norm = sum(value * value for value in reflector)
        if vector_norm <= 1e-30:
            continue
        for j in range(column, columns):
            dot = sum(reflector[row] * work[row][j] for row in range(column, rows))
            factor = 2.0 * dot / vector_norm
            for row in range(column, rows):
                work[row][j] -= factor * reflector[row]
        dot = sum(reflector[row] * target[row] for row in range(column, rows))
        factor = 2.0 * dot / vector_norm
        for row in range(column, rows):
            target[row] -= factor * reflector[row]
    coefficients = [0.0] * columns
    for row in range(columns - 1, -1, -1):
        total = target[row]
        for column in range(row + 1, columns):
            total -= work[row][column] * coefficients[column]
        if abs(work[row][row]) <= 1e-14:
            return None
        coefficients[row] = total / work[row][row]
    if any(not math.isfinite(value) for value in coefficients):
        return None
    return coefficients


def _betacf(a, b, x):
    maximum_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_survival(statistic, numerator_df, denominator_df):
    """Upper-tail probability of an F distribution."""
    if statistic is None or not math.isfinite(statistic) or statistic < 0:
        return None
    if numerator_df <= 0 or denominator_df <= 0:
        return None
    x = numerator_df * statistic / (numerator_df * statistic + denominator_df)
    return max(0.0, min(1.0, _betai(denominator_df / 2.0, numerator_df / 2.0, 1.0 - x)))


def t_two_sided(statistic, degrees_of_freedom):
    if statistic is None or degrees_of_freedom <= 0:
        return None
    if not math.isfinite(statistic):
        return 0.0
    value = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return max(0.0, min(1.0, _betai(degrees_of_freedom / 2.0, 0.5, value)))


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return {
        "count": len(values),
        "mean": clean(mean),
        "sd": clean(math.sqrt(variance)),
        "minimum": clean(min(values)),
        "maximum": clean(max(values)),
        "p05": clean(quantile(values, 0.05)),
        "p50": clean(quantile(values, 0.50)),
        "p95": clean(quantile(values, 0.95)),
    }


def convex_combination(points, target, tolerance=1e-8):
    """Test whether ``target`` lies in the convex hull of ``points``.

    Uses a Phase-I simplex on ``sum(w)=1, P^T w=target, w>=0``.  The test is
    exact up to floating-point tolerance and scales with the number of unique
    design points rather than with all subsets of them.
    """
    unique = []
    seen = set()
    for point in points:
        key = tuple(round(float(value), 12) for value in point)
        if key not in seen:
            seen.add(key)
            unique.append([float(value) for value in point])
    if not unique:
        return {"inside": False, "design_points": 0, "method": "phase-i simplex"}
    dimension = len(unique[0])
    rows = [[1.0] * len(unique)]
    rhs = [1.0]
    for axis in range(dimension):
        rows.append([point[axis] for point in unique])
        rhs.append(float(target[axis]))
    feasible = _phase_one(rows, rhs, tolerance)
    return {
        "inside": feasible,
        "design_points": len(unique),
        "method": "phase-i simplex",
        "distance": None,
    }


def _phase_one(matrix, vector, tolerance):
    row_count = len(matrix)
    if row_count == 0:
        return True
    column_count = len(matrix[0])
    values = [list(row) for row in matrix]
    rhs = list(vector)
    for index in range(row_count):
        if rhs[index] < 0:
            values[index] = [-value for value in values[index]]
            rhs[index] = -rhs[index]
    artificial = column_count
    total = column_count + row_count
    tableau = []
    for row in range(row_count):
        line = [0.0] * (total + 1)
        line[:column_count] = values[row]
        line[artificial + row] = 1.0
        line[-1] = rhs[row]
        tableau.append(line)
    basis = [artificial + row for row in range(row_count)]
    # Minimise the sum of artificial variables.
    objective = [0.0] * (total + 1)
    for column in range(column_count):
        objective[column] = 0.0
    for column in range(artificial, total):
        objective[column] = 1.0
    for row in range(row_count):
        for column in range(total + 1):
            objective[column] -= tableau[row][column]
    for _ in range(10000):
        entering = None
        for column in range(total):
            if objective[column] < -tolerance:
                entering = column
                break
        if entering is None:
            break
        leaving = None
        best_ratio = None
        for row in range(row_count):
            coefficient = tableau[row][entering]
            if coefficient > tolerance:
                ratio = tableau[row][-1] / coefficient
                if (
                    best_ratio is None
                    or ratio < best_ratio - tolerance
                    or (
                        abs(ratio - best_ratio) <= tolerance
                        and basis[row] < basis[leaving]
                    )
                ):
                    best_ratio = ratio
                    leaving = row
        if leaving is None:
            return False
        pivot = tableau[leaving][entering]
        tableau[leaving] = [value / pivot for value in tableau[leaving]]
        for row in range(row_count):
            if row == leaving:
                continue
            factor = tableau[row][entering]
            if factor == 0:
                continue
            tableau[row] = [
                tableau[row][column] - factor * tableau[leaving][column]
                for column in range(total + 1)
            ]
        factor = objective[entering]
        objective = [
            objective[column] - factor * tableau[leaving][column]
            for column in range(total + 1)
        ]
        basis[leaving] = entering
    return objective[-1] >= -1e-7
