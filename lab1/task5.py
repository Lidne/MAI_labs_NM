import math


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def householder_matrix(column, start, size):
    tail = column[start:]
    tail_norm = vector_norm(tail)
    if tail_norm < 1e-15:
        return identity(size)

    sign = 1.0 if tail[0] >= 0 else -1.0
    v = [0.0] * size
    # Вектор v строится по формуле b + sign(b1) * ||b|| * e1.
    v[start] = tail[0] + sign * tail_norm
    for i in range(start + 1, size):
        v[i] = column[i]

    vv = sum(value * value for value in v)
    if vv < 1e-15:
        return identity(size)

    h = identity(size)
    for i in range(size):
        for j in range(size):
            h[i][j] -= 2.0 * v[i] * v[j] / vv
    return h


def qr_decomposition_householder(a):
    n = len(a)
    r = [row[:] for row in a]
    q = identity(n)

    for k in range(n - 1):
        column = [r[i][k] for i in range(n)]
        h = householder_matrix(column, k, n)
        r = matmul(h, r)
        q = matmul(q, h)

    return q, r


def subdiagonal_abs(a, col, size):
    return math.sqrt(sum(a[row][col] * a[row][col] for row in range(col + 1, size)))


def eigenvalues_from_2x2_block(a, start):
    a11 = a[start][start]
    a12 = a[start][start + 1]
    a21 = a[start + 1][start]
    a22 = a[start + 1][start + 1]
    trace = a11 + a22
    determinant = a11 * a22 - a12 * a21
    discriminant = trace * trace - 4.0 * determinant

    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        return [(trace + root) / 2.0, (trace - root) / 2.0]

    root = math.sqrt(-discriminant)
    return [
        complex(trace / 2.0, root / 2.0),
        complex(trace / 2.0, -root / 2.0),
    ]


def pair_distance(first_pair, second_pair):
    direct = max(abs(first_pair[0] - second_pair[0]), abs(first_pair[1] - second_pair[1]))
    swapped = max(abs(first_pair[0] - second_pair[1]), abs(first_pair[1] - second_pair[0]))
    return min(direct, swapped)


def qr_algorithm(a, eps=1e-10, max_iter=10000):
    n = len(a)
    current = [row[:] for row in a]
    eigenvalues = [None] * n
    active = n - 1
    iterations = 0
    previous_block_eigenvalues = None

    while active > 0 and iterations < max_iter:
        size = active + 1
        submatrix = [row[:size] for row in current[:size]]
        q, r = qr_decomposition_householder(submatrix)
        next_submatrix = matmul(r, q)

        for i in range(size):
            for j in range(size):
                current[i][j] = next_submatrix[i][j]

        for i in range(1, size):
            if abs(current[i][i - 1]) <= eps:
                current[i][i - 1] = 0.0

        iterations += 1

        if abs(current[active][active - 1]) <= eps:
            eigenvalues[active] = current[active][active]
            current[active][active - 1] = 0.0
            active -= 1
            previous_block_eigenvalues = None
            continue

        if active == 1:
            block_eigenvalues = eigenvalues_from_2x2_block(current, 0)
            if previous_block_eigenvalues is not None and pair_distance(block_eigenvalues, previous_block_eigenvalues) <= eps:
                eigenvalues[0], eigenvalues[1] = block_eigenvalues
                return eigenvalues, iterations, current
            previous_block_eigenvalues = block_eigenvalues

    if active == 0 and eigenvalues[0] is None:
        eigenvalues[0] = current[0][0]
    elif active == 1 and (eigenvalues[0] is None or eigenvalues[1] is None):
        block_eigenvalues = eigenvalues_from_2x2_block(current, 0)
        eigenvalues[0], eigenvalues[1] = block_eigenvalues

    for i in range(n):
        if eigenvalues[i] is None:
            eigenvalues[i] = current[i][i]

    return eigenvalues, iterations, current


def print_matrix(title, matrix):
    print(title)
    for row in matrix:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def main():
    a = [
        [-5.0, 8.0, 4.0],
        [4.0, 2.0, 6.0],
        [-2.0, 5.0, -6.0],
    ]
    eps = 1e-10

    q, r = qr_decomposition_householder(a)
    eigenvalues, iterations, quasi_triangular = qr_algorithm(a, eps=eps)

    print_matrix("Матрица Q в QR-разложении:", q)
    print_matrix("Матрица R в QR-разложении:", r)
    print_matrix("Матрица A^(k), близкая к квазитреугольной:", quasi_triangular)

    print("Собственные значения по QR-алгоритму:")
    for i, value in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {value}")
    print(f"\nКоличество QR-итераций: {iterations}")
    print("Если остается блок 2x2, его собственные значения берутся из квадратного уравнения.")


if __name__ == "__main__":
    main()
