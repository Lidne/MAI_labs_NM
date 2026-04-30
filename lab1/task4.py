import math


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


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


def jacobi_measure(a):
    """sqrt(sum(a_ij^2, i<j))"""
    total = 0.0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            total += a[i][j] * a[i][j]
    return math.sqrt(total)


def max_off_diagonal_index(a):
    n = len(a)
    row, col = 0, 1
    best = abs(a[0][1])
    for i in range(n):
        for j in range(i + 1, n):
            current = abs(a[i][j])
            if current > best:
                best = current
                row, col = i, j
    return row, col


def jacobi_rotation_method(a, eps=1e-10, max_iter=10000):
    n = len(a)
    current = [row[:] for row in a]
    eigenvectors = identity(n)
    history = [jacobi_measure(current)]

    for _ in range(max_iter):
        if history[-1] <= eps:
            break

        i, j = max_off_diagonal_index(current)
        if abs(current[i][j]) < 1e-15:
            break

        if abs(current[i][i] - current[j][j]) < 1e-15:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan(2.0 * current[i][j] / (current[i][i] - current[j][j]))

        c = math.cos(phi)
        s = math.sin(phi)

        rotation = identity(n)
        rotation[i][i] = c
        rotation[j][j] = c
        rotation[i][j] = -s
        rotation[j][i] = s

        current = matmul(matmul(transpose(rotation), current), rotation)
        eigenvectors = matmul(eigenvectors, rotation)
        history.append(jacobi_measure(current))

    eigenvalues = [current[i][i] for i in range(n)]
    return eigenvalues, eigenvectors, history


def main():
    a = [
        [4.0, 7.0, -1.0],
        [7.0, 9.0, -6.0],
        [-1.0, -6.0, -4.0],
    ]
    eps = 1e-10

    eigenvalues, eigenvectors, history = jacobi_rotation_method(a, eps=eps)

    print("Собственные значения:")
    for i, value in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {value:.10f}")

    print("\nСобственные векторы (столбцы матрицы U):")
    for row in eigenvectors:
        print(" ".join(f"{value: .10f}" for value in row))

    print("\nИстория t(A^(k)) = sqrt(sum(a_lm^2), l < m):")
    for iteration, value in enumerate(history):
        print(f"iter {iteration:3d}: {value:.12e}")

    print("\nСтолбцы итоговой матрицы U являются собственными векторами матрицы A.")


if __name__ == "__main__":
    main()
