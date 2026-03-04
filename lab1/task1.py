from copy import deepcopy


def lu_decomposition_with_pivoting(a):
    n = len(a)
    lu = deepcopy(a)
    p = list(range(n))
    swap_count = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if abs(lu[pivot_row][k]) < 1e-15:
            raise ValueError("Matrix is singular")

        if pivot_row != k:
            lu[k], lu[pivot_row] = lu[pivot_row], lu[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            swap_count += 1

        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]

    return lu, p, swap_count


def lu_solve(lu, p, b):
    n = len(lu)
    pb = [b[p[i]] for i in range(n)]

    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(lu[i][j] * y[j] for j in range(i))

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lu[i][j] * x[j] for j in range(i + 1, n))) / lu[i][i]

    return x


def determinant_from_lu(lu, swap_count):
    det = 1.0
    for i in range(len(lu)):
        det *= lu[i][i]
    if swap_count % 2 == 1:
        det = -det
    return det


def inverse_from_lu(lu, p):
    n = len(lu)
    inv = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        x = lu_solve(lu, p, e)
        for row in range(n):
            inv[row][col] = x[row]
    return inv


def print_vector(name, v):
    print(name)
    for i, value in enumerate(v, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def print_matrix(name, m):
    print(name)
    for row in m:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def main():
    a = [
        [-7.0, 3.0, -4.0, 7.0],
        [8.0, -1.0, -7.0, 6.0],
        [9.0, 9.0, 3.0, -6.0],
        [-7.0, -9.0, -8.0, -5.0],
    ]
    b = [-126.0, 29.0, 27.0, 34.0]

    lu, p, swap_count = lu_decomposition_with_pivoting(a)
    x = lu_solve(lu, p, b)
    det = determinant_from_lu(lu, swap_count)
    inv = inverse_from_lu(lu, p)

    print_vector("Solution of SLAE:", x)
    print(f"Determinant: {det:.10f}\n")
    print_matrix("Inverse matrix:", inv)


if __name__ == "__main__":
    main()
