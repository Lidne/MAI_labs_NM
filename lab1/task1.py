def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def permutation_matrix_from_order(order):
    n = len(order)
    p = [[0.0] * n for _ in range(n)]
    for i, row_index in enumerate(order):
        p[i][row_index] = 1.0
    return p


def lu_decomposition_with_pivoting(a):
    n = len(a)
    u = [row[:] for row in a]
    l = identity_matrix(n)
    order = list(range(n))
    swap_count = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(u[i][k]))
        if abs(u[pivot_row][k]) < 1e-15:
            raise ValueError("Матрица вырождена: ведущий элемент равен нулю.")

        if pivot_row != k:
            # При перестановке строк в U нужно переставить уже найденные множители в L.
            u[k], u[pivot_row] = u[pivot_row], u[k]
            order[k], order[pivot_row] = order[pivot_row], order[k]
            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]
            swap_count += 1

        for i in range(k + 1, n):
            mu = u[i][k] / u[k][k]
            l[i][k] = mu
            for j in range(k, n):
                u[i][j] -= mu * u[k][j]
            u[i][k] = 0.0

    return l, u, order, swap_count


def apply_permutation(order, vector):
    return [vector[row_index] for row_index in order]


def forward_substitution(l, b):
    n = len(l)
    z = [0.0] * n
    for i in range(n):
        # Нижняя треугольная система Lz = b решается сверху вниз.
        z[i] = b[i] - sum(l[i][j] * z[j] for j in range(i))
    return z


def backward_substitution(u, z):
    n = len(u)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # Верхняя треугольная система Ux = z решается снизу вверх.
        tail = sum(u[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (z[i] - tail) / u[i][i]
    return x


def solve_with_lu(l, u, order, b):
    pb = apply_permutation(order, b)
    z = forward_substitution(l, pb)
    return backward_substitution(u, z)


def determinant_from_u(u, swap_count):
    det = 1.0
    for i in range(len(u)):
        det *= u[i][i]
    return -det if swap_count % 2 else det


def inverse_from_lu(l, u, order):
    n = len(l)
    inverse = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        column = solve_with_lu(l, u, order, e)
        for row in range(n):
            inverse[row][col] = column[row]
    return inverse


def print_vector(title, vector):
    print(title)
    for i, value in enumerate(vector, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def print_matrix(title, matrix):
    print(title)
    for row in matrix:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def main():
    a = [
        [-7.0, -9.0, 1.0, -9.0],
        [-6.0, -8.0, -5.0, 2.0],
        [-3.0, 6.0, 5.0, -9.0],
        [-2.0, 0.0, -5.0, -9.0],
    ]
    b = [29.0, 42.0, 11.0, 75.0]

    l, u, order, swap_count = lu_decomposition_with_pivoting(a)
    p = permutation_matrix_from_order(order)
    x = solve_with_lu(l, u, order, b)
    det_a = determinant_from_u(u, swap_count)
    inverse = inverse_from_lu(l, u, order)

    print_matrix("Матрица перестановки P:", p)
    print_matrix("Нижняя треугольная матрица L:", l)
    print_matrix("Верхняя треугольная матрица U:", u)
    print_vector("Решение СЛАУ:", x)
    print(f"det(A) = {det_a:.10f}\n")
    print_matrix("Обратная матрица A^(-1):", inverse)


if __name__ == "__main__":
    main()
