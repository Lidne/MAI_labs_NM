def vector_max_norm(vector):
    return max(abs(value) for value in vector)


def matrix_row_sum_norm(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def transform_to_iteration_form(a, b):
    n = len(a)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n

    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            raise ValueError("Нулевой диагональный элемент не позволяет построить x = β + αx.")
        beta[i] = b[i] / a[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -a[i][j] / a[i][i]

    return alpha, beta


def split_alpha(alpha):
    n = len(alpha)
    b_matrix = [[0.0] * n for _ in range(n)]
    c_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if j < i:
                b_matrix[i][j] = alpha[i][j]
            else:
                c_matrix[i][j] = alpha[i][j]

    return b_matrix, c_matrix


def simple_iterations(a, b, eps=1e-8, max_iter=100000):
    alpha, beta = transform_to_iteration_form(a, b)
    alpha_norm = matrix_row_sum_norm(alpha)
    use_estimate = alpha_norm < 1.0
    x_prev = beta[:]
    history = []

    for iteration in range(1, max_iter + 1):
        x_curr = [
            beta[i] + sum(alpha[i][j] * x_prev[j] for j in range(len(a)))
            for i in range(len(a))
        ]
        diff = vector_max_norm([x_curr[i] - x_prev[i] for i in range(len(a))])
        estimate = alpha_norm / (1.0 - alpha_norm) * diff if use_estimate else diff
        history.append((iteration, diff, estimate))
        if estimate <= eps:
            return {
                "solution": x_curr,
                "iterations": iteration,
                "alpha": alpha,
                "beta": beta,
                "alpha_norm": alpha_norm,
                "used_estimate": use_estimate,
                "history": history,
            }
        x_prev = x_curr

    raise RuntimeError("Метод простых итераций не сошелся за отведенное число шагов.")


def gauss_seidel(a, b, eps=1e-8, max_iter=100000):
    alpha, beta = transform_to_iteration_form(a, b)
    alpha_norm = matrix_row_sum_norm(alpha)
    _, c_matrix = split_alpha(alpha)
    c_norm = matrix_row_sum_norm(c_matrix)
    use_estimate = alpha_norm < 1.0
    x_prev = beta[:]
    history = []
    n = len(a)

    for iteration in range(1, max_iter + 1):
        x_curr = x_prev[:]
        for i in range(n):
            left_sum = sum(alpha[i][j] * x_curr[j] for j in range(i))
            right_sum = sum(alpha[i][j] * x_prev[j] for j in range(i, n))
            x_curr[i] = beta[i] + left_sum + right_sum

        diff = vector_max_norm([x_curr[i] - x_prev[i] for i in range(n)])
        estimate = c_norm / (1.0 - alpha_norm) * diff if use_estimate else diff
        history.append((iteration, diff, estimate))
        if estimate <= eps:
            return {
                "solution": x_curr,
                "iterations": iteration,
                "alpha": alpha,
                "beta": beta,
                "alpha_norm": alpha_norm,
                "c_norm": c_norm,
                "used_estimate": use_estimate,
                "history": history,
            }
        x_prev = x_curr

    raise RuntimeError("Метод Зейделя не сошелся за отведенное число шагов.")


def print_matrix(title, matrix):
    print(title)
    for row in matrix:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def print_vector(title, vector, prefix):
    print(title)
    for i, value in enumerate(vector, start=1):
        print(f"{prefix}{i} = {value:.10f}")
    print()


def print_result(title, result):
    print(title)
    for i, value in enumerate(result["solution"], start=1):
        print(f"x{i} = {value:.10f}")
    print(f"Количество итераций: {result['iterations']}")
    print(f"||alpha||_c = {result['alpha_norm']:.10f}")
    if "c_norm" in result:
        print(f"||C||_c = {result['c_norm']:.10f}")
    if result["used_estimate"]:
        print(f"Останов по оценке погрешности: {result['history'][-1][2]:.10e}")
    else:
        print(f"Останов по ||x^(k) - x^(k-1)||: {result['history'][-1][2]:.10e}")
    print()


def main():
    a = [
        [12.0, -3.0, -1.0, 3.0],
        [5.0, 20.0, 9.0, 1.0],
        [6.0, -3.0, -21.0, -7.0],
        [8.0, -7.0, 3.0, -27.0],
    ]
    b = [-31.0, 90.0, 119.0, 71.0]
    eps = 1e-8

    alpha, beta = transform_to_iteration_form(a, b)
    simple_result = simple_iterations(a, b, eps)
    seidel_result = gauss_seidel(a, b, eps)

    print_matrix("Матрица alpha эквивалентной системы x = beta + alpha * x:", alpha)
    print_vector("Вектор beta:", beta, "beta")
    print_result("Метод простых итераций:", simple_result)
    print_result("Метод Зейделя:", seidel_result)


if __name__ == "__main__":
    main()
