def transform_to_iteration_form(a, b):
    n = len(a)
    c = [0.0] * n
    alpha = [[0.0] * n for _ in range(n)]

    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            raise ValueError("Нулевой диагональный элемент")
        c[i] = b[i] / a[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -a[i][j] / a[i][i]
    return alpha, c


def simple_iterations(a, b, eps=1e-8, max_iter=100000):
    alpha, c = transform_to_iteration_form(a, b)
    n = len(a)
    x = [0.0] * n

    for k in range(1, max_iter + 1):
        x_new = [c[i] + sum(alpha[i][j] * x[j] for j in range(n)) for i in range(n)]
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new
        if diff < eps:
            return x, k
    raise RuntimeError("Метод простых итераций не сошелся")


def gauss_seidel(a, b, eps=1e-8, max_iter=100000):
    alpha, c = transform_to_iteration_form(a, b)
    n = len(a)
    x = [0.0] * n

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            left = sum(alpha[i][j] * x[j] for j in range(i))
            right = sum(alpha[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = c[i] + left + right
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if diff < eps:
            return x, k
    raise RuntimeError("Метод Гаусса-Зейделя не сошелся")


def print_solution(title, x, iterations):
    print(title)
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.10f}")
    print(f"iterations = {iterations}\n")


def main():
    a = [
        [12.0, -3.0, -1.0, 3.0],
        [5.0, 20.0, 9.0, 1.0],
        [6.0, -3.0, -21.0, -7.0],
        [8.0, -7.0, 3.0, -27.0],
    ]
    b = [-31.0, 90.0, 119.0, 71.0]
    eps = 1e-8

    x_iter, k_iter = simple_iterations(a, b, eps)
    x_seidel, k_seidel = gauss_seidel(a, b, eps)

    print_solution("Simple iterations:", x_iter, k_iter)
    print_solution("Gauss-Seidel:", x_seidel, k_seidel)


if __name__ == "__main__":
    main()
