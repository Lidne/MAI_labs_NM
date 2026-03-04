def solve_linear_system(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise ValueError("Singular matrix in normal equations.")
        m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div

        for i in range(n):
            if i == col:
                continue
            factor = m[i][col]
            for j in range(col, n + 1):
                m[i][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def least_squares_poly(x, y, degree):
    n = degree + 1
    sums = [0.0] * (2 * degree + 1)
    for p in range(2 * degree + 1):
        sums[p] = sum(xi ** p for xi in x)

    rhs = [0.0] * n
    for i in range(n):
        rhs[i] = sum((x[j] ** i) * y[j] for j in range(len(x)))

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = sums[i + j]

    return solve_linear_system(matrix, rhs)


def poly_value(coeffs, x):
    return sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))


def sse(coeffs, x, y):
    return sum((poly_value(coeffs, x[i]) - y[i]) ** 2 for i in range(len(x)))


def build_plot(x, y, c1, c2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Plot is skipped.")
        return

    x_min, x_max = min(x), max(x)
    points = 300
    xs = [x_min + (x_max - x_min) * i / (points - 1) for i in range(points)]
    y1 = [poly_value(c1, xi) for xi in xs]
    y2 = [poly_value(c2, xi) for xi in xs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, y1, label="Least squares, degree 1")
    plt.plot(xs, y2, label="Least squares, degree 2")
    plt.scatter(x, y, color="black", label="Table points")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Task 3.3 approximation")
    output = "lab3/task3_plot.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")


def main():
    x = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    y = [-1.3734, -1.2490, -0.7854, 0.7854, 1.2490, 1.3734]

    coeff_deg1 = least_squares_poly(x, y, 1)
    coeff_deg2 = least_squares_poly(x, y, 2)
    err1 = sse(coeff_deg1, x, y)
    err2 = sse(coeff_deg2, x, y)

    print("Task 3.3: Least squares approximation")
    print(f"Degree 1 coefficients: {coeff_deg1}")
    print(f"Degree 1 SSE: {err1:.12f}")
    print(f"Degree 2 coefficients: {coeff_deg2}")
    print(f"Degree 2 SSE: {err2:.12f}")

    build_plot(x, y, coeff_deg1, coeff_deg2)


if __name__ == "__main__":
    main()
