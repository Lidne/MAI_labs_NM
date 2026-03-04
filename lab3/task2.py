def natural_cubic_spline_coefficients(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]

    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])

    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n

    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1.0
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)
    a = y[:-1]

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1] if j < n - 1 else 0.0
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (2.0 * c[j] + c[j + 1]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    return a, b, c[:-1], d


def spline_value(x_nodes, coeffs, x_star):
    a, b, c, d = coeffs
    interval = len(x_nodes) - 2
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_star <= x_nodes[i + 1]:
            interval = i
            break

    dx = x_star - x_nodes[interval]
    return a[interval] + b[interval] * dx + c[interval] * dx * dx + d[interval] * dx * dx * dx


def main():
    x = [-3.0, -1.0, 1.0, 3.0, 5.0]
    y = [-1.2490, -0.78540, 0.78540, 1.2490, 1.3734]
    x_star = -0.5

    coeffs = natural_cubic_spline_coefficients(x, y)
    y_star = spline_value(x, coeffs, x_star)

    print("Task 3.2: Natural cubic spline")
    print(f"x* = {x_star}")
    print(f"S(x*) = {y_star:.12f}")


if __name__ == "__main__":
    main()
