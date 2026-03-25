def thomas_algorithm(a, b, c, d):
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("Неверные размеры трехдиагональной матрицы")

    cp = [0.0] * (n - 1)
    dp = [0.0] * n

    if abs(b[0]) < 1e-15:
        raise ValueError("Нулевой опорный элемент в алгоритме Томаса")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise ValueError("Нулевой опорный элемент в алгоритме Томаса")
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def main():
    # a - subdiagonal, b - diagonal, c - superdiagonal
    a = [-2.0, 2.0, -8.0, -7.0]
    b = [8.0, 12.0, -9.0, 17.0, 13.0]
    c = [-4.0, -7.0, 1.0, -4.0]
    d = [32.0, 15.0, -10.0, 133.0, -76.0]

    x = thomas_algorithm(a, b, c, d)

    print("Solution of tridiagonal SLAE:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.10f}")


if __name__ == "__main__":
    main()
