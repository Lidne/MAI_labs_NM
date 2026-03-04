def thomas_algorithm(a, b, c, d):
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("Invalid tridiagonal dimensions")

    cp = [0.0] * (n - 1)
    dp = [0.0] * n

    if abs(b[0]) < 1e-15:
        raise ValueError("Zero pivot in Thomas algorithm")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise ValueError("Zero pivot in Thomas algorithm")
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
    a = [6.0, -3.0, 9.0, 5.0]
    b = [-7.0, 12.0, 5.0, 21.0, -6.0]
    c = [-6.0, 0.0, 0.0, 8.0]
    d = [-75.0, 126.0, 13.0, -40.0, -24.0]

    x = thomas_algorithm(a, b, c, d)

    print("Solution of tridiagonal SLAE:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.10f}")


if __name__ == "__main__":
    main()
