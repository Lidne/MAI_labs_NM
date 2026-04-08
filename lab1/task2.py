def right_sweep(lower, diag, upper, rhs):
    n = len(diag)
    if len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError("Размеры коэффициентов трехдиагональной системы некорректны.")

    a = [0.0] + lower[:]
    b = diag[:]
    c = upper[:] + [0.0]
    d = rhs[:]

    p = [0.0] * n
    q = [0.0] * n

    if abs(b[0]) < 1e-15:
        raise ValueError("На первом шаге прогонки знаменатель равен нулю.")
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        # Крайние уравнения выделяются отдельно, потому что a1 = 0 и cn = 0.
        denominator = b[i] + a[i] * p[i - 1]
        if abs(denominator) < 1e-15:
            raise ValueError(f"На шаге {i + 1} прогонки знаменатель равен нулю.")
        p[i] = -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator

    denominator = b[n - 1] + a[n - 1] * p[n - 2]
    if abs(denominator) < 1e-15:
        raise ValueError("На последнем шаге прогонки знаменатель равен нулю.")
    p[n - 1] = 0.0
    q[n - 1] = (d[n - 1] - a[n - 1] * q[n - 2]) / denominator

    x = [0.0] * n
    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return p, q, x


def main():
    lower = [-2.0, 2.0, -8.0, -7.0]
    diag = [8.0, 12.0, -9.0, 17.0, 13.0]
    upper = [-4.0, -7.0, 1.0, -4.0]
    rhs = [32.0, 15.0, -10.0, 133.0, -76.0]

    p, q, x = right_sweep(lower, diag, upper, rhs)

    print("Прогоночные коэффициенты P_i:")
    for i, value in enumerate(p, start=1):
        print(f"P{i} = {value:.10f}")
    print()

    print("Прогоночные коэффициенты Q_i:")
    for i, value in enumerate(q, start=1):
        print(f"Q{i} = {value:.10f}")
    print()

    print("Решение СЛАУ методом правой прогонки:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.10f}")


if __name__ == "__main__":
    main()
