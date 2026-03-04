def first_derivative_central(y_prev, y_next, h):
    return (y_next - y_prev) / (2.0 * h)


def second_derivative_central(y_prev, y_curr, y_next, h):
    return (y_next - 2.0 * y_curr + y_prev) / (h * h)


def main():
    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [0.0, 0.97943, 1.8415, 2.4975, 2.9093]
    x_star = 1.0

    i = x.index(x_star)
    h = x[i] - x[i - 1]

    d1 = first_derivative_central(y[i - 1], y[i + 1], h)
    d2 = second_derivative_central(y[i - 1], y[i], y[i + 1], h)

    print("Task 3.4: Numerical differentiation")
    print(f"x* = {x_star}")
    print(f"f'(x*)  ≈ {d1:.12f}")
    print(f"f''(x*) ≈ {d2:.12f}")


if __name__ == "__main__":
    main()
