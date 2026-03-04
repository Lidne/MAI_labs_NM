import math


def lagrange_value(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


def newton_divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = y_nodes[:]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coef


def newton_value(x_nodes, coef, x):
    result = coef[0]
    prod = 1.0
    for i in range(1, len(coef)):
        prod *= x - x_nodes[i - 1]
        result += coef[i] * prod
    return result


def print_case(label, x_nodes, x_star):
    y_nodes = [math.atan(x) for x in x_nodes]
    y_true = math.atan(x_star)

    y_lagrange = lagrange_value(x_nodes, y_nodes, x_star)
    newton_coef = newton_divided_differences(x_nodes, y_nodes)
    y_newton = newton_value(x_nodes, newton_coef, x_star)

    print(f"\nCase {label}")
    print(f"Nodes: {x_nodes}")
    print(f"x* = {x_star}")
    print(f"f(x*) = arctan(x*) = {y_true:.12f}")
    print(f"Lagrange P(x*) = {y_lagrange:.12f}")
    print(f"Newton   P(x*) = {y_newton:.12f}")
    print(f"|f(x*) - P_L(x*)| = {abs(y_true - y_lagrange):.12e}")
    print(f"|f(x*) - P_N(x*)| = {abs(y_true - y_newton):.12e}")


def main():
    x_star = -0.5
    print("Task 3.1: Interpolation for y = arctan(x)")
    print_case("a", [-3.0, -1.0, 1.0, 3.0], x_star)
    print_case("b", [-3.0, 0.0, 1.0, 3.0], x_star)


if __name__ == "__main__":
    main()
