import math
from typing import Callable, List, Tuple


State = Tuple[float, float]
RHS = Callable[[float, State], State]


def f_system(x: float, state: State) -> State:
    y, z = state
    dy = z
    dz = -z * math.tan(x) - math.cos(x) ** 2
    return dy, dz


def exact_solution(x: float) -> float:
    return math.sin(x) - 0.5 * (math.sin(x) ** 2)


def euler_method(
    f: RHS, x0: float, y0: State, x_end: float, h: float
) -> Tuple[List[float], List[State]]:
    n = int(round((x_end - x0) / h))
    xs = [x0 + i * h for i in range(n + 1)]
    ys = [y0]

    for i in range(n):
        xi = xs[i]
        yi = ys[-1]
        k1 = f(xi, yi)
        ys.append((yi[0] + h * k1[0], yi[1] + h * k1[1]))

    return xs, ys


def rk4_method(
    f: RHS, x0: float, y0: State, x_end: float, h: float
) -> Tuple[List[float], List[State]]:
    n = int(round((x_end - x0) / h))
    xs = [x0 + i * h for i in range(n + 1)]
    ys = [y0]

    for i in range(n):
        xi = xs[i]
        yi = ys[-1]

        k1 = f(xi, yi)
        k2 = f(xi + h / 2, (yi[0] + h * k1[0] / 2, yi[1] + h * k1[1] / 2))
        k3 = f(xi + h / 2, (yi[0] + h * k2[0] / 2, yi[1] + h * k2[1] / 2))
        k4 = f(xi + h, (yi[0] + h * k3[0], yi[1] + h * k3[1]))

        y_next = (
            yi[0] + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
            yi[1] + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
        )
        ys.append(y_next)

    return xs, ys


def adams4_method(
    f: RHS, x0: float, y0: State, x_end: float, h: float
) -> Tuple[List[float], List[State]]:
    n = int(round((x_end - x0) / h))
    if n < 4:
        raise ValueError("Adams-4 requires at least 4 steps.")

    xs = [x0 + i * h for i in range(n + 1)]
    _, ys_start = rk4_method(f, x0, y0, x0 + 3 * h, h)
    ys = ys_start[:]

    for i in range(3, n):
        f_i = f(xs[i], ys[i])
        f_i1 = f(xs[i - 1], ys[i - 1])
        f_i2 = f(xs[i - 2], ys[i - 2])
        f_i3 = f(xs[i - 3], ys[i - 3])

        y_next = (
            ys[i][0] + h * (55 * f_i[0] - 59 * f_i1[0] + 37 * f_i2[0] - 9 * f_i3[0]) / 24,
            ys[i][1] + h * (55 * f_i[1] - 59 * f_i1[1] + 37 * f_i2[1] - 9 * f_i3[1]) / 24,
        )
        ys.append(y_next)

    return xs, ys


def runge_romberg(
    y_h: List[float], y_h2: List[float], p: int
) -> List[float]:
    return [abs(y_h2[2 * i] - y_h[i]) / (2**p - 1) for i in range(len(y_h))]


def solve_with_method(
    name: str, method: Callable[[RHS, float, State, float, float], Tuple[List[float], List[State]]], order: int
) -> None:
    x0, x_end = 0.0, 1.0
    y0 = (0.0, 1.0)
    h = 0.1

    xs_h, sol_h = method(f_system, x0, y0, x_end, h)
    xs_h2, sol_h2 = method(f_system, x0, y0, x_end, h / 2)

    y_h = [s[0] for s in sol_h]
    y_h2 = [s[0] for s in sol_h2]
    y_exact = [exact_solution(x) for x in xs_h]

    rr_err = runge_romberg(y_h, y_h2, order)
    exact_err = [abs(yn - ye) for yn, ye in zip(y_h, y_exact)]

    print(f"\n{name} (h = {h})")
    print("x      y_num           y_exact         |err_exact|      RR_error")
    for x, yn, ye, ee, re in zip(xs_h, y_h, y_exact, exact_err, rr_err):
        print(f"{x:0.1f}  {yn:14.10f}  {ye:14.10f}  {ee:12.4e}  {re:10.4e}")

    print(f"Max |err_exact| = {max(exact_err):.4e}")
    print(f"Max RR error    = {max(rr_err):.4e}")


def main() -> None:
    print("Lab 4.1: Cauchy problem for ODE of 2nd order")
    print("y'' + y' * tan(x) + cos^2(x) = 0")
    print("y(0)=0, y'(0)=1, x in [0,1], h=0.1")
    print("Exact solution: y = sin(x) - 0.5*sin^2(x)")

    solve_with_method("Euler method", euler_method, order=1)
    solve_with_method("Runge-Kutta 4 method", rk4_method, order=4)
    solve_with_method("Adams 4 method", adams4_method, order=4)


if __name__ == "__main__":
    main()
