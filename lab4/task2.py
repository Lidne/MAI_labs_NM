import math
from typing import List, Tuple


def exact_solution(x: float) -> float:
    return math.exp(x) * x * x


def f_system(x: float, y: float, z: float) -> Tuple[float, float]:
    dy = z
    dz = ((2 * x + 1) * z - (x + 1) * y) / x
    return dy, dz


def rk4_step(x: float, y: float, z: float, h: float) -> Tuple[float, float]:
    k1y, k1z = f_system(x, y, z)
    k2y, k2z = f_system(x + h / 2, y + h * k1y / 2, z + h * k1z / 2)
    k3y, k3z = f_system(x + h / 2, y + h * k2y / 2, z + h * k2z / 2)
    k4y, k4z = f_system(x + h, y + h * k3y, z + h * k3z)

    y_next = y + h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    z_next = z + h * (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    return y_next, z_next


def integrate_ivp(s: float, h: float) -> Tuple[List[float], List[float], List[float]]:
    x0, x_end = 1.0, 2.0
    n = int(round((x_end - x0) / h))
    xs = [x0 + i * h for i in range(n + 1)]

    y = [0.0] * (n + 1)
    z = [0.0] * (n + 1)
    y[0] = s
    z[0] = 3 * math.e

    for i in range(n):
        y[i + 1], z[i + 1] = rk4_step(xs[i], y[i], z[i], h)

    return xs, y, z


def shooting_method(h: float, s0: float = 0.0, s1: float = 8.0, tol: float = 1e-12, max_iter: int = 100):
    def phi(s: float) -> float:
        _, y, z = integrate_ivp(s, h)
        return z[-1] - 2 * y[-1]

    p0 = phi(s0)
    p1 = phi(s1)

    for _ in range(max_iter):
        if abs(p1 - p0) < 1e-16:
            raise RuntimeError("Secant denominator is too small in shooting method.")

        s2 = s1 - p1 * (s1 - s0) / (p1 - p0)
        p2 = phi(s2)

        if abs(p2) < tol:
            return integrate_ivp(s2, h)

        s0, p0 = s1, p1
        s1, p1 = s2, p2

    raise RuntimeError("Shooting method did not converge.")


def gaussian_elimination(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r
        if abs(a[pivot][i]) < 1e-16:
            raise RuntimeError("Singular matrix in finite-difference solver.")

        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]

        div = a[i][i]
        for j in range(i, n):
            a[i][j] /= div
        b[i] /= div

        for r in range(i + 1, n):
            factor = a[r][i]
            if factor == 0.0:
                continue
            for j in range(i, n):
                a[r][j] -= factor * a[i][j]
            b[r] -= factor * b[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = b[i] - s
    return x


def finite_difference_method(h: float) -> Tuple[List[float], List[float]]:
    x0, x_end = 1.0, 2.0
    n = int(round((x_end - x0) / h))
    xs = [x0 + i * h for i in range(n + 1)]

    size = n + 1
    a = [[0.0 for _ in range(size)] for _ in range(size)]
    b = [0.0 for _ in range(size)]

    a[0][0] = -3.0 / (2 * h)
    a[0][1] = 4.0 / (2 * h)
    a[0][2] = -1.0 / (2 * h)
    b[0] = 3 * math.e

    for i in range(1, n):
        x = xs[i]
        p = -(2 * x + 1) / x
        q = (x + 1) / x

        a[i][i - 1] = 1 / (h * h) - p / (2 * h)
        a[i][i] = -2 / (h * h) + q
        a[i][i + 1] = 1 / (h * h) + p / (2 * h)
        b[i] = 0.0

    a[n][n] = 3.0 / (2 * h) - 2.0
    a[n][n - 1] = -4.0 / (2 * h)
    a[n][n - 2] = 1.0 / (2 * h)
    b[n] = 0.0

    y = gaussian_elimination(a, b)
    return xs, y


def runge_romberg(y_h: List[float], y_h2: List[float], p: int) -> List[float]:
    return [abs(y_h2[2 * i] - y_h[i]) / (2**p - 1) for i in range(len(y_h))]


def print_result(name: str, xs: List[float], y_h: List[float], y_h2: List[float], order: int) -> None:
    y_exact = [exact_solution(x) for x in xs]
    exact_err = [abs(yn - ye) for yn, ye in zip(y_h, y_exact)]
    rr_err = runge_romberg(y_h, y_h2, order)

    print(f"\n{name}")
    print("x      y_num           y_exact         |err_exact|      RR_error")
    for x, yn, ye, ee, re in zip(xs, y_h, y_exact, exact_err, rr_err):
        print(f"{x:0.1f}  {yn:14.10f}  {ye:14.10f}  {ee:12.4e}  {re:10.4e}")

    print(f"Max |err_exact| = {max(exact_err):.4e}")
    print(f"Max RR error    = {max(rr_err):.4e}")


def main() -> None:
    print("Lab 4.2: Boundary value problem for ODE of 2nd order")
    print("x*y'' - (2x+1)y' + (x+1)y = 0, x in [1,2]")
    print("y'(1)=3e, y'(2)-2y(2)=0")
    print("Exact solution: y = e^x * x^2")

    h = 0.1

    xs_s, y_s, _ = shooting_method(h)
    _, y_s_h2, _ = shooting_method(h / 2)
    print_result("Shooting method (RK4 + secant)", xs_s, y_s, y_s_h2, order=4)

    xs_fd, y_fd = finite_difference_method(h)
    _, y_fd_h2 = finite_difference_method(h / 2)
    print_result("Finite-difference method", xs_fd, y_fd, y_fd_h2, order=2)


if __name__ == "__main__":
    main()
