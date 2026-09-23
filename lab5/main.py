import math

import numpy as np

L = math.pi
BOUNDARIES = ("2p1", "3p2", "2p2")


def exact(x, t, a):
    return math.exp(-a * a * t) * math.sin(x)


def g0(t, a):
    return math.exp(-a * a * t)


def g1(t, a):
    return -math.exp(-a * a * t)


def max_error(u, xs, t, a):
    return max(abs(float(ui) - exact(float(x), t, a)) for ui, x in zip(u, xs))


def _sweep(lower, diag, upper, rhs):
    n = len(diag)
    A = np.zeros(n)
    B = np.zeros(n)
    for j in range(n):
        denominator = diag[j] + (lower[j] * A[j - 1] if j else 0.0)
        if abs(denominator) < 1e-14:
            raise ArithmeticError("Нулевой ведущий элемент при прогонке")
        A[j] = -upper[j] / denominator if j < n - 1 else 0.0
        B[j] = (rhs[j] - (lower[j] * B[j - 1] if j else 0.0)) / denominator
    result = np.empty(n)
    result[-1] = B[-1]
    for j in range(n - 2, -1, -1):
        result[j] = A[j] * result[j + 1] + B[j]
    return result


def _step_explicit(u, bc, h, sigma, t, t_new, a):
    v = np.empty_like(u)
    v[1:-1] = sigma * u[:-2] + (1 - 2 * sigma) * u[1:-1] + sigma * u[2:]
    if bc == "2p1":
        v[0] = v[1] - h * g0(t_new, a)
        v[-1] = v[-2] + h * g1(t_new, a)
    elif bc == "3p2":
        v[0] = (4 * v[1] - v[2] - 2 * h * g0(t_new, a)) / 3
        v[-1] = (4 * v[-2] - v[-3] + 2 * h * g1(t_new, a)) / 3
    else:
        v[0] = (1 - 2 * sigma) * u[0] + 2 * sigma * u[1] - 2 * sigma * h * g0(t, a)
        v[-1] = 2 * sigma * u[-2] + (1 - 2 * sigma) * u[-1] + 2 * sigma * h * g1(t, a)
    return v


def _step_implicit(u, bc, h, sigma, t, t_new, a):
    theta = 1.0
    n = len(u)
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    lower[1:-1] = -theta * sigma
    diag[1:-1] = 1 + 2 * theta * sigma
    upper[1:-1] = -theta * sigma
    rhs[1:-1] = u[1:-1] + (1 - theta) * sigma * (u[:-2] - 2 * u[1:-1] + u[2:])

    if bc == "2p1":
        diag[0], upper[0], rhs[0] = -1 / h, 1 / h, g0(t_new, a)
        lower[-1], diag[-1], rhs[-1] = -1 / h, 1 / h, g1(t_new, a)
    elif bc == "3p2":
        left_factor = (-1 / (2 * h)) / upper[1]
        diag[0] = -3 / (2 * h) - left_factor * lower[1]
        upper[0] = 2 / h - left_factor * diag[1]
        rhs[0] = g0(t_new, a) - left_factor * rhs[1]

        right_factor = (1 / (2 * h)) / lower[-2]
        lower[-1] = -2 / h - right_factor * diag[-2]
        diag[-1] = 3 / (2 * h) - right_factor * upper[-2]
        rhs[-1] = g1(t_new, a) - right_factor * rhs[-2]
    else:
        diag[0] = diag[-1] = 1 + 2 * theta * sigma
        upper[0] = lower[-1] = -2 * theta * sigma
        rhs[0] = (
            u[0] + (1 - theta) * sigma * (2 * u[1] - 2 * u[0] - 2 * h * g0(t, a)) - 2 * theta * sigma * h * g0(t_new, a)
        )
        rhs[-1] = (
            u[-1]
            + (1 - theta) * sigma * (2 * u[-2] - 2 * u[-1] + 2 * h * g1(t, a))
            + 2 * theta * sigma * h * g1(t_new, a)
        )
    return _sweep(lower, diag, upper, rhs)


def _step_crank(u, bc, h, sigma, t, t_new, a):
    theta = 0.5
    n = len(u)
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    lower[1:-1] = -theta * sigma
    diag[1:-1] = 1 + 2 * theta * sigma
    upper[1:-1] = -theta * sigma
    rhs[1:-1] = u[1:-1] + (1 - theta) * sigma * (u[:-2] - 2 * u[1:-1] + u[2:])

    if bc == "2p1":
        diag[0], upper[0], rhs[0] = -1 / h, 1 / h, g0(t_new, a)
        lower[-1], diag[-1], rhs[-1] = -1 / h, 1 / h, g1(t_new, a)
    elif bc == "3p2":
        left_factor = (-1 / (2 * h)) / upper[1]
        diag[0] = -3 / (2 * h) - left_factor * lower[1]
        upper[0] = 2 / h - left_factor * diag[1]
        rhs[0] = g0(t_new, a) - left_factor * rhs[1]

        right_factor = (1 / (2 * h)) / lower[-2]
        lower[-1] = -2 / h - right_factor * diag[-2]
        diag[-1] = 3 / (2 * h) - right_factor * upper[-2]
        rhs[-1] = g1(t_new, a) - right_factor * rhs[-2]
    else:
        diag[0] = diag[-1] = 1 + 2 * theta * sigma
        upper[0] = lower[-1] = -2 * theta * sigma
        rhs[0] = (
            u[0] + (1 - theta) * sigma * (2 * u[1] - 2 * u[0] - 2 * h * g0(t, a)) - 2 * theta * sigma * h * g0(t_new, a)
        )
        rhs[-1] = (
            u[-1]
            + (1 - theta) * sigma * (2 * u[-2] - 2 * u[-1] + 2 * h * g1(t, a))
            + 2 * theta * sigma * h * g1(t_new, a)
        )
    return _sweep(lower, diag, upper, rhs)


def _prepare(bc, a, N, tau, T, t_out):
    if bc not in BOUNDARIES:
        raise ValueError("Неизвестная аппроксимация границы")
    if N < 3 or a <= 0 or tau <= 0 or T <= 0:
        raise ValueError("Требуется N >= 3 и положительные a, tau, T")
    steps = round(T / tau)
    if steps < 1 or not math.isclose(steps * tau, T, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("T должно быть кратно tau")
    targets = [T] if t_out is None else list(t_out)
    output_steps = {}
    for tt in targets:
        j = round(tt / tau)
        if j < 0 or j > steps or not math.isclose(j * tau, tt, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Время вывода {tt} должно быть узлом сетки [0, T]")
        output_steps.setdefault(j, []).append(tt)
    h = L / N
    sigma = a * a * tau / (h * h)
    xs = np.linspace(0, L, N + 1)
    u = np.sin(xs)
    saved = {tt: u.copy() for tt in output_steps.get(0, [])}
    return xs, u, saved, h, sigma, steps, output_steps


def solve_explicit(bc, a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    xs, u, saved, h, sigma, steps, output_steps = _prepare(bc, a, N, tau, T, t_out)
    if sigma > 0.5 + 1e-12:
        raise ValueError(f"Явная схема неустойчива: σ={sigma:.4g} > 1/2")
    for k in range(steps):
        t, t_new = k * tau, (k + 1) * tau
        u = _step_explicit(u, bc, h, sigma, t, t_new, a)
        for tt in output_steps.get(k + 1, []):
            saved[tt] = u.copy()
    return xs, saved


def solve_implicit(bc, a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    xs, u, saved, h, sigma, steps, output_steps = _prepare(bc, a, N, tau, T, t_out)
    for k in range(steps):
        t, t_new = k * tau, (k + 1) * tau
        u = _step_implicit(u, bc, h, sigma, t, t_new, a)
        for tt in output_steps.get(k + 1, []):
            saved[tt] = u.copy()
    return xs, saved


def solve_crank(bc, a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    xs, u, saved, h, sigma, steps, output_steps = _prepare(bc, a, N, tau, T, t_out)
    for k in range(steps):
        t, t_new = k * tau, (k + 1) * tau
        u = _step_crank(u, bc, h, sigma, t, t_new, a)
        for tt in output_steps.get(k + 1, []):
            saved[tt] = u.copy()
    return xs, saved


def _print_error_row(scheme, bc, xs, saved, t_out, a):
    errors = [max_error(saved[tt], xs, tt, a) for tt in t_out]
    print(f"{scheme:<10}{bc:<6}" + "".join(f"{err:<14.3e}" for err in errors))


def _one_error(xs, saved, at, a):
    return f"{max_error(saved[at], xs, at, a):.3e}"


def _print_grid_row(value, scheme, cells, width):
    print(f"{value:<{width}}{scheme:<10}" + "".join(f"{cell:<12}" for cell in cells))


def report_errors(a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    if t_out is None:
        t_out = [0.2, 0.5, 1.0]
    print(f"N={N}, h={L / N:.6g}, tau={tau}, T={T}, a={a}")
    print(f"{'схема':<10}{'ГУ':<6}" + "".join(f"t={tt:<12}" for tt in t_out))
    for bc in BOUNDARIES:
        xs, saved = solve_explicit(bc, a, N, tau, T, t_out)
        _print_error_row("explicit", bc, xs, saved, t_out, a)

        xs, saved = solve_implicit(bc, a, N, tau, T, t_out)
        _print_error_row("implicit", bc, xs, saved, t_out, a)

        xs, saved = solve_crank(bc, a, N, tau, T, t_out)
        _print_error_row("crank", bc, xs, saved, t_out, a)


def main():
    a = 1.0
    print("Макс ошибка по x:")
    report_errors(a, N=50, tau=0.001, T=1.0, t_out=[0.2, 0.5, 1.0])

    print("\nСход по h: tau=0.0001, T=0.5")
    print(f"{'N':<6}{'схема':<10}" + "".join(f"{bc:<12}" for bc in BOUNDARIES))
    for N in (10, 20, 40, 80, 160):
        explicit_errors = []
        implicit_errors = []
        crank_errors = []
        for bc in BOUNDARIES:
            xs, saved = solve_explicit(bc, a, N, 0.0001, 0.5, [0.5])
            explicit_errors.append(_one_error(xs, saved, 0.5, a))

            xs, saved = solve_implicit(bc, a, N, 0.0001, 0.5, [0.5])
            implicit_errors.append(_one_error(xs, saved, 0.5, a))

            xs, saved = solve_crank(bc, a, N, 0.0001, 0.5, [0.5])
            crank_errors.append(_one_error(xs, saved, 0.5, a))
        _print_grid_row(N, "explicit", explicit_errors, 6)
        _print_grid_row(N, "implicit", implicit_errors, 6)
        _print_grid_row(N, "crank", crank_errors, 6)

    print("\nСход по tau: N=50, T=0.5")
    print(f"{'tau':<10}{'схема':<10}" + "".join(f"{bc:<12}" for bc in BOUNDARIES))
    for tau in (0.01, 0.005, 0.002, 0.001, 0.0005):
        explicit_errors = []
        implicit_errors = []
        crank_errors = []
        for bc in BOUNDARIES:
            if a * a * tau / (L / 50) ** 2 <= 0.5 + 1e-12:
                xs, saved = solve_explicit(bc, a, 50, tau, 0.5, [0.5])
                explicit_errors.append(_one_error(xs, saved, 0.5, a))
            else:
                explicit_errors.append("неуст.")

            xs, saved = solve_implicit(bc, a, 50, tau, 0.5, [0.5])
            implicit_errors.append(_one_error(xs, saved, 0.5, a))

            xs, saved = solve_crank(bc, a, 50, tau, 0.5, [0.5])
            crank_errors.append(_one_error(xs, saved, 0.5, a))
        _print_grid_row(tau, "explicit", explicit_errors, 10)
        _print_grid_row(tau, "implicit", implicit_errors, 10)
        _print_grid_row(tau, "crank", crank_errors, 10)


if __name__ == "__main__":
    main()
