"""Численное решение u_tt = u_xx - 5u на отрезке [0, 1]."""

import math

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu


BOUNDARIES = ("2p1", "3p2", "2p2")


def exact(x, t):
    return np.exp(2 * x) * np.cos(t)


def error(u, x, t):
    return np.max(np.abs(u - exact(x, t)))


def explicit_step(current, previous, h, tau, boundary):
    next_layer = np.empty_like(current)
    next_layer[1:-1] = (
        2 * current[1:-1]
        - previous[1:-1]
        + (tau / h) ** 2 * (current[:-2] - 2 * current[1:-1] + current[2:])
        - 5 * tau**2 * current[1:-1]
    )

    if boundary == "2p1":
        next_layer[0] = next_layer[1] / (1 + 2 * h)
        next_layer[-1] = next_layer[-2] / (1 - 2 * h)
    elif boundary == "3p2":
        next_layer[0] = (4 * next_layer[1] - next_layer[2]) / (3 + 4 * h)
        next_layer[-1] = (4 * next_layer[-2] - next_layer[-3]) / (3 - 4 * h)
    else:
        # Уравнение на границе, полученное из двухточечной формулы O(h²).
        next_layer[0] = (
            2 * current[0] - previous[0]
            + tau**2 * (2 * (current[1] - current[0]) / h**2 - 4 * current[0] / h - 5 * current[0])
        )
        next_layer[-1] = (
            2 * current[-1] - previous[-1]
            + tau**2 * (2 * (current[-2] - current[-1]) / h**2 + 4 * current[-1] / h - 5 * current[-1])
        )
    return next_layer


def implicit_matrix(n, h, tau, boundary):
    matrix = lil_matrix((n + 1, n + 1))
    r = (tau / h) ** 2
    for i in range(1, n):
        matrix[i, i - 1] = -r
        matrix[i, i] = 1 + 2 * r + 5 * tau**2
        matrix[i, i + 1] = -r

    if boundary == "2p1":
        matrix[0, 0], matrix[0, 1] = -(1 + 2 * h), 1
        matrix[n, n - 1], matrix[n, n] = -1, 1 - 2 * h
    elif boundary == "3p2":
        matrix[0, 0], matrix[0, 1], matrix[0, 2] = -(3 + 4 * h), 4, -1
        matrix[n, n - 2], matrix[n, n - 1], matrix[n, n] = 1, -4, 3 - 4 * h
    else:
        matrix[0, 0] = 1 + tau**2 * (2 / h**2 + 4 / h + 5)
        matrix[0, 1] = -2 * r
        matrix[n, n - 1] = -2 * r
        matrix[n, n] = 1 + tau**2 * (2 / h**2 - 4 / h + 5)
    return splu(matrix.tocsc())


def solve(scheme, boundary, initial_order, n=40, steps=200, output_times=(0.25, 0.5, 1.0)):
    """Вернуть узлы x и решения в выбранные моменты времени (T = 1)."""
    if scheme not in ("explicit", "implicit") or boundary not in BOUNDARIES:
        raise ValueError("Неизвестная схема или аппроксимация границы")
    if initial_order not in (1, 2) or n < 3 or steps < 1:
        raise ValueError("Требуются n >= 3, steps >= 1 и порядок 1 или 2")

    h, tau = 1 / n, 1 / steps
    if scheme == "explicit" and tau > 2 / math.sqrt(4 / h**2 + 5):
        raise ValueError("Для явной схемы уменьшите шаг по времени: нарушено условие устойчивости")

    x = np.linspace(0, 1, n + 1)
    previous = np.exp(2 * x)  # u(x, 0)
    current = previous.copy()  # u_t(x, 0) = 0; первый порядок по tau
    if initial_order == 2:
        # u_tt(x, 0) = u_xx(x, 0) - 5u(x, 0) = -exp(2x).
        current *= 1 - tau**2 / 2

    output_steps = {round(t * steps): t for t in output_times}
    for k, t in output_steps.items():
        if not (0 <= t <= 1) or not math.isclose(k / steps, t, abs_tol=1e-12):
            raise ValueError("Время вывода должно лежать на сетке в [0, 1]")
    saved = {output_steps[0]: previous.copy()} if 0 in output_steps else {}
    if 1 in output_steps:
        saved[output_steps[1]] = current.copy()

    factor = implicit_matrix(n, h, tau, boundary) if scheme == "implicit" else None
    for k in range(2, steps + 1):
        if scheme == "explicit":
            next_layer = explicit_step(current, previous, h, tau, boundary)
        else:
            rhs = 2 * current - previous
            if boundary != "2p2":
                rhs[0] = rhs[-1] = 0
            next_layer = factor.solve(rhs)
        previous, current = current, next_layer
        if k in output_steps:
            saved[output_steps[k]] = current.copy()
    return x, saved


def print_errors():
    print("Максимальная ошибка по пространству; n=40, tau=0.005")
    print(f"{'Схема':<10} {'Граница':<8} {'Нач.':<5} {'t=0.25':>12} {'t=0.5':>12} {'t=1':>12}")
    for scheme in ("explicit", "implicit"):
        for boundary in BOUNDARIES:
            for order in (1, 2):
                x, saved = solve(scheme, boundary, order)
                values = [error(saved[t], x, t) for t in (0.25, 0.5, 1.0)]
                print(f"{scheme:<10} {boundary:<8} {order:<5} " + " ".join(f"{e:12.5e}" for e in values))

    print("\nЗависимость ошибки при t=1 от tau; n=20, второй порядок начального условия")
    print(f"{'Схема':<10} {'Граница':<8} " + " ".join(f"{1/s:>12.5f}" for s in (100, 200, 400, 800)))
    for scheme in ("explicit", "implicit"):
        for boundary in BOUNDARIES:
            values = []
            for steps in (100, 200, 400, 800):
                x, saved = solve(scheme, boundary, 2, n=20, steps=steps, output_times=(1.0,))
                values.append(error(saved[1.0], x, 1.0))
            print(f"{scheme:<10} {boundary:<8} " + " ".join(f"{e:12.5e}" for e in values))

    print("\nЗависимость ошибки при t=1 от h; tau=0.0025, второй порядок начального условия")
    print(f"{'Схема':<10} {'Граница':<8} " + " ".join(f"{1/n:>12.5f}" for n in (10, 20, 40, 80)))
    for scheme in ("explicit", "implicit"):
        for boundary in BOUNDARIES:
            values = []
            for n in (10, 20, 40, 80):
                x, saved = solve(scheme, boundary, 2, n=n, steps=400, output_times=(1.0,))
                values.append(error(saved[1.0], x, 1.0))
            print(f"{scheme:<10} {boundary:<8} " + " ".join(f"{e:12.5e}" for e in values))


if __name__ == "__main__":
    print_errors()
