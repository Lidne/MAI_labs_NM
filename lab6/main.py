"""Численное решение u_tt = u_xx - 5u на отрезке [0, 1]."""

import math
from pathlib import Path

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


def plot_results():
    import matplotlib

    matplotlib.use("Agg")  # Сохраняем картинки без графического окна.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(exist_ok=True)
    boundary_names = {"2p1": "2 точки, O(h)", "3p2": "3 точки, O(h²)", "2p2": "2 точки, O(h²)"}
    colors = {"2p1": "tab:red", "3p2": "tab:orange", "2p2": "tab:blue"}
    scheme_names = {"explicit": "Явная схема", "implicit": "Неявная схема"}

    # Профиль решения в три момента: линии должны почти совпадать.
    x, explicit = solve("explicit", "2p2", 2, n=20, steps=100)
    _, implicit = solve("implicit", "2p2", 2, n=20, steps=100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, t in zip(axes, (0.25, 0.5, 1.0)):
        ax.plot(x, exact(x, t), color="black", label="Точное")
        ax.plot(x, explicit[t], "o", markevery=3, ms=4, label="Явная")
        ax.plot(x, implicit[t], "x", markevery=3, ms=4, label="Неявная")
        ax.set_title(f"t = {t:g}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend()
    fig.suptitle("Решение и точная функция: 2 точки O(h²), второй начальный слой")
    fig.tight_layout()
    fig.savefig(output_dir / "solutions.png", dpi=180)
    plt.close(fig)

    # Ошибка относительно точного решения на всём отрезке времени.
    times = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, boundary in zip(axes, BOUNDARIES):
        for scheme, marker in (("explicit", "o"), ("implicit", "x")):
            x, saved = solve(scheme, boundary, 2, output_times=times)
            errors = [error(saved[t], x, t) for t in times]
            ax.plot(times, errors, marker=marker, markevery=[25, 50, 100], label=scheme_names[scheme])
        ax.set_title(boundary_names[boundary])
        ax.set_xlabel("Время t")
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Максимальная ошибка")
    axes[0].legend()
    fig.suptitle("Ошибка относительно U(x, t) = exp(2x) cos(t); h = 0.025, τ = 0.005")
    fig.tight_layout()
    fig.savefig(output_dir / "error_over_time.png", dpi=180)
    plt.close(fig)

    # Как выбор границы и первого временного слоя влияет на ошибку.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    positions = np.arange(len(BOUNDARIES))
    for ax, scheme in zip(axes, scheme_names):
        for order, shift, label in ((1, -0.18, "1-й порядок"), (2, 0.18, "2-й порядок")):
            values = []
            for boundary in BOUNDARIES:
                x, saved = solve(scheme, boundary, order, output_times=(1.0,))
                values.append(error(saved[1.0], x, 1.0))
            ax.bar(positions + shift, values, width=0.34, label=label)
        ax.set_title(scheme_names[scheme])
        ax.set_xticks(positions, [boundary_names[b] for b in BOUNDARIES])
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Максимальная ошибка при t = 1 (лог. шкала)")
    axes[0].legend()
    fig.suptitle("Влияние граничного условия и начального слоя; h = 0.025, τ = 0.005")
    fig.tight_layout()
    fig.savefig(output_dir / "methods.png", dpi=180)
    plt.close(fig)

    # Изменяем только один шаг сетки за раз.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    steps_values = (100, 200, 400, 800)
    n_values = (10, 20, 40, 80)
    for row, scheme in enumerate(scheme_names):
        for boundary in BOUNDARIES:
            time_errors = []
            space_errors = []
            for steps in steps_values:
                x, saved = solve(scheme, boundary, 2, n=20, steps=steps, output_times=(1.0,))
                time_errors.append(error(saved[1.0], x, 1.0))
            for n in n_values:
                x, saved = solve(scheme, boundary, 2, n=n, steps=400, output_times=(1.0,))
                space_errors.append(error(saved[1.0], x, 1.0))
            axes[row, 0].loglog([1 / s for s in steps_values], time_errors, "o-", color=colors[boundary], label=boundary_names[boundary])
            axes[row, 1].loglog([1 / n for n in n_values], space_errors, "o-", color=colors[boundary], label=boundary_names[boundary])
        axes[row, 0].set_title(f"{scheme_names[scheme]}: меняем τ, h = 0.05")
        axes[row, 1].set_title(f"{scheme_names[scheme]}: меняем h, τ = 0.0025")
        for ax in axes[row]:
            ax.set_ylabel("Максимальная ошибка при t = 1")
            ax.grid(which="both", alpha=0.3)
        axes[row, 0].set_xticks([1 / s for s in steps_values], [f"{1/s:g}" for s in steps_values])
        axes[row, 1].set_xticks([1 / n for n in n_values], [f"{1/n:g}" for n in n_values])
        axes[row, 0].xaxis.set_minor_locator(NullLocator())
        axes[row, 1].xaxis.set_minor_locator(NullLocator())
    axes[1, 0].set_xlabel("Шаг по времени τ")
    axes[1, 1].set_xlabel("Шаг по пространству h")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Зависимость ошибки от шагов сетки; второй начальный слой")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_dir / "grid_steps.png", dpi=180)
    plt.close(fig)

    print(f"\nГрафики сохранены в {output_dir}")


if __name__ == "__main__":
    print_errors()
    plot_results()
