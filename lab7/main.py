"""Уравнение Лапласа: метод Либмана, метод Зейделя и метод верхней релаксации."""

from pathlib import Path

import numpy as np


METHODS = {"liebmann": "Либман", "seidel": "Зейдель", "верхняя_релаксация": "Метод верхней релаксации (ω = 1.7)"}


def exact(x, y):
    return np.sin(x[:, None]) * np.exp(y[None, :])


def set_side_boundaries(u, y, hx):
    """Производные на вертикальных сторонах: трёхточечные формулы O(hx²)."""
    term = 2 * hx * np.exp(y[1:-1])
    u[0, 1:-1] = (4 * u[1, 1:-1] - u[2, 1:-1] - term) / 3
    u[-1, 1:-1] = (4 * u[-2, 1:-1] - u[-3, 1:-1] - term) / 3
    # Углы уже заданы условиями Дирихле при y = 0 и y = 1.


def solve(method, nx=40, ny=20, eps=1e-8, omega=1.7, max_iter=100_000):
    """Вернуть x, y, u[i, j] и историю максимальной невязки уравнения.

    nx, ny — числа интервалов; eps — допуск на невязку, а не на ошибку к U.
    Во всех методах решается одна и та же разностная система.
    """
    if method not in METHODS:
        raise ValueError("Метод должен быть liebmann, seidel или верхняя_релаксация")
    if nx < 3 or ny < 2 or not np.isfinite(eps) or eps <= 0 or max_iter < 1:
        raise ValueError("Нужны nx >= 3, ny >= 2, eps > 0 и max_iter >= 1")
    if not 0 < omega < 2:
        raise ValueError("Для релаксации нужно 0 < omega < 2")

    x = np.linspace(0, np.pi, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    hx, hy = np.pi / nx, 1 / ny
    ax, ay = 1 / hx**2, 1 / hy**2

    # Начальное приближение: прямые между заданными нижней и верхней сторонами.
    u = np.sin(x[:, None]) * (1 - y[None, :] + np.e * y[None, :])
    set_side_boundaries(u, y, hx)

    # Исключаем неизвестные боковых сторон из уравнений внутренних узлов.
    # После этого: d[i] * u[i,j] = left[i]*u[i-1,j] + right[i]*u[i+1,j]
    #                              + ay*(u[i,j-1] + u[i,j+1]) + b[i,j].
    left = np.full(nx - 1, ax)
    right = np.full(nx - 1, ax)
    d = np.full(nx - 1, 2 * (ax + ay))
    b = np.zeros((nx - 1, ny - 1))
    left[0], right[0] = 0, 2 * ax / 3
    left[-1], right[-1] = 2 * ax / 3, 0
    d[[0, -1]] -= 4 * ax / 3
    b[0, :] = b[-1, :] = -2 * np.exp(y[1:-1]) / (3 * hx)

    def rhs(values):
        return (
            left[:, None] * values[:-2, 1:-1]
            + right[:, None] * values[2:, 1:-1]
            + ay * (values[1:-1, :-2] + values[1:-1, 2:])
            + b
        )

    history = []
    weight = omega if method == "верхняя_релаксация" else 1.0
    for _ in range(max_iter):
        if method == "liebmann":
            # Правая часть целиком вычисляется до присваивания новых значений.
            u[1:-1, 1:-1] = rhs(u) / d[:, None]
        else:
            for i in range(1, nx):
                for j in range(1, ny):
                    value = (
                        left[i - 1] * u[i - 1, j]
                        + right[i - 1] * u[i + 1, j]
                        + ay * (u[i, j - 1] + u[i, j + 1])
                        + b[i - 1, j - 1]
                    ) / d[i - 1]
                    # Слева и снизу уже лежат значения текущего прохода.
                    u[i, j] += weight * (value - u[i, j])

        residual = float(np.max(np.abs(rhs(u) - d[:, None] * u[1:-1, 1:-1])))
        history.append(residual)
        if not np.isfinite(residual):
            raise RuntimeError("Итерации расходятся; попробуйте уменьшить omega")
        if residual <= eps:
            set_side_boundaries(u, y, hx)
            return x, y, u, np.array(history)

    raise RuntimeError(f"Метод {method} не сошёлся за {max_iter} итераций")


def grid_study():
    """Меняем hx, hy отдельно и вместе, сравнивая все три метода."""
    studies = {
        "Меняем hx, ny = 40": [(nx, 40) for nx in (10, 20, 40, 80)],
        "Меняем hy, nx = 80": [(80, ny) for ny in (5, 10, 20, 40)],
        "Уменьшаем оба шага вдвое": [(10, 5), (20, 10), (40, 20), (80, 40)],
    }
    results = {}
    for title, grids in studies.items():
        print(f"\n{title}")
        print(f"{'nx':>4} {'ny':>4} {'hx':>10} {'hy':>10} {'Метод':<40} {'Ошибка':>12} {'Итерации':>9}")
        for nx, ny in grids:
            for method, name in METHODS.items():
                key = (method, nx, ny)
                if key not in results:
                    x, y, u, history = solve(method, nx, ny)
                    results[key] = (float(np.max(np.abs(u - exact(x, y)))), len(history))
                error, iterations = results[key]
                print(f"{nx:4d} {ny:4d} {np.pi/nx:10.6f} {1/ny:10.6f} {name:<40} {error:12.5e} {iterations:9d}")
    return studies, results


def plot_results(solutions, studies, errors):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(exist_ok=True)
    x, y, _, _ = solutions["liebmann"]
    target = exact(x, y)

    values = [solutions[method][2] for method in METHODS]
    deviations = [np.abs(u - target) for u in values]
    # Общие пределы цвета позволяют сравнивать методы по одному оттенку.
    solution_min = min(u.min() for u in values)
    solution_max = max(u.max() for u in values)
    error_max = max(error.max() for error in deviations)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True, layout="constrained")
    for col, (name, u, error) in enumerate(zip(METHODS.values(), values, deviations)):
        solution_picture = axes[0, col].pcolormesh(
            x, y, u.T, shading="auto", cmap="viridis", vmin=solution_min, vmax=solution_max
        )
        error_picture = axes[1, col].pcolormesh(
            x, y, error.T, shading="auto", cmap="magma", vmin=0, vmax=error_max
        )
        axes[0, col].set_title(name, fontsize=11)
        axes[1, col].set_title(f"Максимальная ошибка: {error.max():.5e}", fontsize=11)
        for ax in axes[:, col]:
            ax.set(xlabel="x", ylabel="y")
    fig.colorbar(solution_picture, ax=axes[0, :], label="Решение u(x, y)")
    fig.colorbar(error_picture, ax=axes[1, :], label="Абсолютная ошибка |u − U|")
    fig.suptitle(f"Решение и абсолютная ошибка: сетка {len(x) - 1} × {len(y) - 1} интервалов")
    fig.savefig(output_dir / "solution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for level, ax in zip((0.5, 0.75), axes):
        j = int(np.argmin(np.abs(y - level)))
        ax.plot(x, target[:, j], "k-", label="Точное решение")
        for (method, name), marker in zip(METHODS.items(), ("o", "x", "+")):
            values = solutions[method][2]
            ax.plot(x, values[:, j], linestyle="--", marker=marker, markevery=4, label=name)
        ax.set(xlabel="x", ylabel="u(x, y)", title=f"Сечение y = {y[j]:g}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sections.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, name in METHODS.items():
        history = solutions[method][3]
        ax.semilogy(np.arange(1, len(history) + 1), history, label=name)
    ax.axhline(1e-8, color="black", linestyle=":", label="Допуск 10⁻⁸")
    ax.set(xlabel="Номер итерации", ylabel="Максимальная невязка", title="Сходимость на сетке 40 × 20 интервалов")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for index, (ax, (title, grids)) in enumerate(zip(axes, studies.items())):
        steps = [1 / ny if index == 1 else np.pi / nx for nx, ny in grids]
        for (method, name), marker, size, style in zip(
            METHODS.items(), ("o", "s", "+"), (10, 7, 6), ("-", "--", ":")
        ):
            values = [errors[method, nx, ny][0] for nx, ny in grids]
            ax.loglog(
                steps, values, marker=marker, markersize=size, markerfacecolor="none",
                linestyle=style, label=name,
            )
        if index == 2:
            nx, ny = grids[-1]
            reference = errors["верхняя_релаксация", nx, ny][0] * (np.array(steps) / steps[-1])**2
            ax.loglog(steps, reference, "k--", label="Ориентир O(h²)")
        ax.set(xlabel="hy" if index == 1 else "hx", ylabel="Максимальная ошибка", title=title)
        ax.set_xticks(steps, [f"{step:.3g}" for step in steps])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(which="both", alpha=0.3)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_dir / "grid_steps.png", dpi=160)
    plt.close(fig)
    print(f"\nГрафики сохранены в {output_dir}")


def main():
    print("Сетка: nx = 40, ny = 20; допуск на невязку: 1e-8")
    print(f"{'Метод':<40} {'Итерации':>9} {'Невязка':>12} {'Макс. ошибка':>14}")
    solutions = {}
    for method, name in METHODS.items():
        x, y, u, history = solve(method)
        solutions[method] = x, y, u, history
        error = np.max(np.abs(u - exact(x, y)))
        print(f"{name:<40} {len(history):9d} {history[-1]:12.5e} {error:14.5e}")
    studies, errors = grid_study()
    plot_results(solutions, studies, errors)


if __name__ == "__main__":
    main()
