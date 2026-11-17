from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

a = 1.0
T = 0.5
NX = NY = 40
NT = 400
NT_SPACE = 1000
GRAPHS = Path(__file__).resolve().parent / "graphs"
GRAPHS.mkdir(exist_ok=True)


def exact(x, y, t):
    return np.cos(2 * x) * np.cosh(y) * np.exp(-3 * a * t)


def coefficients(n, r, neumann=False):
    lower = np.full(n, -r)
    upper = np.full(n, -r)
    diag = np.full(n, 1 + 2 * r)
    if neumann:
        lower[-1] = -2 * r / 3
        diag[-1] = 1 + 2 * r / 3
    for i in range(1, n):
        upper[i - 1] /= diag[i - 1]
        diag[i] -= lower[i] * upper[i - 1]
    return lower, diag, upper


def sweep(rhs, factors):
    lower, diag, upper = factors
    v = rhs.copy()
    v[0] /= diag[0]
    for i in range(1, len(diag)):
        v[i] = (v[i] - lower[i] * v[i - 1]) / diag[i]
    for i in range(len(diag) - 2, -1, -1):
        v[i] -= upper[i] * v[i + 1]
    return v


def solve(method, nx, ny, nt, history=False):
    x = np.linspace(0, np.pi / 4, nx + 1)
    y = np.linspace(0, np.log(2), ny + 1)
    hx, hy, tau = x[1], y[1], T / nt
    multiplier = 0.5 if method == "МПН" else 1.0
    rx = a * tau * multiplier / hx**2
    ry = a * tau * multiplier / hy**2
    fx = coefficients(nx - 1, rx)
    fy = coefficients(ny - 1, ry, neumann=True)
    u = exact(x[:, None], y[None, :], 0)
    errors = [0.0] if history else None

    for k in range(nt):
        tm = (k + 0.5) * tau
        tn = (k + 1) * tau

        v = np.empty_like(u)
        v[0, :] = np.cosh(y) * np.exp(-3 * a * tm)
        v[-1, :] = 0
        v[:, 0] = np.cos(2 * x) * np.exp(-3 * a * tm)

        rhs = u[1:-1, 1:-1].copy()
        if method == "МПН":
            rhs += ry * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
        rhs[0, :] += rx * v[0, 1:-1]
        v[1:-1, 1:-1] = sweep(rhs, fx)
        gm = 0.75 * np.cos(2 * x[1:-1]) * np.exp(-3 * a * tm)
        v[1:-1, -1] = (4 * v[1:-1, -2] - v[1:-1, -3] + 2 * hy * gm) / 3

        w = np.empty_like(u)
        w[0, :] = np.cosh(y) * np.exp(-3 * a * tn)
        w[-1, :] = 0
        w[:, 0] = np.cos(2 * x) * np.exp(-3 * a * tn)

        rhs = v[1:-1, 1:-1].copy()
        if method == "МПН":
            rhs += rx * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1])
        rhs[:, 0] += ry * w[1:-1, 0]
        gn = 0.75 * np.cos(2 * x[1:-1]) * np.exp(-3 * a * tn)
        rhs[:, -1] += 2 * ry * hy * gn / 3
        w[1:-1, 1:-1] = sweep(rhs.T, fy).T
        w[1:-1, -1] = (4 * w[1:-1, -2] - w[1:-1, -3] + 2 * hy * gn) / 3
        u = w

        if history:
            errors.append(np.max(np.abs(u - exact(x[:, None], y[None, :], tn))))

    if history:
        return x, y, u, np.array(errors)
    return np.max(np.abs(u - exact(x[:, None], y[None, :], T)))


methods = ("МПН", "МДШ")
results = {}

plt.figure(figsize=(8, 4.5))
for method in methods:
    x, y, u, errors = solve(method, NX, NY, NT, history=True)
    results[method] = (x, y, u, errors)
    plt.semilogy(np.linspace(0, T, NT + 1)[1:], errors[1:], label=method)
    print(f"\n{method}: hx={x[1]:.6f}, hy={y[1]:.6f}, tau={T / NT:.6f}")
    for t in (0.1, 0.3, T):
        print(f"  t={t:.2f}: max ошибка = {errors[round(t * NT / T)]:.8e}")
plt.title("Максимальная погрешность в зависимости от времени")
plt.xlabel("t")
plt.ylabel("max |u - U|")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(GRAPHS / "error_time.png", dpi=150)
plt.close()

for method in methods:
    x, y, u, _ = results[method]
    error = np.abs(u - exact(x[:, None], y[None, :], T))
    plt.figure(figsize=(7, 4.5))
    plt.pcolormesh(x, y, error.T, shading="auto", cmap="viridis")
    plt.colorbar(label="|u - U|")
    plt.title(f"{method}: распределение погрешности при t={T}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(GRAPHS / f"error_map_{'adi' if method == 'МПН' else 'fractional'}.png", dpi=150)
    plt.close()

experiments = (
    ("tau", [20, 40, 80, 160, 320]),
    ("hx", [8, 16, 32, 64]),
    ("hy", [8, 16, 32, 64]),
)

for parameter, steps in experiments:
    plt.figure(figsize=(8, 4.5))
    print(f"\nЗависимость ошибки от {parameter}:")
    for method in methods:
        values, errors = [], []
        for n in steps:
            if parameter == "tau":
                nx, ny, nt = 64, 64, n
                h = T / nt
            elif parameter == "hx":
                nx, ny, nt = n, 64, NT_SPACE
                h = np.pi / (4 * nx)
            else:
                nx, ny, nt = 64, n, NT_SPACE
                h = np.log(2) / ny
            err = solve(method, nx, ny, nt)
            values.append(h)
            errors.append(err)
            print(f"  {method}: {parameter}={h:.7f}, E={err:.8e}")
        plt.loglog(values, errors, "o-", label=method)
    plt.title(f"Зависимость максимальной погрешности от {parameter} (t={T})")
    plt.xlabel(parameter)
    plt.ylabel("max |u - U|")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS / f"error_{parameter}.png", dpi=150)
    plt.close()
