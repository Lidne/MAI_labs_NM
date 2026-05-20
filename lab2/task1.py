import math

ROOT_INTERVAL = (0.74, 0.80)
GRID_SIZE = 2000


def f(x: float) -> float:
    return math.sin(x) - 2 * x * x + 0.5


def df(x: float) -> float:
    return math.cos(x) - 4 * x


def d2f(x: float) -> float:
    return -math.sin(x) - 4


def phi(x: float) -> float:
    value = (math.sin(x) + 0.5) / 2
    if value < 0:
        raise ValueError("Iteration left the domain of sqrt in phi(x).")
    return math.sqrt(value)


def dphi(x: float) -> float:
    return math.cos(x) / (4 * phi(x))


def sample_interval(func, left: float, right: float, points: int = GRID_SIZE):
    step = (right - left) / points
    return [func(left + i * step) for i in range(points + 1)]


def prepare_iteration_data(left: float, right: float):
    phi_values = sample_interval(phi, left, right)
    dphi_values = sample_interval(lambda x: abs(dphi(x)), left, right)

    phi_min = min(phi_values)
    phi_max = max(phi_values)
    q = max(dphi_values)

    if phi_min < left or phi_max > right:
        raise RuntimeError("phi(x) does not map the localization interval into itself.")
    if q >= 1:
        raise RuntimeError("The simple iteration method is not guaranteed to converge on this interval.")

    return q, phi_min, phi_max


def choose_newton_start(left: float, right: float) -> float:
    for point in (right, left):
        if f(point) * d2f(point) > 0:
            return point
    raise RuntimeError("Failed to choose a Newton starting point that satisfies f(x0) * f''(x0) > 0.")


def simple_iteration(x0: float, eps: float, q: float, max_iter: int = 200):
    history = []
    x_prev = x0
    for k in range(1, max_iter + 1):
        x_next = phi(x_prev)
        step = abs(x_next - x_prev)
        estimate = q * step / (1 - q)
        history.append((k, x_next, step, estimate, abs(f(x_next))))
        if estimate <= eps:
            return x_next, history
        x_prev = x_next
    raise RuntimeError("Simple iteration did not converge within max_iter.")


def newton(x0: float, eps: float, max_iter: int = 100):
    history = []
    x = x0
    for k in range(1, max_iter + 1):
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise RuntimeError("Derivative is too close to zero in Newton method.")
        x_next = x - f(x) / dfx
        step = abs(x_next - x)
        history.append((k, x_next, step, abs(f(x_next))))
        if step < eps:
            return x_next, history
        x = x_next
    raise RuntimeError("Newton method did not converge within max_iter.")


def print_iteration_history(title: str, history):
    print(f"\n{title}")
    print("iter | x_k               | step               | a-posteriori err    | |f(x_k)|")
    for k, xk, step, estimate, residual in history:
        print(f"{k:4d} | {xk:17.12f} | {step:17.10e} | {estimate:17.10e} | {residual:10.3e}")


def print_newton_history(title: str, history):
    print(f"\n{title}")
    print("iter | x_k               | |x_k - x_{k-1}|    | |f(x_k)|")
    for k, xk, step, residual in history:
        print(f"{k:4d} | {xk:17.12f} | {step:17.10e} | {residual:10.3e}")


def main():
    eps = 1e-6
    left, right = ROOT_INTERVAL
    q, phi_min, phi_max = prepare_iteration_data(left, right)
    x0_iteration = (left + right) / 2
    x0_newton = choose_newton_start(left, right)

    print("Решаем: sin(x) - 2x^2 + 0.5 = 0")
    print(f"Локализация: [{left}, {right}]")
    print(f"Простая итерация: phi([{left}, {right}]) subset [{phi_min:.6f}, {phi_max:.6f}], q = {q:.6f}")
    print(f"Начальная аппроксимация для простой интерации: x0 = {x0_iteration}")
    print(f"Начальная аппроксимация для метода ньютона: x0 = {x0_newton}")
    print(f"Точность: eps = {eps}\n")

    root_it, hist_it = simple_iteration(x0_iteration, eps, q)
    root_newton, hist_newton = newton(x0_newton, eps)

    print_iteration_history("Простая итерация:", hist_it)
    print(f"Результат: x = {root_it:.12f}, f(x) = {f(root_it):.3e}")

    print_newton_history("Метод Ньютона:", hist_newton)
    print(f"Результат: x = {root_newton:.12f}, f(x) = {f(root_newton):.3e}")


if __name__ == "__main__":
    main()
