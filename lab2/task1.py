import math


def read_eps() -> float:
    raw = input("Enter precision epsilon (for example 1e-6): ").strip()
    if not raw:
        return 1e-6
    eps = float(raw)
    if eps <= 0:
        raise ValueError("Epsilon must be positive.")
    return eps


def f(x: float) -> float:
    return math.sin(x) - 2 * x * x + 0.5


def df(x: float) -> float:
    return math.cos(x) - 4 * x


def phi(x: float) -> float:
    value = (math.sin(x) + 0.5) / 2
    if value < 0:
        raise ValueError("Iteration left the domain of sqrt in phi(x).")
    return math.sqrt(value)


def simple_iteration(x0: float, eps: float, max_iter: int = 200):
    history = []
    x_prev = x0
    for k in range(1, max_iter + 1):
        x_next = phi(x_prev)
        err = abs(x_next - x_prev)
        history.append((k, x_next, err, abs(f(x_next))))
        if err < eps:
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
        err = abs(x_next - x)
        history.append((k, x_next, err, abs(f(x_next))))
        if err < eps:
            return x_next, history
        x = x_next
    raise RuntimeError("Newton method did not converge within max_iter.")


def print_history(title: str, history):
    print(f"\n{title}")
    print("iter | x_k               | |x_k - x_{k-1}|    | |f(x_k)|")
    for k, xk, err, res in history:
        print(f"{k:4d} | {xk:17.12f} | {err:17.10e} | {res:10.3e}")


def main():
    eps = read_eps()
    x0 = 0.6
    print("\nEquation: sin(x) - 2x^2 + 0.5 = 0")
    print(f"Initial approximation (graphical): x0 = {x0}")
    print(f"Precision: eps = {eps}\n")

    root_it, hist_it = simple_iteration(x0, eps)
    root_newton, hist_newton = newton(x0, eps)

    print_history("Simple iteration method:", hist_it)
    print(f"Result: x = {root_it:.12f}, f(x) = {f(root_it):.3e}")

    print_history("Newton method:", hist_newton)
    print(f"Result: x = {root_newton:.12f}, f(x) = {f(root_newton):.3e}")


if __name__ == "__main__":
    main()
