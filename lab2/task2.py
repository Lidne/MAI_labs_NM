import math


def read_eps() -> float:
    raw = input("Enter precision epsilon (for example 1e-6): ").strip()
    if not raw:
        return 1e-6
    eps = float(raw)
    if eps <= 0:
        raise ValueError("Epsilon must be positive.")
    return eps


def simple_iteration(a: float, x1_0: float, x2_0: float, eps: float, max_iter: int = 500):
    history = []
    x1, x2 = x1_0, x2_0
    for k in range(1, max_iter + 1):
        x1_next = a + math.cos(x2)
        x2_next = a + math.sin(x1)
        err = max(abs(x1_next - x1), abs(x2_next - x2))
        f1 = x1_next - math.cos(x2_next) - a
        f2 = x2_next - math.sin(x1_next) - a
        history.append((k, x1_next, x2_next, err, max(abs(f1), abs(f2))))
        if err < eps:
            return (x1_next, x2_next), history
        x1, x2 = x1_next, x2_next
    raise RuntimeError("Simple iteration for the system did not converge within max_iter.")


def newton(a: float, x1_0: float, x2_0: float, eps: float, max_iter: int = 100):
    history = []
    x1, x2 = x1_0, x2_0
    for k in range(1, max_iter + 1):
        f1 = x1 - math.cos(x2) - a
        f2 = x2 - math.sin(x1) - a

        j11 = 1.0
        j12 = math.sin(x2)
        j21 = -math.cos(x1)
        j22 = 1.0

        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-14:
            raise RuntimeError("Jacobian determinant is too close to zero in Newton method.")

        b1 = -f1
        b2 = -f2
        dx1 = (b1 * j22 - j12 * b2) / det
        dx2 = (j11 * b2 - b1 * j21) / det

        x1_next = x1 + dx1
        x2_next = x2 + dx2
        err = max(abs(dx1), abs(dx2))
        f1_next = x1_next - math.cos(x2_next) - a
        f2_next = x2_next - math.sin(x1_next) - a
        history.append((k, x1_next, x2_next, err, max(abs(f1_next), abs(f2_next))))
        if err < eps:
            return (x1_next, x2_next), history
        x1, x2 = x1_next, x2_next
    raise RuntimeError("Newton method for the system did not converge within max_iter.")


def print_history(title: str, history):
    print(f"\n{title}")
    print("iter | x1_k              | x2_k              | error              | residual")
    for k, x1k, x2k, err, res in history:
        print(f"{k:4d} | {x1k:17.12f} | {x2k:17.12f} | {err:17.10e} | {res:10.3e}")


def main():
    eps = read_eps()
    a = 1.0
    x1_0, x2_0 = 1.5, 1.5

    print("\nSystem:")
    print("x1 - cos(x2) = a")
    print("x2 - sin(x1) = a")
    print(f"a = {a}")
    print(f"Initial approximation (graphical): x1_0 = {x1_0}, x2_0 = {x2_0}")
    print(f"Precision: eps = {eps}\n")

    (x1_it, x2_it), hist_it = simple_iteration(a, x1_0, x2_0, eps)
    (x1_n, x2_n), hist_n = newton(a, x1_0, x2_0, eps)

    print_history("Simple iteration method:", hist_it)
    r1_it = x1_it - math.cos(x2_it) - a
    r2_it = x2_it - math.sin(x1_it) - a
    print(f"Result: x1 = {x1_it:.12f}, x2 = {x2_it:.12f}, residual = {max(abs(r1_it), abs(r2_it)):.3e}")

    print_history("Newton method:", hist_n)
    r1_n = x1_n - math.cos(x2_n) - a
    r2_n = x2_n - math.sin(x1_n) - a
    print(f"Result: x1 = {x1_n:.12f}, x2 = {x2_n:.12f}, residual = {max(abs(r1_n), abs(r2_n)):.3e}")


if __name__ == "__main__":
    main()
