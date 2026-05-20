import math

X1_INTERVAL = (0.81, 0.86)
X2_INTERVAL = (1.72, 1.76)
GRID_SIZE = 2000


def read_eps() -> float:
    raw = input("Enter precision epsilon (for example 1e-6): ").strip()
    if not raw:
        return 1e-6
    eps = float(raw)
    if eps <= 0:
        raise ValueError("Epsilon must be positive.")
    return eps


def f1(x1: float, x2: float, a: float) -> float:
    return x1 - math.cos(x2) - a


def f2(x1: float, x2: float, a: float) -> float:
    return x2 - math.sin(x1) - a


def phi1(x1: float, x2: float, a: float) -> float:
    return a + math.cos(a + math.sin(x1))


def phi2(x1: float, x2: float, a: float) -> float:
    return a + math.sin(x1)


def sample_interval(func, left: float, right: float, points: int = GRID_SIZE):
    step = (right - left) / points
    return [func(left + i * step) for i in range(points + 1)]


def prepare_iteration_data(a: float, x1_interval, x2_interval):
    x1_left, x1_right = x1_interval
    x2_left, x2_right = x2_interval

    phi1_values = sample_interval(lambda x1: phi1(x1, 0.0, a), x1_left, x1_right)
    phi2_values = sample_interval(lambda x1: phi2(x1, 0.0, a), x1_left, x1_right)

    phi1_min = min(phi1_values)
    phi1_max = max(phi1_values)
    phi2_min = min(phi2_values)
    phi2_max = max(phi2_values)

    q_values = sample_interval(
        lambda x1: max(
            abs(-math.sin(a + math.sin(x1)) * math.cos(x1)),
            abs(math.cos(x1)),
        ),
        x1_left,
        x1_right,
    )
    q = max(q_values)

    if phi1_min < x1_left or phi1_max > x1_right or phi2_min < x2_left or phi2_max > x2_right:
        raise RuntimeError("phi(x) does not map the selected box into itself.")
    if q >= 1:
        raise RuntimeError("The simple iteration method is not guaranteed to converge in the selected box.")

    return q, (phi1_min, phi1_max), (phi2_min, phi2_max)


def simple_iteration(a: float, x1_0: float, x2_0: float, eps: float, q: float, max_iter: int = 500):
    history = []
    x1, x2 = x1_0, x2_0
    for k in range(1, max_iter + 1):
        x1_next = phi1(x1, x2, a)
        x2_next = phi2(x1, x2, a)
        step = max(abs(x1_next - x1), abs(x2_next - x2))
        estimate = q * step / (1 - q)
        residual = max(abs(f1(x1_next, x2_next, a)), abs(f2(x1_next, x2_next, a)))
        history.append((k, x1_next, x2_next, step, estimate, residual))
        if estimate <= eps:
            return (x1_next, x2_next), history
        x1, x2 = x1_next, x2_next
    raise RuntimeError("Simple iteration for the system did not converge within max_iter.")


def newton(a: float, x1_0: float, x2_0: float, eps: float, max_iter: int = 100):
    history = []
    x1, x2 = x1_0, x2_0
    for k in range(1, max_iter + 1):
        fx1 = f1(x1, x2, a)
        fx2 = f2(x1, x2, a)

        j11 = 1.0
        j12 = math.sin(x2)
        j21 = -math.cos(x1)
        j22 = 1.0

        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-14:
            raise RuntimeError("Jacobian determinant is too close to zero in Newton method.")

        b1 = -fx1
        b2 = -fx2
        dx1 = (b1 * j22 - j12 * b2) / det
        dx2 = (j11 * b2 - b1 * j21) / det

        x1_next = x1 + dx1
        x2_next = x2 + dx2
        step = max(abs(dx1), abs(dx2))
        residual = max(abs(f1(x1_next, x2_next, a)), abs(f2(x1_next, x2_next, a)))
        history.append((k, x1_next, x2_next, step, residual))
        if step < eps:
            return (x1_next, x2_next), history
        x1, x2 = x1_next, x2_next
    raise RuntimeError("Newton method for the system did not converge within max_iter.")


def print_iteration_history(title: str, history):
    print(f"\n{title}")
    print("iter | x1_k              | x2_k              | step               | a-posteriori err    | residual")
    for k, x1k, x2k, step, estimate, residual in history:
        print(
            f"{k:4d} | {x1k:17.12f} | {x2k:17.12f} | {step:17.10e} | {estimate:17.10e} | {residual:10.3e}"
        )


def print_newton_history(title: str, history):
    print(f"\n{title}")
    print("iter | x1_k              | x2_k              | error              | residual")
    for k, x1k, x2k, step, residual in history:
        print(f"{k:4d} | {x1k:17.12f} | {x2k:17.12f} | {step:17.10e} | {residual:10.3e}")


def main():
    eps = read_eps()
    a = 1.0
    x1_0, x2_0 = 0.83, 1.74
    q, phi1_range, phi2_range = prepare_iteration_data(a, X1_INTERVAL, X2_INTERVAL)

    print("\nSystem:")
    print("x1 - cos(x2) = a")
    print("x2 - sin(x1) = a")
    print(f"a = {a}")
    print(f"Graphical localization of the positive solution: x1 in {X1_INTERVAL}, x2 in {X2_INTERVAL}")
    print(
        "Simple iteration interval check: "
        f"phi1 subset [{phi1_range[0]:.6f}, {phi1_range[1]:.6f}], "
        f"phi2 subset [{phi2_range[0]:.6f}, {phi2_range[1]:.6f}], q = {q:.6f}"
    )
    print(f"Initial approximation (graphical): x1_0 = {x1_0}, x2_0 = {x2_0}")
    print(f"Precision: eps = {eps}\n")

    (x1_it, x2_it), hist_it = simple_iteration(a, x1_0, x2_0, eps, q)
    (x1_n, x2_n), hist_n = newton(a, x1_0, x2_0, eps)

    print_iteration_history("Simple iteration method:", hist_it)
    r1_it = f1(x1_it, x2_it, a)
    r2_it = f2(x1_it, x2_it, a)
    print(f"Result: x1 = {x1_it:.12f}, x2 = {x2_it:.12f}, residual = {max(abs(r1_it), abs(r2_it)):.3e}")

    print_newton_history("Newton method:", hist_n)
    r1_n = f1(x1_n, x2_n, a)
    r2_n = f2(x1_n, x2_n, a)
    print(f"Result: x1 = {x1_n:.12f}, x2 = {x2_n:.12f}, residual = {max(abs(r1_n), abs(r2_n)):.3e}")


if __name__ == "__main__":
    main()
