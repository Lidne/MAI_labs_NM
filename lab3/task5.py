def f(x):
    return x * x / (x * x + 16.0)


def rectangles_midpoint(a, b, h):
    n = int(round((b - a) / h))
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))


def trapezoid(a, b, h):
    n = int(round((b - a) / h))
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s


def simpson(a, b, h):
    n = int(round((b - a) / h))
    if n % 2 != 0:
        raise ValueError("Simpson method requires an even number of intervals.")
    s = f(a) + f(b)
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        s += coef * f(a + i * h)
    return h * s / 3.0


def runge_romberg(i_h1, i_h2, h1, h2, p):
    k = h1 / h2
    error = abs(i_h2 - i_h1) / (k**p - 1.0)
    improved = i_h2 + (i_h2 - i_h1) / (k**p - 1.0)
    return improved, error


def main():
    a, b = 0.0, 2.0
    h1, h2 = 0.5, 0.25

    rect_h1 = rectangles_midpoint(a, b, h1)
    rect_h2 = rectangles_midpoint(a, b, h2)
    trap_h1 = trapezoid(a, b, h1)
    trap_h2 = trapezoid(a, b, h2)
    simp_h1 = simpson(a, b, h1)
    simp_h2 = simpson(a, b, h2)

    rect_rr, rect_err = runge_romberg(rect_h1, rect_h2, h1, h2, p=2)
    trap_rr, trap_err = runge_romberg(trap_h1, trap_h2, h1, h2, p=2)
    simp_rr, simp_err = runge_romberg(simp_h1, simp_h2, h1, h2, p=4)

    print("Task 3.5: Numerical integration")
    print("Integral: int_0^2 x^2/(x^2+16) dx")

    print("\nMidpoint rectangles:")
    print(f"I(h=0.5)  = {rect_h1:.12f}")
    print(f"I(h=0.25) = {rect_h2:.12f}")
    print(f"Runge-Romberg improved = {rect_rr:.12f}")
    print(f"Estimated error        = {rect_err:.12e}")

    print("\nTrapezoid:")
    print(f"I(h=0.5)  = {trap_h1:.12f}")
    print(f"I(h=0.25) = {trap_h2:.12f}")
    print(f"Runge-Romberg improved = {trap_rr:.12f}")
    print(f"Estimated error        = {trap_err:.12e}")

    print("\nSimpson:")
    print(f"I(h=0.5)  = {simp_h1:.12f}")
    print(f"I(h=0.25) = {simp_h2:.12f}")
    print(f"Runge-Romberg improved = {simp_rr:.12f}")
    print(f"Estimated error        = {simp_err:.12e}")


if __name__ == "__main__":
    main()
