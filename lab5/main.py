import math

import numpy as np

L = math.pi


def exact(x, t, a):
    return math.exp(-a * t) * math.sin(x)


def g0(t, a):
    return math.exp(-a * t)


def g1(t, a):
    return -math.exp(-a * t)


def max_error(u, xs, t, a):
    errs = [abs(ui - exact(x, t, a)) for ui, x in zip(u, xs)]
    return max(errs)


def solve(scheme, bc, a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    """Решить задачу одной из схем.

    scheme: 'explicit' | 'implicit' | 'crank'.
    bc: '2p1' | '3p2' | '2p2'.
    Возвращает (xs, {t: u(t)}).
    """
    h = L / N
    M = max(1, int(round(T / tau)))
    tau = T / M
    xs = np.array([i * h for i in range(N + 1)], dtype=float)
    u = np.array([math.sin(x) for x in xs], dtype=float)

    r = a * tau / h**2

    if t_out is None:
        t_out = [T]
    targets = sorted(t_out)
    saved = {}
    if 0.0 in targets:
        saved[0.0] = u.copy()

    mat = None
    if scheme in ("implicit", "crank"):
        mat = np.zeros((N + 1, N + 1))
        if scheme == "implicit":
            for i in range(1, N):
                mat[i, i - 1] = -r
                mat[i, i] = 1.0 + 2.0 * r
                mat[i, i + 1] = -r
        else:
            for i in range(1, N):
                mat[i, i - 1] = -r / 2.0
                mat[i, i] = 1.0 + r
                mat[i, i + 1] = -r / 2.0
        _set_bc_rows(mat, scheme, bc, h, r)

    t = 0.0
    next_out = [tt for tt in targets if tt > 0.0]
    for _ in range(M):
        t_new = t + tau
        if scheme == "explicit":
            u = _step_explicit(u, bc, h, r, t, t_new, a)
        else:
            u = _step_implicit(u, mat, scheme, bc, h, r, t, t_new, a)
        t = t_new
        for tt in list(next_out):
            if t + 1e-12 >= tt:
                saved[tt] = u.copy()
                next_out.remove(tt)

    return xs, saved


def _set_bc_rows(mat, scheme, bc, h, r):
    N = mat.shape[0] - 1
    if bc == "2p1":
        # (u1-u0)/h = g0; (uN-u_{N-1})/h = g1
        mat[0, 0] = -1.0 / h
        mat[0, 1] = 1.0 / h
        mat[N, N - 1] = -1.0 / h
        mat[N, N] = 1.0 / h
    elif bc == "3p2":
        # (-3u0+4u1-u2)/(2h) = g0
        # (u_{N-2}-4u_{N-1}+3uN)/(2h) = g1
        mat[0, 0] = -3.0 / (2.0 * h)
        mat[0, 1] = 2.0 / h
        mat[0, 2] = -1.0 / (2.0 * h)
        mat[N, N - 2] = 1.0 / (2.0 * h)
        mat[N, N - 1] = -2.0 / h
        mat[N, N] = 3.0 / (2.0 * h)
    elif bc == "2p2":
        if scheme == "implicit":
            mat[0, 0] = 1.0 + 2.0 * r
            mat[0, 1] = -2.0 * r
            mat[N, N - 1] = -2.0 * r
            mat[N, N] = 1.0 + 2.0 * r
        else:  # crank
            mat[0, 0] = 1.0 + r
            mat[0, 1] = -r
            mat[N, N - 1] = -r
            mat[N, N] = 1.0 + r
    else:
        raise ValueError(f"unknown bc: {bc}")


def _step_explicit(u, bc, h, r, t, t_new, a):
    N = len(u) - 1
    v = np.zeros_like(u)
    for i in range(1, N):
        v[i] = r * u[i - 1] + (1.0 - 2.0 * r) * u[i] + r * u[i + 1]
    if bc == "2p1":
        v[0] = v[1] - h * g0(t_new, a)
        v[N] = v[N - 1] + h * g1(t_new, a)
    elif bc == "3p2":
        v[0] = (4.0 * v[1] - v[2] - 2.0 * h * g0(t_new, a)) / 3.0
        v[N] = (4.0 * v[N - 1] - v[N - 2] + 2.0 * h * g1(t_new, a)) / 3.0
    elif bc == "2p2":
        v[0] = (1.0 - 2.0 * r) * u[0] + 2.0 * r * u[1] - 2.0 * r * h * g0(t, a)
        v[N] = 2.0 * r * u[N - 1] + (1.0 - 2.0 * r) * u[N] + 2.0 * r * h * g1(t, a)
    else:
        raise ValueError(f"unknown bc: {bc}")
    return v


def _step_implicit(u, mat, scheme, bc, h, r, t, t_new, a):
    N = len(u) - 1
    rhs = np.zeros_like(u)
    if scheme == "implicit":
        rhs[1:N] = u[1:N]
        if bc == "2p2":
            rhs[0] = u[0] - 2.0 * r * h * g0(t_new, a)
            rhs[N] = u[N] + 2.0 * r * h * g1(t_new, a)
        else:
            rhs[0] = g0(t_new, a)
            rhs[N] = g1(t_new, a)
    else:  # crank
        for i in range(1, N):
            rhs[i] = r / 2.0 * u[i - 1] + (1.0 - r) * u[i] + r / 2.0 * u[i + 1]
        if bc == "2p1" or bc == "3p2":
            rhs[0] = g0(t_new, a)
            rhs[N] = g1(t_new, a)
        else:
            # Граничный узел с фиктивной точкой:
            # (1+r)u0-r*u1=(1-r)u0_old+r*u1_old-rh(g0_old+g0_new)
            rhs[0] = (1.0 - r) * u[0] + r * u[1] - r * h * (g0(t, a) + g0(t_new, a))
            rhs[N] = r * u[N - 1] + (1.0 - r) * u[N] + r * h * (g1(t, a) + g1(t_new, a))
    return np.linalg.solve(mat, rhs)


def report_errors(a=1.0, N=50, tau=0.001, T=1.0, t_out=None):
    if t_out is None:
        t_out = [0.2, 0.5, 1.0]
    schemes = ["explicit", "implicit", "crank"]
    bcs = ["2p1", "3p2", "2p2"]
    print(f"N={N} h={L / N:.5f} tau={tau} T={T}")
    header = f"{'схема':<10}{'гу':<5}" + "".join(f"t={t:<10}" for t in t_out)
    print(header)
    for sc in schemes:
        for bc in bcs:
            try:
                xs, saved = solve(sc, bc, a, N, tau, T, t_out)
                errs = [max_error(saved[t], xs, t, a) for t in t_out]
                row = f"{sc:<10}{bc:<5}" + "".join(f"{e:<10.3e}" for e in errs)
            except Exception as e:  # расходимость и т.п.
                row = f"{sc:<10}{bc:<5}fail"
            print(row)


def main():
    a = 1.0

    print("max-ошибка:")
    report_errors(a, N=50, tau=0.001, T=1.0, t_out=[0.2, 0.5, 1.0])

    print("\nh: tau=0.0001 T=0.5")
    print(f"{'N':<6}{'схема':<10}{'2p1':<12}{'3p2':<12}{'2p2':<12}")
    for N in [10, 20, 40, 80, 160]:
        for sc in ["explicit", "implicit", "crank"]:
            errs = []
            for bc in ["2p1", "3p2", "2p2"]:
                try:
                    xs, saved = solve(sc, bc, a, N, 0.0001, 0.5, [0.5])
                    errs.append(f"{max_error(saved[0.5], xs, 0.5, a):.3e}")
                except Exception:
                    errs.append("fail")
            print(f"{N:<6}{sc:<10}{errs[0]:<12}{errs[1]:<12}{errs[2]:<12}")

    print("\ntau: N=50 T=0.5")
    print(f"{'tau':<10}{'схема':<10}{'2p1':<12}{'3p2':<12}{'2p2':<12}")
    for tau in [0.01, 0.005, 0.002, 0.001, 0.0005]:
        for sc in ["explicit", "implicit", "crank"]:
            errs = []
            for bc in ["2p1", "3p2", "2p2"]:
                try:
                    xs, saved = solve(sc, bc, a, 50, tau, 0.5, [0.5])
                    e = max_error(saved[0.5], xs, 0.5, a)
                    errs.append(f"{e:.3e}" if e < 1e6 else "fail")
                except Exception:
                    errs.append("fail")
            print(f"{tau:<10}{sc:<10}{errs[0]:<12}{errs[1]:<12}{errs[2]:<12}")


if __name__ == "__main__":
    main()
