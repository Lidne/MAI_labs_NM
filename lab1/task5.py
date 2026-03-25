import math


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a, b):
    n = len(a)
    p = len(b[0])
    kdim = len(b)
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(kdim):
            aik = a[i][k]
            for j in range(p):
                out[i][j] += aik * b[k][j]
    return out


def qr_decomposition_householder(a):
    n = len(a)
    r = [row[:] for row in a]
    q = identity(n)

    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x < 1e-15:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x[:]
        v[0] += sign * norm_x
        norm_v = math.sqrt(sum(val * val for val in v))
        if norm_v < 1e-15:
            continue
        v = [val / norm_v for val in v]

        h_small = [[0.0] * (n - k) for _ in range(n - k)]
        for i in range(n - k):
            for j in range(n - k):
                h_small[i][j] = (1.0 if i == j else 0.0) - 2.0 * v[i] * v[j]

        h = identity(n)
        for i in range(k, n):
            for j in range(k, n):
                h[i][j] = h_small[i - k][j - k]

        r = matmul(h, r)
        q = matmul(q, h)

    return q, r


def wilkinson_shift_2x2(a11, a12, a21, a22):
    delta = (a11 - a22) / 2.0
    radicand = delta * delta + a12 * a21
    if radicand <= 0:
        return a22
    sign = 1.0 if delta >= 0 else -1.0
    return a22 - sign * (a12 * a21) / (abs(delta) + math.sqrt(radicand))


def qr_eigenvalues(a, eps=1e-10, max_iter=10000):
    n = len(a)
    ak = [row[:] for row in a]
    m = n - 1
    iters = 0

    while m >= 0 and iters < max_iter:
        if m == 0:
            break

        if abs(ak[m][m - 1]) < eps:
            ak[m][m - 1] = 0.0
            m -= 1
            continue

        mu = wilkinson_shift_2x2(ak[m - 1][m - 1], ak[m - 1][m], ak[m][m - 1], ak[m][m])

        sub = [[ak[i][j] - (mu if i == j else 0.0) for j in range(m + 1)] for i in range(m + 1)]
        q, r = qr_decomposition_householder(sub)
        sub_next = matmul(r, q)
        for i in range(m + 1):
            sub_next[i][i] += mu

        for i in range(m + 1):
            for j in range(m + 1):
                ak[i][j] = sub_next[i][j]

        iters += 1

    eigenvalues = []
    i = 0
    while i < n:
        if i < n - 1 and abs(ak[i + 1][i]) > eps:
            a11, a12 = ak[i][i], ak[i][i + 1]
            a21, a22 = ak[i + 1][i], ak[i + 1][i + 1]
            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr * tr - 4.0 * det
            if disc >= 0:
                root = math.sqrt(disc)
                eigenvalues.append((tr + root) / 2.0)
                eigenvalues.append((tr - root) / 2.0)
            else:
                root = math.sqrt(-disc)
                eigenvalues.append(complex(tr / 2.0, root / 2.0))
                eigenvalues.append(complex(tr / 2.0, -root / 2.0))
            i += 2
        else:
            eigenvalues.append(ak[i][i])
            i += 1

    return eigenvalues, iters


def main():
    a = [
        [-5.0, 8.0, 4.0],
        [4.0, 2.0, 6.0],
        [-2.0, 5.0, -6.0],
    ]
    eps = 1e-10

    q, r = qr_decomposition_householder(a)
    eigenvalues, iters = qr_eigenvalues(a, eps=eps)

    print("QR-разложение:")
    print("Q =")
    for row in q:
        print(" ".join(f"{x: .10f}" for x in row))
    print("\nR =")
    for row in r:
        print(" ".join(f"{x: .10f}" for x in row))

    print("\nСобственные значения по QR-алгоритму:")
    for idx, val in enumerate(eigenvalues, start=1):
        print(f"λ{idx} = {val}")
    print(f"итерации = {iters}")


if __name__ == "__main__":
    main()
