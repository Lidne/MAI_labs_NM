import math


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(m):
    return [list(row) for row in zip(*m)]


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


def offdiag_frobenius_norm(a):
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += a[i][j] * a[i][j]
    return math.sqrt(s)


def max_offdiag_index(a):
    n = len(a)
    p, q = 0, 1
    best = abs(a[p][q])
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(a[i][j])
            if val > best:
                best = val
                p, q = i, j
    return p, q


def jacobi_rotation_method(a, eps=1e-10, max_iter=10000):
    n = len(a)
    a_curr = [row[:] for row in a]
    v = identity(n)
    errors = [offdiag_frobenius_norm(a_curr)]

    for it in range(1, max_iter + 1):
        if errors[-1] < eps:
            break

        p, q = max_offdiag_index(a_curr)
        if abs(a_curr[p][q]) < 1e-15:
            break

        if abs(a_curr[p][p] - a_curr[q][q]) < 1e-15:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan2(2 * a_curr[p][q], a_curr[p][p] - a_curr[q][q])

        c = math.cos(phi)
        s = math.sin(phi)

        u = identity(n)
        u[p][p] = c
        u[q][q] = c
        u[p][q] = -s
        u[q][p] = s

        ut = transpose(u)
        a_curr = matmul(matmul(ut, a_curr), u)
        v = matmul(v, u)
        errors.append(offdiag_frobenius_norm(a_curr))

    eigenvalues = [a_curr[i][i] for i in range(n)]
    return eigenvalues, v, errors


def main():
    a = [
        [4.0, 7.0, -1.0],
        [7.0, 9.0, -6.0],
        [-1.0, -6.0, -4.0],
    ]
    eps = 1e-10

    eigenvalues, eigenvectors, errors = jacobi_rotation_method(a, eps=eps)

    print("Eigenvalues:")
    for i, val in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {val:.10f}")

    print("\nEigenvectors (columns):")
    for row in eigenvectors:
        print(" ".join(f"{x: .10f}" for x in row))

    print("\nError vs iterations (off-diagonal Frobenius norm):")
    for i, err in enumerate(errors):
        print(f"iter {i:4d}: {err:.12e}")


if __name__ == "__main__":
    main()
