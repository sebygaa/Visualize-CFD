import numpy as np

try:
    from scipy.spatial import Delaunay
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def idw(X, Y, x, power=1.0, eps=1e-10):
    """
    Standard IDW:
        T(x) = sum(w_i T_i) / sum(w_i),   w_i = 1/(d_i+eps)^power
    Works for interpolation and extrapolation, but extrapolation is "range-bound"
    when all weights are positive.
    """
    X = np.asarray(X, dtype=float)          # shape (N, dim)
    Y = np.asarray(Y, dtype=float).reshape(-1)  # shape (N,)
    x = np.asarray(x, dtype=float).reshape(-1)  # shape (dim,)

    N, dim = X.shape
    if Y.shape[0] != N:
        raise ValueError(f"len(Y)={Y.shape[0]} must equal number of points N={N}")

    d = np.linalg.norm(X - x, axis=1)
    k = np.argmin(d)
    if d[k] <= eps:
        return float(Y[k])

    w = 1.0 / (d + eps) ** power
    return float((w @ Y) / np.sum(w))


def idw_affine(X, Y, x, power=1.0, eps=1e-10, ridge=0.0):
    """
    Affine/linear reproducing "IDW-like" estimator:

        lambda = W C^T (C W C^T)^(-1) c
        T(x)   = sum_i lambda_i T_i

    where:
      - W = diag(w_i),   w_i = 1/(d_i+eps)^power
      - C is (dim+1) x N : [1; x_i; y_i; z_i; ...]
      - c is (dim+1,)    : [1, x, y, z, ...]

    Properties:
      - If T is affine in space, it is reproduced EXACTLY (including extrapolation).
      - Weights lambda_i can be negative (especially in extrapolation).
      - It is NOT the same as standard IDW in general (even inside hull).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    N, dim = X.shape
    if Y.shape[0] != N:
        raise ValueError(f"len(Y)={Y.shape[0]} must equal number of points N={N}")
    if N < dim + 1:
        raise ValueError(f"Need at least dim+1={dim+1} points for affine reproduction; got N={N}")

    d = np.linalg.norm(X - x, axis=1)
    k = np.argmin(d)
    if d[k] <= eps:
        return float(Y[k])

    w = 1.0 / (d + eps) ** power  # positive base weights

    # Build C (dim+1, N) and c (dim+1,)
    C = np.vstack([np.ones(N), X.T])
    c = np.concatenate([[1.0], x])

    # Compute CWCT = C W C^T without forming full W:
    CWCT = (C * w) @ C.T  # (dim+1, dim+1)

    # Optional ridge for numerical stability if nearly singular
    if ridge != 0.0:
        CWCT = CWCT + ridge * np.eye(dim + 1)

    # Solve (C W C^T) alpha = c
    alpha = np.linalg.solve(CWCT, c)  # (dim+1,)

    # lambda = W C^T alpha   -> w * (C^T alpha)
    lam = w * (C.T @ alpha)  # (N,)

    return float(lam @ Y)


def _inside_convex_hull(X, x, tri=None):
    """
    Returns True if x is inside convex hull of X (scattered points).
    Requires SciPy.
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("SciPy not available; cannot do convex hull membership robustly.")
    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_points, ndim = X.shape
    if n_points < ndim + 1:
        # Not enough points to construct Delaunay triangulation
        return False
    if tri is None:
        tri = Delaunay(X)
    return tri.find_simplex(x) >= 0


def idw_auto(X, Y, x, power=1.0, eps=1e-12, ridge=1e-12, tri=None):
    """
    If x is inside convex hull -> use standard IDW (exactly matches your normal IDW).
    If x is outside convex hull -> use affine-reproducing method (extrapolation-friendly).

    This is the cleanest way to satisfy:
      - "Do not change interpolation results"
      - "Improve extrapolation behavior"
    """
    if _HAVE_SCIPY:
        inside = _inside_convex_hull(X, x, tri=tri)
        if inside:
            return idw(X, Y, x, power=power, eps=eps)
        else:
            return idw_affine(X, Y, x, power=power, eps=eps, ridge=ridge)
    else:
        # Fallback: no robust inside/outside test; choose one behavior.
        # Conservative choice: always use standard IDW.
        return idw(X, Y, x, power=power, eps=eps)


if __name__ == "__main__":
    # -------------------------
    # Test 1: 2D, 2 points (IDW interpolation)
    # -------------------------
    X_data = np.array([[1, 1], [2, 2]], dtype=float)
    Y_data = np.array([10, 20], dtype=float)
    x_targ = np.array([1.5, 1.5], dtype=float)
    print("IDW (2D, 2 points):", idw(X_data, Y_data, x_targ, power=1))

    # Note: affine method needs at least dim+1 points -> in 2D needs >=3 points, so we do NOT call idw_affine here.

    # -------------------------
    # Test 2: 3D cube corners
    # T = x (affine field), so affine method should extrapolate correctly
    # -------------------------
    X_data = np.array([
        [1,1,1],
        [1,1,2],
        [1,2,1],
        [2,1,1],
        [1,2,2],
        [2,1,2],
        [2,2,1],
        [2,2,2],
    ], dtype=float)

    # Correct length = 8
    Y_data2 = np.array([1, 1, 1, 2, 1, 2, 2, 2], dtype=float)

    x_in  = np.array([1.5, 1.2, 1.9], dtype=float)  # inside hull
    x_out = np.array([2.5, 1.2, 1.9], dtype=float)  # outside hull

    print("IDW inside:", idw(X_data, Y_data2, x_in, power=1))
    print("Affine inside:", idw_affine(X_data, Y_data2, x_in, power=1, ridge=1e-12))

    if _HAVE_SCIPY:
        tri = Delaunay(X_data)
        print("AUTO inside:", idw_auto(X_data, Y_data2, x_in, power=1, tri=tri))
        print("IDW outside:", idw(X_data, Y_data2, x_out, power=1))
        print("Affine outside:", idw_affine(X_data, Y_data2, x_out, power=1, ridge=1e-12))
        print("AUTO outside:", idw_auto(X_data, Y_data2, x_out, power=1, tri=tri))

