# from xd_ogp.py
'''
load functions with
from helper_funcs.make_c_star import
    M_gauss, L_gauss, IM_gauss, IL_gauss, ILL_gauss
    make_c_star_matrix
    make_G
    make_beta

'''

# IMPORTS
import numpy as np
from numpy import sqrt, pi, exp
from scipy.special import erf
from scipy.linalg import cho_factor, cho_solve, cholesky, solve, solve_triangular



# COVARIANCE ANALYTICAL INTEGRALS 
def M_gauss(x, psi, sigma2=1.0):
    x = np.asarray(x)
    return (sqrt(pi) * sigma2 * psi / 2.0) * (erf((x + 1.0)/psi) - erf((x - 1.0)/psi))

def L_gauss(x, psi, sigma2=1.0):
    x = np.asarray(x)
    return (sigma2 * (psi**2) / 2.0) * (exp(-((x + 1.0)/psi)**2) - exp(-((x - 1.0)/psi)**2)) + x * M_gauss(x, psi, sigma2)

def IM_gauss(psi, sigma2=1.0):
    return 2.0 * sqrt(pi) * sigma2 * psi * erf(2.0/psi) - sigma2 * (psi**2) * (1.0 - exp(-4.0/(psi**2)))

def IL_gauss(psi, sigma2=1.0):
    return 0.0

def ILL_gauss(psi, sigma2=1.0):
    return (sigma2 * (psi**4) / 6.0) * (1.0 - exp(-4.0/(psi**2))) \
         - (sigma2 * (psi**2) / 3.0) * (3.0 - exp(-4.0/(psi**2))) \
         + (2.0 * sqrt(pi) * sigma2 * psi / 3.0) * erf(2.0/psi)


# MAKE DESIGN MATRIX G
def make_G(data, terms, one_based=True, return_names=False):
    # Creates n x p design matrix of the form

    X = np.asarray(data['X_scaled'])
    n, d = X.shape
    
    cols = []
    names = []

    def add_col(name, col):
            cols.append(col.reshape(n, 1))
            names.append(name)

    for t in terms:
        # Intercept
        if t is None:
            add_col('1', np.ones(n))
            continue

        # Single Effect
        if isinstance(t, (int, np.integer)):
            j = t - 1 if one_based else t
            if j < 0 or j >= d:
                raise ValueError(f"Index {t} out of range for d={d} (one_based={one_based}).")
            add_col(f'x{j+1}', X[:, j])
            continue

        # Interaction
        else:
            try:
                idxs = [(i - 1) if one_based else int(i) for i in t]
            except Exception as e:
                raise ValueError(f"Unrecognized term: {t}") from e

        if len(idxs) < 2:
            raise ValueError(f"Interactions must have length >= 2; got {t}")
        if any((i < 0 or i >= d) for i in idxs):
            raise ValueError(f"Some indices in {t} are out of range for d={d}.")


        prod = np.prod(X[:, idxs], axis=1)
        name = '*'.join([f'x{i+1}' for i in idxs])
        add_col(name, prod)

    G = np.hstack(cols) if cols else np.empty((n, 0))
    return (G, names) if return_names else G

# MAKE BETA
def make_beta(y, G, C_chol, lower=True):
    """
    \hat{\beta} = (G^T C_*^{-1} G)^{-1} G^T C_*^{-1} y
    See Eq (2.4)
    Parameters
    ----------
    y    : (n,)
    G    : (n,p)
    chol : Cholesky factor of C_*
    lower: whether `chol` is lower-triangular
    """

    if G.size == 0:  # no mean terms
        return np.zeros(0, dtype=float)

    Cinv = lambda v: cho_solve((C_chol, lower), v, check_finite=False)
    GT_Cinv_G = G.T @ Cinv(G)
    GT_Cinv_y = G.T @ Cinv(y)
    beta_hat  = solve(GT_Cinv_G, GT_Cinv_y, assume_a='sym')
    return beta_hat


# MAKE C STAR MATRIX
def make_c_star_matrix(X, Xp, psi, sigma2, terms, one_based=True):
    """
    Compute C*(X, X') with Eq (2.3) c* = c - h^T H \inv h' (matrix version)
    Inputs:
      X, Xp : (n,d), (m,d) in [-1,1]^d
      psi   : (d,) lengthscales
      sigma2: variance (scale)
      terms : list defining g(x) structure (same format as in make_G)
    """
    def _parse_terms_to_sets(terms, d, one_based=True):
        """
        Map each term to its index set J_i ⊆ {0,...,d-1}.
        None  -> empty set
        int j -> {j-1} (if one_based) or {j} (if not)
        list of ints -> set of indices
        """
        J_sets = []
        for t in terms:
            if t is None:
                J_sets.append(tuple())                  # intercept = empty set
                continue
            if isinstance(t, (int, np.integer)):        # single term = int
                j = t-1 if one_based else t
                if not (0 <= j < d):
                    raise ValueError(f'Index {t} out of range for d={d}')
                J_sets.append((j,))
                continue
            else:                                       # list = interaction
                idxs = [(i-1) if one_based else int(i) for i in t]

            if any(i < 0 or i >= d for i in idxs):
                raise ValueError(f'Indices {idxs} out of range for d={d}')
            if len(set(idxs)) != len(idxs):
                raise ValueError(f'Duplicate indices in interaction {t}')
            if len(idxs) < 2:
                raise ValueError(f'Interactions must have length ≥2; got {t}')
            J_sets.append(tuple(sorted(idxs)))
        return J_sets  # list of tuples
    
    def _base_se_kernel(X, Xp, psi, sigma2=1.0):
        """Separable squared-exp with lengthscales psi (shape: (d,))."""

        X = np.asarray(X)
        Xp = np.asarray(Xp)
        psi = np.asarray(psi)
        dif = (X[:, None, :] - Xp[None, :, :]) / psi  # (n, m, d)
        D2 = np.sum(dif * dif, axis=2)

        return sigma2 * np.exp(-D2)
    
    def _h_matrix(X, psi, sigma2, J_sets):
        """
        Build h(X) matrix (shape: n x p)
        Each column is h(x_i), i=1,...,n where is i the data index (x_i \in \R^d)
        See formulas (2.2) and (3.3)
        """
        X = np.asarray(X)
        n, d = X.shape
        p = len(J_sets)

        # Precompute M_j and L_j for all coords (shape: n×d)
        M_all = np.empty((n, d))
        L_all = np.empty((n, d))
        for j in range(d):
            M_all[:, j] = M_gauss(X[:, j], psi[j], sigma2=1.0)
            L_all[:, j] = L_gauss(X[:, j], psi[j], sigma2=1.0)

        Hcols = np.empty((n, p))
        all_idx = set(range(d))
        for i, Ji in enumerate(J_sets):
            Ji = set(Ji)
            notJ = list(all_idx - Ji)
            Ji   = list(Ji)
            col = np.ones(n)
            if notJ:
                col *= np.prod(M_all[:, notJ], axis=1)
            if Ji:
                col *= np.prod(L_all[:, Ji], axis=1)
            Hcols[:, i] = col
        return Hcols * sigma2  # n×p
    
    def _H_diag(psi, sigma2, J_sets):
        """
        Build H(X) matrix. See formula (2.2) & (3.3)
        """
        psi = np.asarray(psi)
        d = len(psi)
        IM = np.array([IM_gauss(psi[j], sigma2=1.0)  for j in range(d)])
        ILL= np.array([ILL_gauss(psi[j], sigma2=1.0) for j in range(d)])

        Hdiag = []
        all_idx = set(range(d))
        for Ji in J_sets:
            Ji = set(Ji)
            notJ = list(all_idx - Ji)
            Ji   = list(Ji)
            val = 1.0
            if notJ:
                val *= np.prod(IM[notJ])
            if Ji:
                val *= np.prod(ILL[Ji])
            Hdiag.append(val)
        Hdiag = np.asarray(Hdiag)
        if np.any(Hdiag <= 0):
            raise RuntimeError("Non-positive H diagonal (check psi/domain).")
        return Hdiag * sigma2  # shape (p,)
    
    
    X  = np.asarray(X)  
    Xp = np.asarray(Xp)
    d  = X.shape[1]
    J_sets = _parse_terms_to_sets(terms, d, one_based=one_based)

    # Base covariance
    C = _base_se_kernel(X, Xp, psi, sigma2=sigma2)

    if len(J_sets) == 0:
        return C 

    # h(X), h(X')
    hX  = _h_matrix(X,  psi, sigma2, J_sets)     # n×p
    hXp = _h_matrix(Xp, psi, sigma2, J_sets)     # m×p

    # H diagonal and H^{-1}
    Hdiag = _H_diag(psi, sigma2, J_sets)         # (p,)
    Hinvd = 1.0 / Hdiag                          # (p,)

    # c* = c - h(X) diag(H^{-1}) h(X')^T
    C_corr = (hX * Hinvd) @ hXp.T          # n×m
    C_star = C - C_corr
    return C_star