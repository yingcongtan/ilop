# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


# Homogeneous Interior-Point algorithm, Batch Version.

# This code is based directly on the SciPy implementation of the MOSEK interior point
# method, which implements the homogeneous algorithm (primal-dual algorithm applied to
# a homogeneous version of the LP). The original SciPy implementation can be found here:
# https://github.com/scipy/scipy/blob/master/scipy/optimize/_linprog_ip.py

import torch
import numpy as np
from warnings import warn
from torch import bmm, baddbmm
from util import norm, T, sym_cond
from util import _baddbmm, _bsubbmm, _addcdiv, _subcdiv, _addcmul, _subcmul

NINF = -float('inf')
MAX_COND = {
    torch.float32: 1/np.finfo(np.float32).eps,  # inverse of float32 epsilon
    torch.float64: 1/np.finfo(np.float64).eps,  # inverse of float64 epsilon
}

# Function decorator that returns None if the argument is None, otherwise
# actually calls the function.
def pass_none(func):
    def wrapper(arg):
        return func(arg) if arg is not None else None
    return wrapper

# Raised when either the primal or dual is detected to be infeasible
class InfeasibleOrUnboundedError(RuntimeError):
    pass

# Emitted when a high condition number is detected when solving a system of equations
class LinAlgWarning(RuntimeWarning):
    pass

class SolveSymPos(torch.autograd.Function):
    """
    Solves A @ x = b where A is symmetric positive definite.

    The function is as numerically stable as torch.solve(b, A) but is
    faster in two situations:
    
    1. When a sequence b needs to be solved for, this function supports
       pre-factoring the A matrix so that each individual solve is faster.
       Solving three systems this way is about 1.3x faster than solving three
       systems from scratch with torch.solve.
       
    2. When gradients of b and/or A are needed, this function will only
       solve one system per backward (torch.solve's backward solves the
       same system involving A.t() twice). When this function does solve
       during backwards, it re-uses the LU factorization of A from the
       forward pass since A.t() = A. Solving a sequence of three interdependent
       systems and then calling backward on the output is about 3x faster
       with this function than torch.solve.
    """
    
    @staticmethod
    def forward(ctx, b, A, A_LU=None, A_pivots=None):
        if A_LU is None:
            if A_pivots is not None:
                raise ValueError("A_LU must be specified with A_pivots")
            A_LU, A_pivots = torch.lu(A)
        elif A_pivots is None:
            raise ValueError("A_pivots must be specified with A_LU")
        x = b.lu_solve(A_LU, A_pivots)
        ctx.save_for_backward(x, A_LU, A_pivots)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, A_LU, A_pivots = ctx.saved_tensors
        need_b, need_A, *_ = ctx.needs_input_grad
        grad_b = grad_x.lu_solve(A_LU, A_pivots)  # Take advantage of A.t() == A to re-use A_LU and A_pivots
        grad_A = grad_b.bmm(-T(x)) if need_A else None
        return grad_b, grad_A, None, None
    
# Due to limitation in PyTorch 1.4, custom autograd.Function can't be scripted
# (see https://github.com/pytorch/pytorch/issues/22329)
solve_sym_pos = SolveSymPos.apply

# Due to bug isinf can't be scripted (see https://github.com/pytorch/pytorch/issues/31021)
_isinf = torch.jit.trace(torch.isinf, torch.ones((1, 1, 1))) if torch.__version__ == '1.4.0' else torch.isinf

# Due to bug torch.lu can't be scripted (see https://github.com/pytorch/pytorch/issues/33381) 
_lu = torch.jit.trace(torch.lu, torch.ones((3, 2, 2)))

class GetStepSelect(torch.autograd.Function):
    """
    This custom gradient computation is necessary to avoid introducing nan into the
    gradient positions that correspond to d_v == 0.
    
    Specifically the problem happens as follows:
    1) Where d_v == 0 the value of v / d_v will be inf. This *seems* benign
       because masked_fill(d_v >= 0, const) will not propagate those infs.
    2) However, masked_fill backward will (correctly) propagate a gradient
       of 0 to all positions for which d_v >= 0. This means the upstream div
       operation (v / d_v) receives a "gradient at output" of 0.0 at these
       positions.
    3) The div backward then calculates the following expression:
           grad_at_denominator = grad_at_output / denominator
       which introduces 0.0 / 0.0 = nan at the positions with d_v == 0, despite
       the fact that they did not contribute to the output.
       
    To correct for this, the forward still applies masked_fill last, but
    the backward apples the mask as if it came first in the forward.
    This could also be achieved by careful use of torch.where with specific
    constants, and no custom gradient, but this solution seemed simple and
    allows avoiding torch.where (slower than masked_fill)
    """
    @staticmethod
    def forward(ctx, v, d_v, const: float):
        i = d_v >= 0
        z = v.div(d_v).masked_fill_(i, const)
        ctx.save_for_backward(d_v, i, z)
        return z
    
    @staticmethod
    def backward(ctx, grad_z):
        # where d_v >= 0:
        #   z = const
        #   grad_v = 0
        #   grad_d_v = 0
        # where d_v < 0:
        #   z = v / d_v
        #   grad_v = grad_z / d_v
        #   grad_d_v = -grad_z * v / (d_v * d_v) = -grad_v * z
        d_v, i, z = ctx.saved_tensors
        grad_z_over_d_v = grad_z.div(d_v)
        grad_v   = grad_z_over_d_v.masked_fill_(i, 0)               if ctx.needs_input_grad[0] else None
        grad_d_v = grad_z_over_d_v.mul(z).neg_().masked_fill_(i, 0) if ctx.needs_input_grad[1] else None
        return grad_v, grad_d_v, None

_get_step_select = GetStepSelect.apply

# WARNING: Do NOT script _get_step_scale (or any function that calls it)
#          until PyTorch 1.5 is released. PyTorch 1.4 has a TorchScript bug
#          that causes NaNs in grad where should be zeros.
#@torch.jit.script
def _get_step_scale(v, d_v):
    """Returns a (k,1,1)-tensor of k scaling factors for (k,n,1)-tensors v (variables) and d_v (search directions)."""
    alpha = torch.max(_get_step_select(v, d_v, NINF), dim=1, keepdim=True).values.neg_()
    alpha.masked_fill_(_isinf(alpha), 1) # If all d_z[j,:] >= 0 for a particular j, set scale[j] = 1.0
    return alpha

def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0: float):
    """
    An implementation of [4] equation 8.21.
    alpha0 is a scalar, such as 1.0 or 0.99995    
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction    
    alpha = _get_step_scale(x, d_x).mul_(alpha0).clamp_(max=1)
    alpha = torch.min(alpha, _get_step_scale(z, d_z).mul_(alpha0))
    alpha = torch.min(alpha, _get_step_select(tau, d_tau, -1).mul(-alpha0))      # mul_ would cause autograd anomaly
    alpha = torch.min(alpha, _get_step_select(kappa, d_kappa, -1).mul(-alpha0))  # mul_ would cause autograd anomaly
    return alpha

def _get_step_max(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa):
    """
    Same as _get_step but constant-propagate max possible step size alpha0=1.
    """
    alpha = _get_step_scale(x, d_x).clamp_(max=1)
    alpha = torch.min(alpha, _get_step_scale(z, d_z))
    alpha = torch.min(alpha, _get_step_select(-tau, d_tau, 1))
    alpha = torch.min(alpha, _get_step_select(-kappa, d_kappa, 1))
    return alpha

def _solve(b, M, M_LU, M_pivots, cholesky: bool):
    k, m, _ = b.shape
    if m == 0:
        return torch.zeros_like(b)
    if cholesky:
        return torch.cholesky_solve(b, M)
    return solve_sym_pos(b, M, M_LU, M_pivots)

def _detect_badcond(M, idx):
    with torch.no_grad():
        # Mask identifying batch items comprising only finite elements.
        # This check is necessary because it's possible for M to contain inf/nan when
        # things really break down, and sym_cond would raise an exception in that case.
        finite_mask = torch.isfinite(M).view(len(M), -1).all(dim=1)
        all_finite = finite_mask.all()
        if all_finite:
            badcond = sym_cond(M) > MAX_COND[M.dtype]
            
            badcond_finite = badcond
        else:
            bad_idx = idx[finite_mask.logical_not()]
            warn('Non-finite matrix for batch indices %s; result may not be accurate.' % \
                 (bad_idx.numpy()), LinAlgWarning, stacklevel=2)
            badcond = torch.ones_like(finite_mask)  # Set non-finite indices to "badcond=True"
            badcond[finite_mask] = sym_cond(M[finite_mask]) > MAX_COND[M.dtype]
            badcond_finite = badcond & finite_mask
        if badcond_finite.any():
            bad_idx = idx[badcond_finite]
    return badcond.unsqueeze_(1).unsqueeze_(2)

# Can't script _get_delta until PyTorch supports custom autograd functions
# (such as solve_pos_sym) in TorchScript.
# See https://github.com/pytorch/pytorch/issues/22329
# @torch.jit.script
def _get_delta(A, b, c,
        x, y, z, tau, kappa,
        r_P, r_D, r_G, mu,
        AT, bT, cT, idx,
        cholesky: bool,
        check_cond: bool):
    """
    Given standard form problem defined by ``A``, ``b``, and ``c``;
    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;
    algorithmic parameter ``gamma;
    and option ``pc`` (predictor-corrector),
    get the search direction for increments to the variable estimates.
    Parameters
    ----------
    As defined in [4], except:
    pc : bool
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial. Even though it requires
        the solution of an additional linear system, the factorization
        is typically (implicitly) reused so solution is efficient, and the
        number of algorithm iterations is typically reduced.
    Returns
    -------
    Search directions as defined in [4]
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    #  Assemble M from [4] Equation 8.31
    Dinv = x / z
    M = bmm(A, Dinv * AT)

    # Check that M is safely invertible, if needed. If empty constraints, skip the check.
    badcond = _detect_badcond(M, idx) if check_cond and M.shape[1] > 0 else None

    # Factorize M since predictor-corrector solves three systems with it
    if cholesky:
        M = torch.cholesky(M)
        M_LU = M_pivots = M  # Assign M rather than None as dummy Tensor value for sake of JIT
    else:
        M_LU, M_pivots = _lu(M.detach())  # Detach because used for custom gradient on M
    
    # >>> "predictor-corrector" [4] Section 4.1 <<<
    # >>> Initial check of predictor-corrector, with gamma=0 hard-coded <<<
    q = _solve(_baddbmm(b, A, Dinv * c), M, M_LU, M_pivots, cholesky)
    p = _bsubbmm(c, AT, q).mul_(Dinv)
    g = r_D + z
    v = _solve(_baddbmm(r_P, A, Dinv * g), M, M_LU, M_pivots, cholesky)
    u = _bsubbmm(g, AT, v).mul_(Dinv)
    d_tau_denom = _baddbmm(kappa / tau, bT, q).sub_(bmm(cT, p))
    d_tau = _baddbmm(r_G - kappa, cT, u).sub_(bmm(bT, v)).div_(d_tau_denom)
    d_x = _addcmul(u, p, d_tau)
    d_y = _addcmul(v, q, d_tau)
    d_z = _addcdiv(z, z*d_x, x).neg_()
    d_kappa = _addcdiv(kappa, kappa*d_tau, tau).neg_()
    
    # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23    
    alpha = _get_step_max(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa)
    
    # predictor-corrector, [4] definition after 8.12
    # [4] pg. 220 (Table 8.1)    
    beta1 = 0.1
    one_minus_alpha = (1 - alpha)
    gamma = (one_minus_alpha**2).mul_(one_minus_alpha.clamp(max=beta1))
    one_minus_gamma = (1 - gamma)
    
    # >>> Second step of preditor-corrector, with gamma derived from first step <<<
    # Reference [4] Eq. 8.6
    rhatp = r_P.mul_(one_minus_gamma)
    rhatd = r_D.mul_(one_minus_gamma)
    rhatg = r_G.mul_(one_minus_gamma)

    # Reference [4] Eq. 8.7 and 8.13
    gamma_mu = gamma.mul_(mu)
    rhatxs = gamma_mu - _addcmul(x*z, d_x, d_z)
    rhattk = gamma_mu - _addcmul(tau*kappa, d_tau, d_kappa)

    # [4] Equations 8.28 and 8.29
    g = _subcdiv(rhatd, rhatxs, x)
    v = _solve(_baddbmm(rhatp, A, Dinv * g), M, M_LU, M_pivots, cholesky)
    u = _bsubbmm(g, AT, v).mul_(Dinv)
    
    # [4] Results after 8.29
    d_tau = _baddbmm(_addcdiv(rhatg, rhattk, tau), cT, u).sub_(bmm(bT, v)).div_(d_tau_denom)
    d_x = _addcmul(u, p, d_tau)
    d_y = _addcmul(v, q, d_tau)

    # [4] Relations between  after 8.25 and 8.26
    d_z     = _subcmul(rhatxs, z, d_x).div_(x)
    d_kappa = _subcmul(rhattk, kappa, d_tau).div_(tau)

    return d_x, d_y, d_z, d_tau, d_kappa, badcond

def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    """
    An implementation of [4] Equation 8.9
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    x = _addcmul(x, alpha, d_x)
    z = _addcmul(z, alpha, d_z)
    y = _addcmul(y, alpha, d_y)
    tau = _addcmul(tau, alpha, d_tau)
    kappa = _addcmul(kappa, alpha, d_kappa)
    return x, y, z, tau, kappa

@torch.jit.script
def _get_blind_start(A):
    """
    Return the starting point from [4] 4.4
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    k, m, n = A.shape
    x0     = torch.ones( (k, n, 1), dtype=A.dtype, device=A.device)
    y0     = torch.zeros((k, m, 1), dtype=A.dtype, device=A.device)
    z0     = torch.ones( (k, n, 1), dtype=A.dtype, device=A.device)
    tau0   = torch.ones( (k, 1, 1), dtype=A.dtype, device=A.device)
    kappa0 = torch.ones( (k, 1, 1), dtype=A.dtype, device=A.device)
    return x0, y0, z0, tau0, kappa0

def _indicators(A, b, c, x, y, z, tau, kappa, AT, bT, cT, denom_p0, denom_d0, denom_g0):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """

    # residuals for termination are relative to initial values
    k, m, n = A.shape

    # [4] Equation 8.8
    r_P = baddbmm(b * tau, A, x, alpha=-1.0)
    r_D = _addcmul(_baddbmm(z, AT, y).neg_(), c, tau)
    cTx = bmm(cT, x)
    bTy = bmm(bT, y)
    cTx_bTy = cTx - bTy
    r_G = kappa + cTx_bTy

    # See [4], Section 4.5 - The Stopping Criteria
    rho_A = torch.abs(cTx_bTy).div_((torch.abs(bTy).add_(tau)))
    rho_p = norm(r_P).div_(denom_p0)
    rho_d = norm(r_D).div_(denom_d0)
    rho_g = torch.abs(r_G).div_(denom_g0)
    rho_mu = baddbmm(tau * kappa, T(x), z, beta=1/(n+1), alpha=1/(n+1))

    return rho_p, rho_d, rho_A, rho_g, rho_mu, r_P, r_D, r_G, rho_mu

def _get_message(status):  # TODO: currently not used
    """
    Given problem status code, return a more detailed message.
    Parameters
    ----------
    status : int
        An integer representing the exit status of the optimization::
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    Returns
    -------
    message : str
        A string descriptor of the exit status of the optimization.
    """
    messages = (
        ["Optimization terminated successfully.",
         "The iteration limit was reached before the algorithm converged.",
         "The algorithm terminated successfully and determined that the "
         "problem is infeasible.",
         "The algorithm terminated successfully and determined that the "
         "problem is unbounded.",
         "Numerical difficulties were encountered before the problem "
         "converged. Please check your problem formulation for errors, "
         "independence of linear equality constraints, and reasonable "
         "scaling and matrix condition numbers. If you continue to "
         "encounter this error, please submit a bug report."
         ])
    return messages[status]

# Disable TorchScript for _ip_hsd until PyTorch 1.5 released.
# Both a bug in lack of  solve_sym_pos for details.
#@torch.jit.script
def _ip_hsd(c, A, b, alpha0: float, beta: float, maxiter: int, tol: float, check_cond: bool,
             cholesky: bool, want_pdgap: bool, want_grad: bool, numiter: int):
    r"""
    Solve a linear programming problem in standard form:
    Minimize::
        c @ x
    Subject to::
        A @ x == b
            x >= 0
    using the interior point method of [4].
    Parameters
    ----------
    c : (k,n,1)-tensor (float)
        Vectors defining each of the k linear objective functions to be minimized.
    A : (k,m,n)-tensor (float)
        Matrices such that ``A @ x``, gives the values of the equality
        constraints at ``x`` for each of the k problem instances.
    b : (k,m,1)-tensor (float)
        Vectors representing the RHS of each set of equality constraints.
    alpha0 : float
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_3`of [4] Table 8.1
    beta : float
        The desired reduction of the path parameter :math:`\mu` (see  [6]_)
    maxiter : int
        The maximum number of iterations of the algorithm.
    tol : float
        Termination tolerance; see [4]_ Section 4.5.
    cholesky : bool
        Whether to use cholesky factorization for linear system solves.
        Cholesky factorization can be a little faster but is less numerically
        stable for strict tolerances, and can blow up easily.
        The alternative is LU decomposition, which is more numerically stable.

    Returns
    -------
    x_hat : (k,n)-tensor (float)
        The k solution vectors for the given n-dimensional problem instances.
    status : k-tensor (int)
        An integer representing the exit status of each problem instance:
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    niter : k-tensor (int)
        The number of iterations taken to solve each problem instance
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    """
    k, m, n = A.shape
    dtype = c.dtype
    device = c.device

    # Pre-transpose the parameter tensors. The memory will still be shared.
    cT = T(c)
    AT = T(A)
    bT = T(b)

    # Keep references to the original problem variables (all k instances)
    c_, cT_ = c, cT
    A_, AT_ = A, AT
    b_, bT_ = b, bT

    # Bookkeeping variables and constants
    niter  = torch.zeros((k, 1, 1), dtype=torch.int64, device=device)
    status = torch.zeros((k, 1, 1), dtype=torch.int64, device=device)
    infeas = torch.zeros((k, 1, 1), dtype=torch.bool, device=device)
    badcond = torch.zeros((k, 1, 1), dtype=torch.bool, device=device) if (check_cond and m > 0) else None
    status_infeasible = torch.tensor(2, dtype=torch.int64, device=device)
    status_unbounded = torch.tensor(3, dtype=torch.int64, device=device)

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A)

    # Final values of variables as each batch item terminates.
    x_hat = torch.empty_like(x)
    x_ = torch.empty_like(x)         if want_grad else None
    y_ = torch.empty_like(y)         if want_grad else None
    z_ = torch.empty_like(z)         if want_grad else None
    tau_ = torch.empty_like(tau)     if want_grad else None
    kappa_ = torch.empty_like(kappa) if want_grad or want_pdgap else None

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    denom_p0 = norm(b - A.sum(dim=2, keepdim=True)).clamp_(min=1)
    denom_d0 = norm(c - 1).clamp_(min=1)
    denom_g0 = norm(1 + c.sum(dim=1, keepdim=True)).clamp_(min=1)

    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, r_P, r_D, r_G, mu = _indicators(A, b, c, x, y, z, tau, kappa, AT, bT, cT, denom_p0, denom_d0, denom_g0)

    # idx[i] is the original index of the batch item corresponding to current go[i]
    idx = torch.arange(k, device=device)
    
    # Main loop
    curr_iter = 0
    while True:
        with torch.no_grad():
            # go = (rho_p > tol) | (rho_d > tol) | (rho_A > tol) using inplace ops
            if numiter is None:
                go = (rho_p > tol).add_(rho_d > tol).add_(rho_A > tol)
            else:
                go = torch.full(rho_p.shape, curr_iter < numiter, device=rho_p.device, dtype=torch.bool)

            if badcond is not None:
                # Flag any batch items where nan/inf has popped up as unstable.
                non_finite = torch.isfinite(rho_p).mul_(
                            torch.isfinite(rho_d).mul_(
                            torch.isfinite(rho_A).mul_(
                            torch.isfinite(rho_g).mul_(
                            torch.isfinite(rho_mu))))).logical_not_()
                if non_finite.any():
                    badcond.masked_fill_(non_finite, True)
                    warn('Non-finite matrix for batch indices %s; result may not be accurate.' % \
                        (idx[non_finite.view(-1)].numpy()), LinAlgWarning, stacklevel=1)

                go.masked_fill_(badcond, False)

            # After at least iteration, detect status {2,3}.
            if curr_iter > 0:

                # Implement this logic using inplace ops:
                #    infeas = (rho_p < tol & rho_d < tol & rho_g < tol & tau < tol*max(kappa, 1)) |
                #             (rho_mu < tol) & (tau < tol*min(kappa, 1))
                infeas = (rho_p < tol).mul_(rho_d < tol).mul_(rho_g < tol).mul_(tau < kappa.clamp(min=1).mul_(tol)).add_(\
                         (rho_mu < tol).mul_(tau < kappa.clamp(max=1).mul_(tol)))

                go.masked_fill_(infeas, False)  # Primal or dual infeasibility, so stop

        # If any batch item is complete, store the result and remove it from the batch
        if not go.all() or curr_iter >= maxiter:
            # Partition idx into indices that continue, and indices that are terminating.
            # Example:
            #    before: go = [0, 1, 1, 0], idx = [3, 4, 7, 9]
            #    after:  dst = [4, 7], i = [1, 2], j = [0, 3]
            i = go.squeeze_().nonzero(as_tuple=True)      # indices to continue within go
            j = go.logical_not_().nonzero(as_tuple=True)  # indices to terminate within go
            dst = (idx[j],)                               # original indices to terminate

            # Set x_hat, niter, and other variables for the items that are terminating.
            x_hat.index_put_(dst, x[j] / tau[j])          # [4] Statement after Theorem 8.2
            niter.index_fill_(0, dst[0], curr_iter)
            if want_grad:
                x_.index_put_(dst, x[j])
                y_.index_put_(dst, y[j])
                z_.index_put_(dst, z[j])
                tau_.index_put_(dst, tau[j])
            if want_grad or want_pdgap:
                kappa_.index_put_(dst, kappa[j])

            # If any items terminated due to primal or dual infeasibility, set their status.
            with torch.no_grad():
                if curr_iter > 0:
                    if infeas.any():
                        # Narrow j and dst to only those that terminated due to infeas.
                        j = infeas.squeeze_().nonzero(as_tuple=True)
                        dst = (idx[j],)

                        # [4] Lemma 8.4 / Theorem 8.3, but stricter; see https://github.com/scipy/scipy/issues/11617
                        xj = x[j]
                        yj = y[j]
                        xTz = bmm(T(xj), z[j])
                        bTy_pos = bmm(bT[j], yj) >= xTz
                        cTx_pos = bmm(cT[j], xj) > -xTz
                        ATy_neg = (bmm(AT[j], yj) <= 0).all(dim=1, keepdim=True)
                        is_not_dual = bTy_pos.add(cTx_pos)  # not (bTy <  xTz and cTx <= -xTz)
                        is_primal1 = bTy_pos.mul(cTx_pos)   #     (bTy >= xTz and cTx >  -xTz)
                        is_primal2 = bTy_pos.mul_(ATy_neg)  #     (bTy >= xTz and ATy <= 0)

                        # Report infeasible if is_primal1 or (is_not_dual and is_primal2).
                        mask = bmm(bT[j], yj) > 0
                        
                        # Store 2 or 3 at each original index
                        status.index_put_(dst, torch.where(mask, status_infeasible, status_unbounded))

                # if badcond is not None and badcond.any():
                #     # Narrow j and dst to only those that terminated due to badcond. 
                #     j = badcond.squeeze_().nonzero(as_tuple=True)
                #     status.index_fill_(0, idx[j], 4)

            # Now safe to exclude exclude indices for items that have terminated.
            idx = idx[i]

            # If all items have terminated, stop.
            if len(idx) == 0:
                break

            # If items continue, but we've hit maxiter, set the remaining results and stop.
            if curr_iter >= maxiter:
                x_hat.index_put_((idx,), x[i] / tau[i])
                status.index_fill_(0, idx, 1)
                niter.index_fill_(0, idx, curr_iter)
                break

            # Remove all batch items that terminated.
            c = c[i]; A = A[i]; b = b[i]; cT = T(c); AT = T(A); bT = T(b)
            x = x[i]; y = y[i]; z = z[i]; tau = tau[i]; kappa = kappa[i]
            r_P = r_P[i]; r_D = r_D[i]; r_G = r_G[i]; mu = mu[i]
            denom_p0 = denom_p0[i]; denom_d0 = denom_d0[i]; denom_g0 = denom_g0[i]

        # Solve [4] 8.6 and 8.7/8.13/8.23
        d_x, d_y, d_z, d_tau, d_kappa, badcond = _get_delta(A, b, c, x, y, z, tau, kappa, r_P, r_D, r_G, mu, AT, bT, cT, idx, cholesky, check_cond)

        # [4] Section 4.3
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)

        # [4] Equation 8.9
        x, y, z, tau, kappa = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        # [4] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, r_P, r_D, r_G, mu = _indicators(A, b, c, x, y, z, tau, kappa, AT, bT, cT, denom_p0, denom_d0, denom_g0)

        curr_iter += 1

    # Squeeze the final arrays into a convenient shape
    x_hat.squeeze_(2)                # (k,n,1) -> (k,n)
    status.squeeze_(2).squeeze_(1)   # (k,1,1) -> (k,)
    niter.squeeze_(2).squeeze_(1)    # (k,1,1) -> (k,)
    result = [x_hat, status, niter]

    # Return measure of primal-dual gap in homogeneous variables
    if want_pdgap:
        result.append(kappa_.squeeze(2).squeeze_(1))    # (k,1,1) -> (k,)

    # Return a function that can be called to compute the gradient
    if want_grad:
        def grad(dx_hat, homogeneous_mode=True, aoe_closed_form=False):

            # ADG closed form mode
            if aoe_closed_form:
                # ADG loss in closed form. This requires dx_hat to be either positive or negative c-vector.
                c = c_.squeeze(2)
                # print('c in grad', c)
                assert dx_hat is not None

                # print('torch.allclose( dx_hat.abs(), c.abs()/len(dx_hat) )',torch.allclose( dx_hat.abs(), c.abs()/len(dx_hat) ))
                assert torch.allclose( dx_hat.abs(), c.abs()/len(dx_hat) ) == True, "aoe_closed_form requres dx_hat[i,:] to equal plus/minus c[i,:]"
                adg_neg = (torch.sign(dx_hat) != torch.sign(c)).any(dim=1)
                # Convery homogeneous y_ to non-homogeneous y
                y_hat = (y_ / tau_)
                y_hat[adg_neg] *= -1  # Flip sign of y to reflect ADG 'direction'
                # Closed form expression for ADG gradient of an LP

                dc = torch.zeros_like(c_).squeeze_(2)
                dA = -y_hat @ T(x_hat.unsqueeze(2))
                db = y_hat.squeeze_(2)
                dA = dA/len(dx_hat)
                db = db/len(dx_hat)

                return dc, dA, db

            # Matrix inversion mode (homogeneous or regular).
            # Build transpose of the M matrix to invert.
            #  M.T = [[    0,    A, D(z),      0, -c],
            #         [  A.T,    0,    0,      0,  b],
            #         [    I,    0, D(x),      0,  0],
            #         [    0,    0,    0,    tau, -1],
            #         [ -c.T, -b.T,    0,  kappa,  0]],
            i = torch.arange(k).view(-1, 1).expand(-1, n).reshape(-1)
            j0 = torch.arange(n).repeat(k)
            j1 = j0.add(n+m)
            if homogeneous_mode:
                M = c_.new_zeros((k, 2*n + m + 2, 2*n + m + 2))
                M[:,:n,n:n+m] = AT_
                M[:,n:n+m,:n] = A_
                M.index_put_((i, j1, j0), c_.new_tensor(1.))  # I
                M.index_put_((i, j0, j1), z_.view(-1))        # diag(z)
                M.index_put_((i, j1, j1), x_.view(-1))        # diag(x)
                M[:,:n,-1:] = -c_
                M[:,n:n+m,-1:] = b_
                M[:,-1:,:n] = -cT_
                M[:,-1:,n:n+m] = -bT_
                M[:,-1:,-2:-1] = kappa_
                M[:,-2:-1,-2:-1] = tau_
                M[:,-2:-1,-1] = -1
            else:
                M = c_.new_zeros((k, 2*n + m, 2*n + m))
                M[:,:n,n:n+m] = AT_
                M[:,n:n+m,:n] = A_
                M.index_put_((i, j1, j0), c_.new_tensor(1.))     # I
                M.index_put_((i, j0, j1), (z_ / tau_).view(-1))  # diag(z)
                M.index_put_((i, j1, j1), (x_ / tau_).view(-1))  # diag(x)

            # Invert M once. Big matrix :\
            Minv = M.inverse()
            
            # Tensors to accumulate gradients
            dc = torch.zeros_like(c_)
            dA = torch.zeros_like(A_)
            db = torch.zeros_like(b_)
            xT = T(x_)

            def accum_grad(P):
                nonlocal dc, dA, db
                # P is the relevant vector-Jacobian product
                alpha = P[:,:n,:]                  # (k,n,1)
                beta = P[:,n:n+m,:]                # (k,m,1)
                
                if homogeneous_mode:
                    epsilon = P[:,-1:,:]           # (k,1,1)
                    dc += tau_ * alpha + epsilon * x_
                    db += tau_ * beta  - epsilon * y_
                    dA -= y_.bmm(T(alpha)) + beta.bmm(xT)
                else:
                    dc += alpha
                    db += beta
                    dA -= (y_ / tau_).bmm(T(alpha)) + beta.bmm(xT / tau_)

            # Gradient contribution of dx_hat
            if dx_hat is not None:
                # Convert dx_hat of original LP to dx of homogeneous LP
                assert dx_hat.shape == (k, n), 'dx_hat.shape%s, (k,n) = %s'%(dx_hat.shape, [k,n])
                dx_hat = dx_hat.unsqueeze(2)             # (k,n,1)
                if homogeneous_mode:
                    dx = dx_hat / tau_
                    dtau = xT.bmm(dx_hat).div_(tau_**2)  # (k,1,n) @ (k,n,1) => (k,1,1)
                    P = Minv[:,:,:n].bmm(dx)             # (k,2n+m+2,n) @ (k,n,1)  =>  (k,2n+m+2,1)
                    P -= Minv[:,:,-1:] * dtau            # (k,2n+m+2,1) * (k,1,1)  =>  (k,2n+m+2,1)
                    '''print("dx:", dx.numpy())'''
                else:
                    P = Minv[:,:,:n].bmm(dx_hat)         # (k,2n+m+2,n) @ (k,n,1)  =>  (k,2n+m+2,1)
                accum_grad(P)

            return dc.squeeze_(2), dA, db.squeeze_(2)

        result.append(grad)

    return result

def _linprog_batch_std(c, A, b, maxiter, tol, check_cond, cholesky, raise_error, want_pdgap, want_grad, numiter):

    # Choose tolerance automatically.
    # If float32, use a weak tolerance to avoid numerical instability as solution is approached.
    # TODO: can fix this by implementing the least-squares solver as fallback, like scipy.optimize._linprog_ip does.
    if tol is None:
        tol = 1e-8 if c.dtype is torch.float64 else 1e-6

    x, status, *rest = _ip_hsd(
        torch.unsqueeze(c, 2),  # (k,n) -> (k,n,1)
        A,                      # (k,m,n)
        torch.unsqueeze(b, 2),  # (k,m) -> (k,m,1)
        0.99995,  # alpha0
        0.1,      # beta
        maxiter,
        tol,
        check_cond,
        cholesky,
        want_pdgap,
        want_grad,
        numiter)

    if raise_error and torch.any(status >= 1):
        raise InfeasibleOrUnboundedError("At leaast one of the instances failed. Status: %s" % status)
    
    return [x, status] + rest

def linprog_batch_std(c, A, b, maxiter=1000, tol=None, check_cond=False, cholesky=False, raise_error=False,
                      want_pdgap=False, want_grad=False, numiter=None):
    r"""
    Solve a linear programming problem in standard form:
    Minimize::
        c @ x
    Subject to::
        A @ x == b
        x >= 0
    using the interior point method of [4].
    Parameters
    ----------
    c : (k,n)-tensor (float)
        Vectors defining each of the k linear objective functions to be minimized.
    A : (k,m,n)-tensor (float)
        Matrices such that ``A @ x``, gives the values of the equality
        constraints at ``x`` for each of the k problem instances.
    b : (k,m)-tensor (float)
        Vectors representing the RHS of each set of equality constraints.
    maxiter : int
        The maximum number of iterations of the algorithm.
    tol : float
        Termination tolerance; see [4]_ Section 4.5.
        Note that extremely small tolerances can result in numerical instability
        as the solution is approached. For float32 dtypes, even tolerances of 1e-6
        can cause instability.
    raise_error : bool
        Raise an exception upon encountering any failure, such as any problem instance
        being detected as infeasible or unbounded. Otherwise `status` contains failures.
    check_cond : bool
        Checks the condition number of each solve and returns status 4 for any
        batch item ill-conditioned during solve.
    Returns
    -------
    x : (k,n)-tensor (float)
        The k solution vectors for the given n-dimensional problem instances.
    status : k-tensor (int)
        An integer representing the exit status of each problem instance:
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    niter : k-tensor (int)
        The number of iterations taken to solve each problem instance
    """
    # Stub that forwards sanitized args to _linprog_batch_std
    assert isinstance(c, torch.Tensor)
    assert isinstance(A, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert A.dim() == 3
    k, m, n = A.shape
    dtype = A.dtype
    device = A.device
    assert c.shape == (k, n)
    assert c.dtype == dtype
    assert c.device == device
    assert b.shape == (k, m)
    assert b.dtype == dtype
    assert b.device == device
    return _linprog_batch_std(c, A, b, maxiter=maxiter, tol=tol, check_cond=check_cond,
                              cholesky=cholesky, raise_error=raise_error,
                              want_pdgap=want_pdgap, want_grad=want_grad, numiter=numiter)


def linprog_scipy_batch(c, A_ub, b_ub, A_eq=None, b_eq=None, nneg=False, maxiter=1000, tol=None,
                        check_cond=False, cholesky=False, raise_error=False, want_pdgap=False,
                        want_grad=False, numiter=None):
    r"""
    Solve a linear programming problem of the form:
    Minimize::
        c @ x
    Subject to::
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        x >= 0                    # Optional. Assumed if nneg=True.
    by converting it to standard form and then solving it with linprog_batch_std.
    Returns (c, A, b) as a tuple so that it may be used with linprog_batch.

    All other parameters and return values are the same as linprog_batch_std.
    """
    assert isinstance(c, torch.Tensor)
    assert isinstance(A_ub, torch.Tensor)
    assert isinstance(b_ub, torch.Tensor)
    assert A_ub.dim() == 3
    k, m_ub, n = A_ub.shape
    dtype = A_ub.dtype
    device = A_ub.device
    assert c.shape == (k, n), 'c.shape=%s, (k,n)=%s '%(c.shape, [k,n])
    assert c.dtype == dtype
    assert c.device == device
    assert b_ub.shape == (k, m_ub)
    assert b_ub.dtype == dtype
    assert b_ub.device == device
    if A_eq is not None:
        assert isinstance(A_eq, torch.Tensor)
        assert isinstance(b_eq, torch.Tensor)
        assert A_eq.dim() == 3
        m_eq = A_eq.shape[1]
        assert A_eq.shape == (k, m_eq, n)
        assert A_eq.dtype == dtype
        assert A_eq.device == device
        assert b_eq.shape == (k, m_eq)
        assert b_eq.dtype == dtype
        assert b_eq.device == device

    # TODO: Allow nneg to be a mask specifying which variables are assumed to be non-negative.
    #       None/0 would specify that none of them are.
    #       True/1 would specify that all of them are.
    #       A uint8 mask or a list of integers would specify a subset.

    # Build the standard form LP:
    #
    #          x     s
    #     c = [c,    0]
    #
    #     A = [A_ub, I;     b = [b_ub,
    #          A_eq, 0]          b_eq]
    #
    #     x, s >= 0
    #
    c_std = torch.cat((c, torch.zeros((k, m_ub), dtype=dtype, device=device)), 1)
    A_std = torch.cat((A_ub, torch.eye(m_ub, dtype=dtype, device=device).expand(k, m_ub, m_ub)), 2)
    b_std = b_ub
    if A_eq is not None:
        A_std = torch.cat((A_std, torch.cat((A_eq, torch.zeros((k, m_eq, m_ub), dtype=dtype, device=device)), 2)), 1)
        b_std = torch.cat((b_std, b_eq), 1)

    if not nneg:
        # If non-negativity constraints on x should be omitted, we must
        # introduce other non-negative variables z to build standard form LP:
        #
        #          x     s      z
        #     c = [c,    0,    -c]
        #
        #     A = [A_ub, I, -A_ub;     b = [b_ub,
        #          A_eq, 0, -A_eq]          b_eq]
        # 
        #     x, s, z >= 0
        #
        c_std = torch.cat((c_std, -c), 1)
        A_std = torch.cat((A_std, -A_ub if A_eq is None else
                                  -torch.cat((A_ub, A_eq), 1)), 2)

    x_std, *rest = _linprog_batch_std(c_std, A_std, b_std, maxiter=maxiter, tol=tol, check_cond=check_cond,
                                      cholesky=cholesky, raise_error=raise_error, want_pdgap=want_pdgap,
                                      want_grad=want_grad, numiter=numiter)

    # Pull out just the original decision variables, or reconstruct from the split non-negative components.
    x = x_std[:,:n]
    if not nneg:
        # x.sub_(x_std[:,-n:])
        x = x - x_std[:,-n:]      # fix a bug causde by in-place substraction

    # Wrap the gradient callback for standard-form solution, if applicable.
    if want_grad:
        if nneg:
            # With non-negativity constraints on x
            @pass_none
            def dx_std(dx):
                r = dx.new_empty((k, n+m_ub))  #update 04/16
                r[:,:n] = dx
                r[:,n:] = 0
                return r

            @pass_none
            def dc(dc_std): return dc_std[:,:n]

            @pass_none
            def dA_ub(dA_std): return dA_std[:,:m_ub,:n]

            @pass_none
            def db_ub(db_std): return db_std[:,:m_ub]

            @pass_none
            def dA_eq(dA_std): return dA_std[:,m_ub:,:n]

            @pass_none
            def db_eq(db_std): return db_std[:,m_ub:]

        else:
            # Without non-negativity constraints on x
            @pass_none

            def dx_std(dx):
                r = dx.new_empty((k, n+m_ub+n))  #update 04/16
                r[:,:n] = dx
                r[:,n:-n] = 0
                r[:,-n:] = -dx
                return r
            
            @pass_none
            def dc(dc_std): return dc_std[:,:n] - dc_std[:,-n:]

            @pass_none
            def dA_ub(dA_std): return dA_std[:,:m_ub,:n] - dA_std[:,:m_ub,-n:]

            @pass_none
            def db_ub(db_std): return db_std[:,:m_ub]

            @pass_none
            def dA_eq(dA_std): return dA_std[:,m_ub:,:n] - dA_std[:,m_ub:,-n:]

            @pass_none
            def db_eq(db_std): return db_std[:,m_ub:]

        grad_std = rest[-1]  # grad is last argument

        def grad(dx, homogeneous_mode=True, aoe_closed_form=False):
            if dx is not None: assert dx.shape == (k, n)
            dc_std, dA_std, db_std = grad_std(dx_std(dx), homogeneous_mode=homogeneous_mode, aoe_closed_form=aoe_closed_form)

            return dc(dc_std), dA_ub(dA_std), db_ub(db_std), dA_eq(dA_std), db_eq(db_std)

        # Swap out the standard-form grad for our new grad
        rest = rest[:-1] + [grad]

    return [x] + rest
