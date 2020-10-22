# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for reproducing the results of NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


import torch
import sys
import os
sys.path.append("..")
from util import as_tensor, as_numpy, tensor, HiddenPrints
from linprog_solver import linprog_scipy_batch
from copy import deepcopy
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer
import time
import numpy as np

DT = torch.float64



def homogeneous_solver(c, A_ub, b_ub, A_eq, b_eq, tol, nneg=False, want_grad = False):
    """
    call homogeneous IPM, 
    
    c, A_ub, b_ub, A_eq, b_eq: LP coefficients
          tol: foward solver tolerance
         nneg: non-negative variables
    want_grad: True/False, enable/disable gradient callbacks of the homogeneous IPM.
     std_form: True/False the LP is standard form or not

    c.shape = nIns, nVar, 1
    A.shape = nIns, nCons, nVar
    b.shape = nIns, nVar, 1
    for homogeneous solver, one need to Squeeze to remove 3rd dimension of c and b

    Status:
    0 : Optimization terminated successfully
    1 : Iteration limit reached
    2 : Problem appears to be infeasible
    3 : Problem appears to be unbounded
    4 : Serious numerical difficulties encountered
    """
    # Squeeze to remove 3rd dimension if it exists
    c = c.squeeze(2)
    b_ub = b_ub.squeeze(2)
    b_eq = b_eq.squeeze(2) if b_eq is not None else None
    
    x_sol, *rest = linprog_scipy_batch(c, A_ub, b_ub, A_eq, b_eq, 
                                          tol = tol, nneg = nneg, check_cond = True,
                                          want_pdgap = True, want_grad = want_grad)

    x_sol.unsqueeze_(2)
    return [x_sol] + rest

class LinprogImplicitDiff(torch.autograd.Function): 
    """
    customized torch.autograd.Function which allows the use of gradient callbacks of homogeneous IPM 
    """
    @staticmethod
    def forward(ctx, c, A_ub, b_ub, A_eq, b_eq, tol, nneg, aoe_closed_form):
        # Dont need gradient function for backpropagation, thus, detach lp parameter to speed up the forward process
        x, s, niter, pdgap, grad = homogeneous_solver(c.detach(),  
                                                      A_ub.detach() if A_ub is not None else None, 
                                                      b_ub.detach() if b_ub is not None else None, 
                                                      A_eq.detach() if A_eq is not None else None, 
                                                      b_eq.detach() if b_eq is not None else None, 
                                                      tol  = tol, 
                                                      nneg = nneg, 
                                                      want_grad = True)

        ctx.mark_non_differentiable(s, niter, pdgap)  # mark non-differentiable to speed up the process
        ctx.lp_grad = grad  # save the gradient function for the backward process
        ctx.aoe_closed_form = aoe_closed_form
        x.requires_grad_()  # enable gradients of x only if one of the lp parameters also requires gradients
        
        return x, s, niter, pdgap
            
    @staticmethod
    def backward(ctx, grad_x, grad_s, grad_niter, grad_pdgap):
        # print('grad_x in backward', grad_x.squeeze(2))
        # use the gradient function and grad_x to compute the gradient of each lp parameter using the implicit differentiation formula
        dc, dA_ub, db_ub, dA_eq, db_eq = ctx.lp_grad(grad_x.squeeze(-1), 
                                                     homogeneous_mode = False,
                                                     aoe_closed_form = ctx.aoe_closed_form)
        if dA_eq.nelement() == 0:
            dA_eq = None 
            db_eq = None    
        return dc.unsqueeze(2) if dc is not None else dc,\
               dA_ub, db_ub.unsqueeze(2) if db_ub is not None else db_ub,\
               dA_eq, db_eq.unsqueeze(2) if db_eq is not None else db_eq, None, None, None

def call_cvxpylayer(lp, layer, tol = 1e-8, verbose = True):
    """
    send input data to an optimization layer defined with cvxpylayer and solve the forward problem

    Input:
        lp    = (c, A_ub, b_ub, A_eq, b_eq)
        layer: an optimization layer encoded with cvxpylayer
            disable the multi-threading by setting 'n_jobs_forward':1, 'n_jobs_backward':1
            forcing cvxpylayer to search for high precision solutions by setting 'max_iters': int(1e8)
        tol:   solution precision of the forward problem, the default value is 1e-8

    Output:
        soln:   (N, n),  N instances (batch size), n variables
        status: (N)
    """
    c, A_ub, b_ub, A_eq, b_eq = lp

    N, m, n = A_ub.shape

    try:
        if A_eq is not None:
            with HiddenPrints():
                soln, = layer(c.squeeze(-1), A_ub, b_ub.squeeze(-1), A_eq, b_eq.squeeze(-1), 
                            solver_args = {'n_jobs_forward':1, 'n_jobs_backward':1, 'eps':tol, 'max_iters': int(1e8)})
            
        else:
            with HiddenPrints():
                soln, = layer(c.squeeze(-1), A_ub, b_ub.squeeze(-1), 
                            solver_args = {'n_jobs_forward':1, 'n_jobs_backward':1, 'eps':tol, 'max_iters': int(1e8)})
        status = torch.zeros((N), dtype = DT)
    # Mannually add exceptions to catch any error of the forward solver and return solution status of 5.
    except Exception as exception:
        if exception.__class__.__name__ == 'SolverError':
            if verbose == True: 
                print('[%s]: the LP is unbounded or infeasible, return arbitarily large values for soln'%(exception.__class__.__name__) )
            status = torch.ones(len(c), dtype = DT)*5
            soln = torch.ones( (N,n) ,dtype=DT)
        else:
            if verbose == True: 
                print('Other Error: [%s], return arbitarily large values for soln'%(exception.__class__.__name__) )
            status = torch.ones((N), dtype = DT)*5         
            soln = torch.ones( (N,n) ,dtype=DT)

    return soln.unsqueeze(-1), status, None

def solve_lp(lp, grad_mode, tol, nneg, layer = None):
    """
    According to grad_mode, call different functions to solve the forward problems

    Input 
        lp    = (c, A_ub, b_ub, A_eq, b_eq)
        grad_mode:  numerical_grad  - forward:  homogeneous_solver
                                    backward: infinite difference
                    None            - forward:  homogeneous_solver
                                    backward: None
                    cvxpylayer      - forward:  cvxpylayer
                                    backward: cvxpylayer
                    bprop           - forward:  homogeneous_solver
                                    backward: backpropagation (PyTorch automatic differentiation)
                    implicit        - forward:  homogeneous_solver
                                    backward: implicit differentiation (see backward in LinprogImplicitDiff)
                    direct          - forward:  homogeneous_solver
                                    backward: closed-form expression (see backward in LinprogImplicitDiff)
                
        layer: an optimization layer encoded with cvxpylayer
        tol:   solution precision of the forward problem, the default value is 1e-8
        nneg = True/False (decision variables are non-negative or not)

    Ouput:
        solutions of forward problem
    """
    if grad_mode == 'implicit':
        return LinprogImplicitDiff.apply(*lp, tol, nneg, False)
    elif grad_mode == 'direct':
        return LinprogImplicitDiff.apply(*lp, tol, nneg, True)

    elif grad_mode in ['backprop', None, 'numerical_grad']:
        # print('c', lp[0].suqeeze(0))
        # print('A_ub', lp[1].suqeeze(0))
        # print('b_ub', lp[2].suqeeze(0))
        return homogeneous_solver(*lp, 
                                  tol = tol, 
                                  nneg = nneg, 
                                  want_grad= False)
    elif grad_mode =='cvxpylayer':
        assert layer is not None
        return call_cvxpylayer(lp, layer, tol, False)
    else:
        raise ValueError('Wrong grad_mode value')

def is_feasible(x, A_ub, b_ub, tolerance=0.0):
    """
    if a solution x satisfy the ineq. constraints A_ub@x <= b_ub
    """
    return np.all(A_ub @ x <= b_ub + tolerance)  


def plot_linprog_hull(c, A_ub, b_ub, alpha, 
                        c_color, cons_color, axes, cxy = None):
    """
    plot convexhull of a 2D linear program with ineq. constraints only.
    """
    c, A_ub, b_ub = as_numpy(c, A_ub, b_ub)
    
    b_ub = b_ub.reshape(-1,1)
    A_ub = A_ub.squeeze()
    n, m = A_ub.shape
        
    if cxy:
        plot_c(c, c_color, alpha, cx = cxy[0], cy = cxy[1], axes = axes)
    else:
        plot_c(c, c_color, alpha, cx = 0, cy = 0, axes = axes)

    # Plot inequality constratints
    vertices, intersect = enumerate_polytope_vertices(A_ub, b_ub)
    
    for i in range(n):
        pt_ind = [j for j in range(len(intersect)) if i in intersect[j]] 
        x = vertices[pt_ind,0]
        y = vertices[pt_ind,1]
        axes.plot(x, y, color=cons_color, alpha=alpha, linewidth=1.5)

# Plot cost vectors
def plot_c(c, color, alpha, cx, cy, axes):
    """
    plot cost vector of a 2D linear program.
    """
    c1, c2 = c.ravel().copy()
    _sum = (abs(c1)+abs(c2))
    c1 /= _sum
    c2 /= _sum
    axes.arrow(cx, cy, c1, c2, color=color, 
                alpha=alpha, head_width=0.10, 
                length_includes_head=True)

def enumerate_polytope_vertices(A_ub, b_ub):
    """
    enumerate all vertices of a polytope, 2D only
    """
    A_ub, b_ub = as_numpy(A_ub, b_ub)
    n, m = A_ub.shape
    assert m == 2, "Only 2D supported"
    vertices = []
    intersect = []

    # check every unique (i, j) combination
    for i in range(n):
        for j in np.arange(i+1, n):
            try:
                # intersect constraints i and j and collect it if it's feasible
                x = np.linalg.inv(A_ub[(i, j), :]) @ b_ub[(i, j), :]
                if is_feasible(x, A_ub, b_ub, 1e-6):
                    vertices.append(x.T)
                    intersect.append((i, j))
            except np.linalg.LinAlgError:
                pass  # if constraints i and j happen to be aligned, skip this pair

    return np.vstack(vertices), np.array(intersect)


