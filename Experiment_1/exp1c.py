# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for reproducing the results of NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


from generator import convexhull_generator, read_baseline_convexhull
import os
import sys
# from util.py importing pre-define functions which are shared for all experiments
sys.path.append("..")
from util import T, tensor, as_tensor, as_numpy, _bsubbmm, ParametricLP
from util import _AOE_, normalize_c, create_cvx_layer
from util import EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded, UnboundedWeightsValue
from util import VERBOSE_ERROR_MESSAGE, GRAD_MODE_ERROR_MESSAGE, OUTER_METHOD_MESSAGE
from util import collect_calls, compute_loss_over_time, plot_success_rate

# from exp1_util.py importing pre-define functions which are used for experiment 1
from exp1_util import homogeneous_solver, LinprogImplicitDiff, solve_lp
from exp1_util import plot_linprog_hull, plot_c, is_feasible, enumerate_polytope_vertices

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import pickle
import scipy.optimize as opt

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

import warnings
warnings.simplefilter('once')

def read_InvLP(filename, 
               verbose = 0):
    """
    read inverse LP instance from file
    """
    assert verbose in [0,1,2], 'invalid value for verbose\n verbose = 0\tdont show alg log\n, verbose = 1\tshow the final alg res, verbose = 2\tshow details of alg log'
    lines=[]
    with open(filename, "rt") as fp:
        for line in fp:
            lines.append(line.rstrip('\n')) 
    
    nCons, nVar = np.fromstring(lines[1], dtype=float, sep=' ').astype(int)

    if verbose == 2: print('Reading from %s\n[num constraint %d], [num variables %d]'%(filename, nCons, nVar))
        
    temp = [lines[i+1:i+nVar+1] for i in range(len(lines)) if "c" in lines[i] ]
    c = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ c\n\t',c.ravel())
        
    temp = [lines[i+1:i+nCons+1] for i in range(len(lines)) if "A_ub" in lines[i]]
    A_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ').reshape(1,-1) for row in temp[0]])
    if verbose == 2: print('+++ A_ub\n\t', A_ub)

    temp = [lines[i+1:i+nCons+1] for i in range(len(lines)) if "b_ub" in lines[i]]
    b_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ b_ub\n\t', b_ub.ravel())

    temp = [lines[i:i+2] for i in range(len(lines)) if "x_target" in lines[i]]
    x_target = np.array(np.fromstring(temp[0][1], dtype=float, sep=' ')).reshape(-1,1)
    if verbose == 2: print('+++ x_target\n\t',x_target.ravel())

    target_type = int(temp[0][0][-1])
    
    return c, A_ub, b_ub, x_target, target_type

def record_InvLP(filename, 
                 data, 
                 target_type):
    """
    save inverse LP instance in .txt file

    data = (A_ub, b_ub, c, x_target)
    target_type = 1/0, target is feasible/infeasible w.r.t. A_ub, b_ub
    """

    (A_ub, b_ub, c, x_target) = data

    nCons,nVar = A_ub.shape
    with open(filename, 'w') as LP_file:
        # print LP size, m - # constrains, n - # # var  
        LP_file.write("LP Size\n")      
        np.savetxt(LP_file, np.array(A_ub.shape).reshape(1,-1), fmt="%.d")
        LP_file.write("\n")

        # print LP, Matrix A and b for constraints Ax <= b
        LP_file.write("A_ub\n")
        np.savetxt(LP_file, A_ub, fmt="%.6f")
        LP_file.write("b_ub\n")
        np.savetxt(LP_file, b_ub, fmt="%.6f")
        LP_file.write("\n")
        
        LP_file.write("c\n")
        np.savetxt(LP_file, c, fmt="%.6f")
        LP_file.write("\n")

        LP_file.write("x_target %d\n"%target_type)
        np.savetxt(LP_file, x_target.reshape(1,nVar), fmt="%.6f")
        

def _generate_InvLP(ind_ins, 
                    nVar, 
                    nCons, 
                    target_type,    # 1 - feasible target, 0 - infeasible target
                    directory   =  None,
                    verbose     =  False):
    
    """
    ind_ins: index of instance
       nVar: num. of variables
      nCons: num. of constraints
    target_type: 1-x_target is feasible w.r.t. A_ub, b_ub, 0 - x_target is infeasible w.r.t A_ub, b_ub

    Generate an IO instance for learning coefficients (c, A_ub, b_ub) of a LP directly (with ineq. constraints only) through the following steps:
    1). read  baseline convexhull from file (i.e., A_ub, b_ub and vertices)
    2). sample c vector
    3). solve LP(c, A_ub, b_ub) and obtain x_
    4). If target_type == 1: x_target is a convex combination of all vertices
        If target_tyep == 2: x_target = x_ + random noise.
    """

    sigma           =  1.0    # used for random sampling
    feasibility_tol =  1e-5   # tolerance to decide if x_target if feasible or infeasible

    def corrupt_sol(x, vertices, target_type):
        if target_type == 0 :  # infeasible target
            x_ = x + np.random.uniform(-sigma, sigma, size=(1, nVar, 1))
            
        elif target_type == 1:  #feasible target
            a  = np.random.dirichlet(np.ones(len(vertices)), 1).ravel()
            x_ = (a@vertices).reshape(x.shape)
        else:
            print('ERROR: Invalid entry for target_type')
            
        return x_

    def calc_residual(b, A, x): 
        if A is None:
            return None
        b, A, x = as_tensor(b, A, x)
        # Using _bsubbmm().neg_() avoids temporaries and is slightly faster than (b - bmm(A, x))
        return _bsubbmm(b, A, x).neg_().view(-1)  # -(-b + A@x) == b - A@x
            
    lp_file = directory + 'LP_baseline/%dvar%dcons/%dvar%dcons_LP%d.txt'%(nVar, nCons, nVar, nCons, ind_ins)
    A_ub, b_ub, vertices = read_baseline_convexhull(lp_file)
    A_eq, b_eq = None, None
    if verbose == True: 
        print('==== read base_lp from ', lp_file)
        
    np.random.seed(ind_ins)
    if verbose == True: 
        print('==== generate true c ~U(-sigmam, sigma) with np.random.seed(%d)'%ind_ins)

    s_sum = 1
    trail_c = 0
    while s_sum != 0:
        c   = np.random.uniform(-sigma, sigma, size=(nVar,1)).round(6)
        c_, A_ub_, b_ub_ = as_tensor(c[None, :,:], A_ub[None, :,:], b_ub[None, :,:])
        c_   = normalize_c(c_)

        soln_true = homogeneous_solver(c_, A_ub_, b_ub_, None, None, tol=1e-8)
        s_sum = sum(soln_true[1])
        trail_c +=1
        if verbose == True:
            print('\tTrail [%d], LP status = %s'%((trail_c), [int(s_) for s_ in soln_true[1]]))


    fea_eq    = None
    fea_ub    = None
    x_feasibility  = None
    iter_i = 0
    np.random.seed(ind_ins)
    if verbose == True: 
        if target_type == 1:
            print('==== Generate a feasible target as convex combination of all vertices')
        elif target_type == 0:
            print('==== Generate a infeasible target adding random noise ~U(-sigmam, sigma) to the current soln')


    while x_feasibility != target_type:
        x_target = corrupt_sol(soln_true[0].detach().numpy(), vertices, target_type)

        if A_eq:
            fea_eq  = all(abs(calc_residual(b_eq[None, :, :], A_eq[None, :, :], x_target)) < feasibility_tol)   # for A_eq@x - b = 0, check if |b - A_eq@x| <= tol

        fea_ub = all(calc_residual(b_ub[None, :, :], A_ub[None, :, :], x_target) > - feasibility_tol) # for A_ub@x - b <=0 check if b_ub - A_ub@x >= -tol

        x_feasibility = max(fea_eq, fea_ub) if fea_eq else fea_ub
        if verbose >=1:
            print('\tTrail [%d], target_type= %s, fea_eq = %s, fea_ub = %s'%(iter_i+1, target_type, fea_eq, fea_ub))
        
        iter_i +=1

    dir_ = directory + 'InvLP_Ins/%dvar%dcons/'%(nVar, nCons)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    if nVar == 2:
        alpha = 1

        fig, ax1 = plt.subplots(1,1,figsize=(10,10))
        plot_linprog_hull(c, A_ub, b_ub, 
                            alpha, c_color = 'g', cons_color='g', axes = ax1)
        
        ax1.plot(*x_target.ravel(), linestyle = 'None',
                marker = 'o', mfc='None', mec='orange',
                ms =15, mew=3, alpha=0.5)
        ax1.set_title('Init InvLP - %d var %d cons, Ins %d'%(nVar, nCons, ind_ins), fontsize=14)

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        fig.savefig(dir_+'%dvar%dcons_InvLP%d.png'%(nVar, nCons, ind_ins), dpi=100)


    filename = dir_ + '%dvar%dcons_InvLP%d.txt'%(nVar, nCons, ind_ins)
    record_InvLP(filename, (A_ub, b_ub, c, x_target), target_type)

def generate_InvLP(lp_size, 
                    nIns,
                    directory,
                    verbose):
    """
    call _generate_InvLP to generate IO instances for learning coefficients of LP directly
    """
    for lp_i in range(len(lp_size)):
        nVar, nCons = lp_size[lp_i]
        i_ = np.arange(0, nIns, 1)

        # randomly split instances to generate feasible/infeasible target
        np.random.seed(lp_i)
        i_permutation     = np.random.permutation(nIns)
        ins_fea_target    = i_[i_permutation[ :nIns//2  ]]
        ins_infea_target  = i_[i_permutation[  nIns//2: ]]
        if verbose > 1:
            print('LP Size: %d Variable, %d Constraints'%(nVar, nCons))
            print('InvLP instances whoes target soln is !!feasible!! w.r.t. the current convexhull\n',np.sort(ins_fea_target))
            print('InvLP instances whoes target soln is !!infeasible!! w.r.t. the current convexhull\n',np.sort(ins_infea_target))
        
        for ind_ins in range(nIns):
            if verbose >=1 :
                print('==== generate InvPLP instances for Exp 1.b: %d var, %d cons, ins %d'%(nVar, nCons, ind_ins))

            target_type = 1 if ind_ins in ins_fea_target else 0

            _generate_InvLP(ind_ins, 
                           nVar, 
                           nCons, 
                           target_type     =  target_type,    # 1 - feasible target, 0 - infeasible target
                           directory       =  directory, 
                           verbose         =  0)

def inverse_linprog(nVar, 
                    nCons, 
                    lp, 
                    x_target, 
                    inner_tol        =  1e-8,  # by default, use high-precision forward solver for the inner problem.
                    outer_method     =  None,
                    grad_mode        =  None,
                    runtime_limit    =  20,
                    verbose          =  0,    
                    collect_evals    =  False):

    """
    Learn LP coefficients directly, that is, w comprises all LP coefficients

    Input:
         u_train:   list of feature u for training, len(u) = N
        x_target:   list of target solutions, len(x_target) = N
             plp:   a parametric LP class (see parametric_lp_vectorized for more details)
       inner_tol:   tolerance of forward solver
    outer_method:   algorithms for solving the bi-level ILOP formulation
                        COBYLA: gradient-free NLP solver
                         SLSQP: gradient-based NLP solver (SQP algorithm encoded in scipy)
       grad_mode:   options for computing gradients when using SLSQP
                                  None: no gradients
                        numerical_grad: infinite difference
                              backprop: backpropogation
                              implicit: implicit differentiation
                                direct: close-form expression
                            cvxpylayer: cvxpylayer
    Output:
      best_loss:  best loss value
          evals:  (if collect_evals = True), a dict contains all relevant results (e.g., collection of w and loss at every iteration)
    """

    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training if loss<loss_tol
    outer_tol        =  1e-15       # tolerance for scipy.opt package
    outer_maxiter    =  10000       # iteration for scipy.opt package (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    outer_maxeval    =  10000       # max iteration for evaluation fop (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance
    boundedness_cons =  False       
    nneg             =  False       # argument of homogeneous_solver, False means there is no non-negativity constraints

    assert verbose in [0,1,2], VERBOSE_ERROR_MESSAGE
    assert grad_mode in (None, 'numerical_grad', 'backprop', 'implicit', 'direct', 'cvxpylayer'), GRAD_MODE_ERROR_MESSAGE
    assert outer_method in ('RS', 'COBYLA', 'SLSQP'), OUTER_METHOD_MESSAGE

    num_evals      = 0
    curr_w         = None
    curr_lp        = None
    curr_loss      = 999
    curr_soln      = None                     # The linprog results to the LPs for curr_w
    curr_target_feasibility_ub = None    # The b_ub - A_ub @ x_target residuals for curr_w
    curr_target_feasibility_eq = None    # The b_eq - A_eq @ x_target residuals for curr_w    
    best_loss      = float(999)
    loss_collect   = []
    requires_grad  = (outer_method != 'COBYLA') and (grad_mode in ('backprop', 'implicit', 'direct', 'cvxpylayer'))
    layer          = None   # check cvxpylayer object as None, which will be initialized only if grad_mode == 'cvxpylayer'

    # update curr_w = w
    def is_w_curr(w):        
        return (curr_w is not None) and np.array_equal(w, curr_w.detach().numpy())
    
    # Returning cur_lp, i.e., LP(c, A_ub, b_ub, A_eq, b_eq)
    def get_curr_lp(w):
        nonlocal curr_w, curr_lp, curr_soln, curr_loss
        nonlocal curr_target_feasibility_ub
        nonlocal curr_target_feasibility_eq
        if not is_w_curr(w):
            curr_w       =   torch.tensor(w,requires_grad = requires_grad)    # initialize parameters and enable gradient for SLSQP
            c            =   curr_w[:nVar].view(-1,1).unsqueeze(0) 
            c            =   normalize_c(c)
            A_ub         =   curr_w[nVar : -nCons].view(nCons,nVar).unsqueeze(0)
            b_ub         =   curr_w[-nCons : ].view(-1,1).unsqueeze(0)  
            curr_lp      =   (c, A_ub, b_ub, None, None)
            curr_soln = None
            curr_loss = None
            curr_target_feasibility_ub = None
            curr_target_feasibility_eq = None
        return curr_lp
    
    # Return curr_sol by solving LP(c, A_ub, b_ub, A_eq, b_eq)
    def get_curr_soln(w):  # w is ndarray
        nonlocal curr_soln
        if not is_w_curr(w) or curr_soln is None:
            curr_soln = solve_lp(get_curr_lp(w), 
                                 grad_mode, 
                                 inner_tol, 
                                 nneg,
                                 layer = layer)
        return curr_soln
    
    # get the status code from the homogeneous solver
    def get_curr_status(w):
        return get_curr_soln(w)[1]
        
    def get_curr_loss(w):  # w is ndarray
        nonlocal curr_loss
        if not is_w_curr(w) or curr_loss is None:
            c = get_curr_lp(w)[0]
            x, s, *_, = get_curr_soln(w)
            l = loss_fn(c, x_target, x).view(-1)
            if boundedness_cons == True:
                l.masked_fill_(s >= 1, 1)  # Without this, huge loss makes SLSQP go crazy and it never converges
            curr_loss = l.mean()
        return curr_loss

    # Capture the primal-dual gap
    def get_curr_pdgap_residual(w):  # w is ndarray
        return -get_curr_soln(w)[3] + 1e-5

    # Compute constraint residual, i.e, b - A@x
    def calc_residual(b, A, x): 
        if A is None:
            return None
        # Using _bsubbmm().neg_() avoids temporaries and is slightly faster than (b - bmm(A, x))
        return _bsubbmm(b, A, x).neg_().view(-1)  # -(-b + A@x) == b - A@x

    # Check A_ub @ X <= b_ub
    def get_curr_target_feasibility_ub(w):  # w is ndarray
        nonlocal curr_target_feasibility_ub
        if not is_w_curr(w) or curr_target_feasibility_ub is None:
            _, A_ub, b_ub, _, _ = get_curr_lp(w)
            curr_target_feasibility_ub = calc_residual(b_ub, A_ub, x_target)
        return curr_target_feasibility_ub

    # Check A_eq @ X == b_eq
    def get_curr_target_feasibility_eq(w):  # w is ndarray
        nonlocal curr_target_feasibility_eq
        if not is_w_curr(w) or curr_target_feasibility_eq is None:
            _, _, _, A_eq, b_eq = get_curr_lp(w)
            curr_target_feasibility_eq = calc_residual(b_eq, A_eq, x_target)
        return curr_target_feasibility_eq
    
    # Return feasibility violation
    def get_curr_target_feasibility_violation(w):  # w is ndarray
        r_ub = get_curr_target_feasibility_ub(w)
        r_eq = get_curr_target_feasibility_eq(w)
        r = 0.0
        if r_ub is not None: r = max(r, -float(r_ub.detach().min()))        # Maximum inequality violation
        if r_eq is not None: r = max(r,  float(r_eq.detach().abs().max()))  # Maximum equality violation
        return r

    # Return primal-dual gap violation
    def get_curr_pdgap_violation(w):  # w is ndarray
        r = get_curr_pdgap_residual(w)
        return float(r.detach().max())

    def get_curr_status_codes(w):
        return [int(s) for s in get_curr_status(w).view(-1)]
    
    def zero_w_grad():
        if curr_w.grad is not None:
            curr_w.grad.zero_()
    
    # return jacobian of residual
    def fill_jac(jac, r, start, size):
        for i in range(size):
            zero_w_grad()
            r[i].backward(retain_graph=True)   # Always retain graph, partially re-used by loss.
            jac[start+i,:] = curr_w.grad.data  # Gradient wrt constraint residual i copied into jac
    
    # Function to evaluate loss at point w in weight space
    def f(w):
        nonlocal num_evals, best_loss, best_w

        if num_evals >= outer_maxeval:
            raise EvaluationLimitExceeded()

        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        num_evals += 1
        l = float(get_curr_loss(w))
        
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        # Record the best loss for feasible w, in case we later hit evaluation limit
        if l < best_loss and (get_curr_status(w) == 0 ).all():
            v = get_curr_target_feasibility_violation(w)
            # Record loss only if outer problem constraints are satisfied and fop is feasible
            if v < violation_tol:
                best_loss = float(l)
                best_w = w.copy()     
        loss_collect.append(best_loss)
        if verbose == 2:
            print('   [%d]  f(w[:10]=%s) \n \t  l=[%.8f]  infea=[%.4f]  pd=%s  s=%s'% (num_evals, w.ravel()[:10], l,
                                                get_curr_target_feasibility_violation(w),
                                                get_curr_pdgap_violation(w) if boundedness_cons else None,
                                                get_curr_status_codes(w)))

        if best_loss < loss_tol:
            raise LossToleranceExceeded()
            
        return l

    # Function to evaluate gradient of loss at w
    def f_grad(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] df(w[:10]=%s)'% (num_evals, w.ravel()[:10]))

        zero_w_grad()
        l    =  get_curr_loss(w)
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
        l.backward(retain_graph=True)

        return curr_w.grad.numpy()

    # Function to evaluate ineq. constraint residuals of outer problem at w
    def g(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d]  g(w[:10]=%s)'% (num_evals, w.ravel()[:10]))

        r1 = get_curr_target_feasibility_ub(w).detach().numpy() # x_target is feasible
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        if boundedness_cons == True:
            r2 = get_curr_pdgap_residual(w).detach().numpy() # curr_lp is bounded
            r = np.concatenate((r1, r2)) if r1 is not None else r2
        else:
            r = r1.copy()   

        return r

    # Function to evaluate gradient of ineq. constraint residuals of outer problem at w
    def g_grad(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] dg(w[:10]=%s)'% (num_evals, w.ravel()[:10]))

        r1 = get_curr_target_feasibility_ub(w)
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        m1 = len(r1) if r1 is not None else 0
        if boundedness_cons == True:
            r2 = get_curr_pdgap_residual(w)
            m2 = len(r2)
            jac = np.empty((m1 + m2, len(w)))
            fill_jac(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)
            fill_jac(jac, r2, m1, m2)  # Fill wrt boundedness
        else:
            jac = np.empty((m1, len(w)))
            fill_jac(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)


        return jac

    # Function to evaluate eq. constraint residuals of outer problem at w
    def h(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
        if verbose == 2:
            print('[%d]  h(w[:10]=%s)'% (num_evals, w.ravel()[:10]))

        r = get_curr_target_feasibility_eq(w).detach().numpy()
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        return r

    # Function to evaluate gradient of eq. constraint residuals of outer problem at w
    def h_grad(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
        if verbose == 2:
            print('[%d] dh(w[:10]=%s)'% (num_evals, w.ravel()[:10]))
        r = get_curr_target_feasibility_eq(w)

        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
        jac = np.empty((len(r), len(w)))
        fill_jac(jac, r, 0, len(r))
        return jac

    if collect_evals:
        f = collect_calls(f)
        g = collect_calls(g)
        h = collect_calls(h)
        f_grad = collect_calls(f_grad)
        g_grad = collect_calls(g_grad)
        h_grad = collect_calls(h_grad)

    if outer_method == 'COBYLA':
        f_jac = None
        g_jac = None
        h_jac = None
    elif outer_method == 'SLSQP':
        if grad_mode == 'numerical_grad':
            f_jac = None
            g_jac = None
            h_jac = None
        elif grad_mode in ('backprop', 'implicit', 'direct'):
            f_jac = f_grad
            g_jac = g_grad    
            h_jac = h_grad
        elif grad_mode == 'cvxpylayer':
            f_jac = f_grad
            g_jac = g_grad    
            h_jac = h_grad

            layer = create_cvx_layer((p.requires_grad_() if p is not None else None for p in as_tensor(*lp)), 
                                     nneg = False)
        else:
            raise Exception('Invalid grad_mode for SLSQP')

    c, A_ub, b_ub, A_eq, b_eq = lp
    x_target = tensor(x_target)

    # w comprises all LP coefficients
    w_init = np.hstack([c.ravel(), A_ub.ravel(), b_ub.ravel()])
    best_w         = np.ones(len(w_init))*999

    constraints = [{'type': 'ineq', 'fun': g, 'jac': g_jac}]
    if A_eq:
        constraints.append({'type': 'eq', 'fun': h, 'jac': h_jac})

    io_tic = time.time()
    try:
        res = opt.minimize(f, w_init, jac=f_jac, 
                           constraints=constraints, 
                           tol=outer_tol,
                           method=outer_method,
                           options = {'maxiter': outer_maxiter})
        if not res.success:
            if verbose >=1:
                print("WARNING: scipy minimize returned failure:")
        
    except (EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded, UnboundedWeightsValue) as e:
        if verbose >=1:
            print("EXCEPTION earily terminaton [%s]"%e.__class__.__name__)
        if collect_evals and isinstance(e, LossToleranceExceeded):
                assert len(f.rvals) == len(f.calls)-1
                f.rvals.append(best_loss)
    io_toc = time.time()
    
    if len(loss_collect) > 0:
        assert best_loss == loss_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_loss, loss_collect[-1])
    else:
        loss_collect = [999]

    if verbose >= 1:
        print("++++           call [f %d times], [f_grd %d times], [g %d times], [g_grd %d times]" \
                    % (len(f.calls), len(f_grad.calls), len(g.calls), len(g_grad.calls)) )
        print("++++           initial_loss [%.8f], best_loss [%.8f], runtime [%.4f s]"%(loss_collect[0], best_loss, (io_toc-io_tic)))
        if len(best_w.ravel()) > 10:
            print("++++           best_w is to long, only print first 10 elements\n %s"%(best_w.ravel()[:10]))
        else:
            print("++++           best_w %s"%(best_w.ravel()))

    if collect_evals:
        evals = {
            'f': {'calls': f.calls, 'rvals': f.rvals, 'times': f.times, 'loss': loss_collect},
            'g': {'calls': g.calls, 'rvals': g.rvals, 'times': g.times},
            'h': {'calls': h.calls, 'rvals': h.rvals, 'times': h.times},
            'f_grad': {'calls': f_grad.calls, 'rvals': f_grad.rvals, 'times': f_grad.times},
            'g_grad': {'calls': g_grad.calls, 'rvals': g_grad.rvals, 'times': g_grad.times},
            'h_grad': {'calls': h_grad.calls, 'rvals': h_grad.rvals, 'times': h_grad.times},
            'res': {'runtime': (io_tic, io_toc), 'best_l': best_loss, 'best_w':best_w},
        }
        return best_loss, evals
    
    return best_loss

def InvLP_experiment(nVar, 
                     nCons, 
                     ind_ins, 
                     inner_tol         =  1e-8,
                     outer_method      =  None,
                     grad_mode         =  None,
                     runtime_limit     =  30,
                     directory         =  None,
                     verbose           =  0):

    """
    run ILOP experiment for a specific inverse LP instance

    Input:
          nVar, nCons: num. variabel, num. constriants,
              ind_ins:  instance index
            inner_tol: tolerance of forward solver
         outer_method: algorithms for solving the bi-level ILOP
            grad_mode: backward methods for computing gradients
        runtime_limit: runtime limit for solving ILOP (in seconds)
    """

    LP_filename = directory + '%dvar%dcons_InvLP%d.txt'\
            %(nVar, nCons, ind_ins)
    
    c, A_ub, b_ub, x_target, target_type = read_InvLP(LP_filename)
    lp = (c[None, :, :], A_ub[None, :, :], b_ub[None, :, :], None, None)

    if verbose >=1:
        print('************************************')
        print('ILOP Exp1c %dvar%dcon ins%d '%(nVar, nCons, ind_ins) )
        if outer_method =='RS':
            print('\twith Random Search [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
        else:
            if outer_method == 'COBYLA':
                print('\twith COBYLA [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
            elif outer_method == 'SLSQP':
                print('\twith SLSQP [%s] %s(tol = %.2e), '
                        %(grad_mode, 'cvx' if grad_mode == 'cvxpylayer' else 'homogeneous', inner_tol))

        print('\tloss_fn = [AOE]' )

    l_best, res = inverse_linprog(nVar             =  nVar,
                                  nCons            =  nCons, 
                                  lp               =  lp,
                                  x_target         =  x_target[None,:,:], 
                                  inner_tol        =  inner_tol,
                                  outer_method     =  outer_method,
                                  grad_mode        =  grad_mode,
                                  runtime_limit    =  runtime_limit,
                                  verbose          =  verbose,
                                  collect_evals    =  True)
                    
    if outer_method == 'SLSQP':
        save_file_directory = directory + '%s_%s'%(outer_method, grad_mode)      
    elif outer_method == 'COBYLA':
        save_file_directory = directory + '%s'%(outer_method)
    
    if not os.path.exists(save_file_directory):
        os.makedirs(save_file_directory)

    _file_record = save_file_directory + '/%dvar%dcons_InvLP%s_%s_res.pkl'\
                %(nVar, nCons, ind_ins, 'AOE')
    with open(_file_record, 'wb') as output:
        pickle.dump(res, output)

    if verbose >= 1:
        print('************************************\n')

def InvLP(lp_size,
          nIns,
          runtime_limit, 
          framework,
          directory,
          verbose):
    """
    Complete experiment for inverse LP problem with ineq. constraints only.

    Input:
          lp_size:  a list of tuple (nVar, nCons)
             nIns: number of instances for each lp_size
    runtime_limit: runtime limit for solving the IO problem
        framework: a list of tuple(outer_method, grad_mode)
    """
    for framework_i in framework:
        outer_method, grad_mode = framework_i
        for _lp in lp_size:
            nVar, nCons    =  _lp
            for ind_ins in range(nIns):
                direct = directory + 'InvLP_Ins/%dvar%dcons/'%(nVar, nCons)

                InvLP_experiment(nVar, nCons, ind_ins,
                                inner_tol        =  1e-8,
                                outer_method     =  outer_method,
                                grad_mode        =  grad_mode,
                                runtime_limit    =  runtime_limit,
                                directory        =  direct,
                                verbose          =  verbose)

def read_exp_result(file):
    """
    read from result file (in pkl format) and return the following data:

    w_collect: collection of parameter w at each iteration
    l_collect: collection of loss l at each iteration
     timestep: time steps of each iteration 
      runtime: Total runtime
       best_l: best loss
       best_w: parameter w associated with the best_l, 
               Note, w comprises all LP coefficients, w =[c, A_ub, b_ub]
    """
    with open(file+'_res.pkl', 'rb') as pkl_file:
        res = pickle.load(pkl_file)
    
    w_collect  =   np.vstack(as_numpy(res['f']['calls']))
    if len(res['f']['loss'])>0:
        l_collect  =   np.vstack(res['f']['loss'])
    else:
        l_collect = np.array([999])
    
    io_tic = res['res']['runtime'][0]
    if len(res['f']['times']) >0:
        
        timestep   =   np.vstack([t[-1]-io_tic for t in res['f']['times']]) # end time (clock time) of each f 
                                                                            #       = end time of f - start time of the algorithm
    else:
        timestep = np.array([0])
    runtime    =   float(res['res']['runtime'][1] - res['res']['runtime'][0])   # total runtime
    best_l     =   res['res']['best_l']  # best loss
    best_w     =   res['res']['best_w']  # best weights

    return w_collect, l_collect, timestep, runtime, best_l, best_w


def InvLP_data_analysis(lp_size, 
                         nIns,
                         framework,
                         directory     =  None,
                         plot_callback =  None,
                         verbose       =  False # True - print log of data analysis
                         ):
    """
    read experiment results from pkl files.  
    
    Input:
          lp_size:  a list of tuple (nVar, nCons)
             nIns: number of instances for each lp_size
        framework: a list of tuple(outer_method, grad_mode)

    Output 
        (l, w, timestep, runtime): a tuple for each method
                   l: collection of loss at each iteration
                   w: collection of w at each iteration
            timestep: collection of clock time of each iteration
              runtim: total runtime
        Note, each of l, w, timestep, runtime contains results for len(lp_size)*nIns instances.
    """

    if verbose >1:
        print('\n+++++ collect experiment results for analysis')
    ins_target_type          =  np.zeros((len(lp_size), nIns))

    # loss collection of each method
    l_COBYLA            =  []
    l_SLSQP_backprop    =  []
    l_SLSQP_implicit    =  []
    l_SLSQP_direct      =  []
    l_SLSQP_cvxpylayer  =  []

    # clock time of each f(w)
    timestep_COBYLA           =  []
    timestep_SLSQP_backprop   =  []
    timestep_SLSQP_implicit   =  []
    timestep_SLSQP_direct     =  []
    timestep_SLSQP_cvxpylayer =  []

    # total runtime
    runtime_COBYLA           =  []
    runtime_SLSQP_backprop   =  []
    runtime_SLSQP_implicit   =  []
    runtime_SLSQP_direct     =  []
    runtime_SLSQP_cvxpylayer =  []



    for ind_lp in range(len(lp_size)):
        nVar, nCons = lp_size[ind_lp]
        for i in range(len(framework)):
            outer_method, grad_mode = framework[i]

            if outer_method == 'COBYLA':
                _file_path = directory + 'InvLP_Ins/%dvar%dcons/%s'\
                                          %(nVar, nCons, outer_method)
            elif outer_method == 'SLSQP':
                _file_path = directory + 'InvLP_Ins/%dvar%dcons/%s_%s'%(nVar, nCons, outer_method, grad_mode)

            for ind_ins in range(nIns):
                if verbose > 1:
                    print('\n=======  %d var, %d cons, Ins %d, loss = %s '
                                    %(nVar, nCons, ind_ins, 'AOE'))
                    print('         Outer_method %s, grad_mode [%s]'
                                    %(outer_method, grad_mode))
                
                result_file = _file_path+'/%dvar%dcons_InvLP%d_%s'\
                        %(nVar, nCons, ind_ins, 'AOE')

                w_collect, l_collect, timestep, runtime, best_l, best_w = read_exp_result(result_file)
                assert best_l == l_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_l == l_collect[-1])

                # runtime is recorded when f(w) is called, even f(w) triggered early termination due to other exceptions
                num_evals = min((len(w_collect), len(l_collect), len(timestep)) )

                w_collect = w_collect[:num_evals]
                l_collect = l_collect[:num_evals]
                timestep  = timestep[:num_evals]
                
                if l_collect[0] > 900:   # infeasible target will have initial loss of 999
                    target_type = 0
                else:                    # feasible target will have a initial loss value
                    target_type = 1
                    
                ins_target_type[ind_lp, ind_ins] = int(target_type) # record target type for plotting later

                if outer_method == 'COBYLA':
                    l_COBYLA.append(l_collect)                  # loss_collect at each f(w)
                    timestep_COBYLA.append(np.array(timestep))  # clock time of each f(w)
                    runtime_COBYLA.append(runtime)              # total runtime
                if outer_method == 'SLSQP':
                    if grad_mode == 'backprop':
                        l_SLSQP_backprop.append(l_collect)
                        timestep_SLSQP_backprop.append(np.array(timestep))
                        runtime_SLSQP_backprop.append(runtime)    
                    elif grad_mode == 'implicit':
                        l_SLSQP_implicit.append(l_collect)
                        timestep_SLSQP_implicit.append(np.array(timestep))
                        runtime_SLSQP_implicit.append(runtime) 
                    elif grad_mode == 'direct':
                        l_SLSQP_direct.append(l_collect)
                        timestep_SLSQP_direct.append(np.array(timestep))
                        runtime_SLSQP_direct.append(runtime) 
                    elif grad_mode == 'cvxpylayer':
                        l_SLSQP_cvxpylayer.append(l_collect)
                        timestep_SLSQP_cvxpylayer.append(np.array(timestep))
                        runtime_SLSQP_cvxpylayer.append(runtime)

                if verbose > 1:
                    print('initial loss [%.3e], final loss [%.3e] '%(l_collect[0], l_collect[-1]) )
                    print('final w (first 10 elements)', best_w[:10])        

    return  ins_target_type, \
            (l_COBYLA,           timestep_COBYLA,           runtime_COBYLA), \
            (l_SLSQP_backprop,   timestep_SLSQP_backprop,   runtime_SLSQP_backprop),\
            (l_SLSQP_implicit,   timestep_SLSQP_implicit,   runtime_SLSQP_implicit),\
            (l_SLSQP_direct,     timestep_SLSQP_direct,     runtime_SLSQP_direct),\
            (l_SLSQP_cvxpylayer, timestep_SLSQP_cvxpylayer, runtime_SLSQP_cvxpylayer)

def boxplot_InvLP(ax, 
                 lp_size, 
                 loss_fn, 
                 nIns,
                 end_time, 
                 data, 
                 directory, 
                 verbose = False):
    """
    plot boxplot of loss.  

    Input:
         lp_size: a list of tuple (nVar, nCons)
         loss_fn: function for evluation loss
            nIns: num. of instances
        end_time: plot loss at the end time
            data: a nested list (methods, result_data, instance)
                        methods: RS, COBYLA, SQP_cvx, SQP_bprop, SQP_impl, SQP_dir, SQP_cvx
                    result_data: loss_at_each_iter, w_at_each_iter, clockTime_at_each_iter
                       instance: 100 instances for each lp_size
    """
    
    numIns = len(data[0][0])
    numMethod = len(data)
    loss_initial = np.ones((numIns,))*999  # collect initial loss
    loss_train = np.ones((numMethod, numIns))*999  # collect initial loss

    # collect loss_init and loss_final
    for ind_ins in range(numIns):
        for outer_method_i in range(numMethod-1):
            #                    method            l   ins     init_l
            ini_l_1 = float(data[outer_method_i  ][0][ind_ins][ 0 ])
            ini_l_2 = float(data[outer_method_i+1][0][ind_ins][ 0 ])
            
            if outer_method_i != 1 and outer_method_i+1 != 1:
                # sanity check if COBYLA, SQP_bprop, SQP_implic and SQP_close have the same initial loss
                # SQP_cvx uses different forward solver which will have different initial loss, thus, it is 
                # not included in this sanity check.
                assert ini_l_1 == ini_l_2, 'ind_ins[%d], outer_method %d [%.5f] and %d [%.5f] have different initial loss'%(ind_ins, outer_method_i, ini_l_1, outer_method_i+1, ini_l_2)
            
        l_init_ = float(data[0][0][ind_ins][0])
        loss_initial[ind_ins] = 100 if l_init_ >= 999 else l_init_
        
        for outer_method_i in range(numMethod):
            loss_collect    = data[outer_method_i][0][ind_ins].copy()
            time_collect    = data[outer_method_i][2][ind_ins].copy()
            
            idx = np.where(time_collect <= end_time)[0] # find all iter that finished before reaching runtime_limit
            if len(idx) > 0:
                assert len(time_collect) == len(loss_collect), 'len(time_collect) = %d,  len(loss_collect) = %d'%(len(time_collect), len(loss_collect))
                ind = max(np.where(time_collect <= end_time)[0])  
                l_train = loss_collect[ind] 
                loss_train[outer_method_i][ind_ins] = float(l_train) 
            else:
                loss_train[outer_method_i][ind_ins] = float(999) 

            if verbose > 1:
                print('init [%.5f], train [%.5f]'%(loss_initial[ind_ins], l_train))

    if verbose > 1:
        print('num l_train <= 1e-5: ', [sum(loss_train[i,:]<=1e-5) for i in range(len(loss_train))])

    loss_initial  = np.array(loss_initial)
    loss_initial = np.clip(loss_initial, 1e-5, 1e2)
    loss_train = np.array(loss_train)
    loss_train = np.clip(loss_train, 1e-5, 1e2)
    
    loss_plot = [loss_initial]+[loss_train[i,:] for i in range(len(loss_train))]
    
    ax.boxplot(loss_plot, sym='')
    x_ticks = np.linspace(0.7, 1.3, nIns)
            # initial, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
    marker = ['+',      'x',     'd',     '^',       's',        'o', ]
    color  = ['gray',      'b',     'k',      'r',      'orange',   'g', ]
    

    for i in range(len(loss_plot)):
        ax.plot(x_ticks+i, loss_plot[i], 'o', 
                marker=marker[i], mfc='None',
                ms=5, mew=1, color = color[i],
                alpha = 0.5,)
    ax.set_yscale('log')
    ax.set_ylim(5e-6,5e2)
    ax.get_xaxis().set_visible(False)
    ax.set_xlim(0,len(loss_plot)+1)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    return loss_initial, loss_train


def fig_exp1c(lp_size, 
              nIns, 
              loss_fn, 
              framework,
              directory, 
              verbose = False):
    time_range = [np.arange(1e-3, 5, 1e-3).round(3),    #   time steps for 2D.
                np.arange(1e-3, 30, 1e-3).round(3)]   #   time steps for 10D

    ins_target_type, res_COBYLA, res_SLSQP_backprop,\
    res_SLSQP_implicit, res_SLSQP_direct,\
    res_SLSQP_cvxpylayer  = InvLP_data_analysis(lp_size       =  lp_size,
                                                nIns          =  nIns,
                                                framework     = framework,
                                                directory     =  directory,
                                                verbose       =  verbose)

    def subplot(res_c, 
                res_s_backprop, 
                res_s_implicit, 
                res_s_direct, 
                res_s_cvxpylayer,
                target_type, 
                time_range):
        """
        create plots for each lp_size
        """
        success_rate = []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
        res_ = plot_success_rate(time_range = time_range,
                                  res = [ res_c[0], None, res_c[1], None], 
                                  ax = ax1,
                                  plot_style = ['b', '-', 'COBYLA'])
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                          res = [ res_s_cvxpylayer[0], None, res_s_cvxpylayer[1], None],
                          ax = ax1,
                          plot_style = ['k', '-', 'SQP_cvx'])
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                                  res = [ res_s_backprop[0], None, res_s_backprop[1], None],
                                  ax = ax1,
                                  plot_style = ['r', '-', 'SQP_bprop'])   
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                                  res = [ res_s_implicit[0], None, res_s_implicit[1], None],
                                  ax = ax1,
                                  plot_style = ['orange', '-', 'SQP_impl'])   
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                                  res = [ res_s_direct[0], None, res_s_direct[1], None],
                                  ax = ax1,
                                  plot_style = ['g', '-', 'SQP_dir'])   
        success_rate.append(res_[-1])

        if verbose > 1:
            print('success rate: %s'%(success_rate))

        ax1.tick_params( axis='x', which='major', labelsize=12 )
        ax1.set_xlim(-0.1, time_range[-1]+0.1)
        ax1.set_ylim(-5, 105)
        ax1.set_yticks(np.linspace(0,100,5))
        ax1.set_yticklabels([str(i)+'%' for i in np.linspace(0,100,5, dtype = np.int)])
        
        _ = boxplot_InvLP( ax = ax2, 
                            lp_size = None, 
                            nIns    =  nIns,
                            end_time = time_range[-1], 
                            data = ([ res_c[0],             None, res_c[1],             None],
                                    [ res_s_cvxpylayer[0],  None, res_s_cvxpylayer[1],  None], 
                                    [ res_s_backprop[0],    None, res_s_backprop[1],    None], 
                                    [ res_s_implicit[0],  None, res_s_implicit[1],  None], 
                                    [ res_s_direct[0], None, res_s_direct[1], None]), 
                            loss_fn = loss_fn,
                            directory = directory,
                            verbose   = verbose)
        import matplotlib
        locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
        ax2.yaxis.set_major_locator(locmaj)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax2.yaxis.set_minor_locator(locmin)
        ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.tick_params(axis='x', which='minor', size = 12)
    
        return fig, (ax1, ax2)
    
    
    for fig_i in range(len(lp_size)): # for each class
        if lp_size[fig_i][0] == 2:
            t_range = time_range[0]    #   time steps for 2D.
        elif lp_size[fig_i][0] == 10:
            t_range = time_range[-1]    #   time steps for 10D.
        else:
            print('ERROR: invalid lp_size [%dvar, %dcons]'%(lp_size[fig_i][0], lp_size[fig_i][1]))
            raise ValueError()

        if verbose >=1:
            print('====== Analyze experiment results and plot figure for Exp1b %dvar%dcons'%(lp_size[fig_i][0], lp_size[fig_i][1]))
            
        fig, _ = subplot([data_i[fig_i*nIns:(fig_i+1)*nIns] for data_i in res_COBYLA],
                         [data_i[fig_i*nIns:(fig_i+1)*nIns] for data_i in res_SLSQP_backprop], 
                         [data_i[fig_i*nIns:(fig_i+1)*nIns] for data_i in res_SLSQP_implicit], 
                         [data_i[fig_i*nIns:(fig_i+1)*nIns] for data_i in res_SLSQP_direct],
                         [data_i[fig_i*nIns:(fig_i+1)*nIns] for data_i in res_SLSQP_cvxpylayer], 
                         ins_target_type[fig_i],
                         time_range = t_range)
        if directory:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.subplots_adjust(wspace=0.02, hspace=0)
            fig.savefig(directory+'fig_exp1c_%dvar%dcons.pdf'%(lp_size[fig_i]),dpi=100)


def main(directory = None):

    direct = './' if directory is None else directory

    LP_SIZE             =  (10,80)      # 6 classes of LPs
    NUM_INS             =  100    
    LOSS_FN             =  _AOE_         # loss function for IO evaluation

    VERBOSE         =  0    # 2 - print detailed information, e.g., log of each iter in the solving process, 
                            #     initialization, etc. This includes many printing functions during 
                            #     the solving process, thus, using VERBOSE = 2 will influence the experiment results. 
                            # 1 - print basic information, e.g., final results, initialization. 
                            #     all printing function  happen outside of the solving process, 
                            #     thus, does not influence the experiment results 
                            # 0 - don't display any algorithm log         
    RUN_TIME            =  30             # runtime limit for 2D and 10D

    #run the follow cell for a complete InvPLP experiment
    FRAMEWORK       =( # outer_method, grad_mode,    
                        ['COBYLA',      None], 
                        ['SLSQP',      'cvxpylayer'],
                        ['SLSQP',      'backprop'], 
                        ['SLSQP',      'implicit'],
                        ['SLSQP',      'direct'],
                       )

    nVar, nCons = LP_SIZE
    dir_ = direct + 'LP_baseline/%dvar%dcons'%(nVar, nCons)
    if not os.path.exists(dir_):
        convexhull_generator(nVar, nCons, NUM_INS)

    generate_InvLP(lp_size          =  ((nVar, nCons),),
                    nIns            =  NUM_INS,
                    directory       =  direct,
                    verbose         =  VERBOSE)

    InvLP(lp_size           = ((nVar, nCons),),
          nIns              = NUM_INS,
          runtime_limit     = RUN_TIME, 
          framework         = FRAMEWORK,
          directory         = direct,
          verbose           = VERBOSE)

    fig_exp1c(lp_size      = ((nVar, nCons),),
              nIns         = NUM_INS,
              loss_fn      = LOSS_FN, 
              framework    = FRAMEWORK,
              directory    = direct,
              verbose      = VERBOSE)

if __name__ == "__main__":
        

    main()