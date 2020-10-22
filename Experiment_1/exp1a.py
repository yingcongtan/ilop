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

from linprog_solver import linprog_scipy_batch

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
DT = torch.float64

def record_InvPLP(filename, 
                  LP, 
                  weights_true, 
                  w_A_coef, 
                  u):
    """
    Save PLP instance in txt file in the following format:
    nCons, nVar  # num of rows and columns
    c                 (nVar, 1), base parameter of lp
    A_ub              (nCons, nVar), base parameter of lp
    b_ub              (nCons, 1), base parameter of lp
    
    w_c_true          true weights, used to generate InvPLP
    w_A_true          true weights, used to generate InvPLP
    w_b_true          true weights, used to generate InvPLP
    Observations      Observation u's used for generate InvPLP from LP
    """
    c, A_ub, b_ub = LP
    w_c_true, w_A_true, w_b_true = weights_true

    with open(filename, 'w') as LP_file:
        LP_file.write("LP Size\n")
        np.savetxt(LP_file, np.array(A_ub.shape).reshape((1,2)), fmt="%.d")
        LP_file.write("\n")

        LP_file.write("c\n")
        np.savetxt(LP_file, c, fmt="%.6f")
        LP_file.write("\n")
        LP_file.write("A_ub\n")
        np.savetxt(LP_file, A_ub, fmt="%.6f")
        LP_file.write("\n")
        LP_file.write("b_ub\n")
        np.savetxt(LP_file, b_ub, fmt="%.6f")
        LP_file.write("\n")        

        LP_file.write("w_A_coeffcient\n")
        np.savetxt(LP_file, w_A_coef.ravel(), fmt="%.6f")
        LP_file.write("\n")
        
        LP_file.write("w_c_true\n")
        np.savetxt(LP_file, w_c_true.ravel(), fmt="%.6f")
        LP_file.write("\n")
        LP_file.write("w_A_true\n")
        np.savetxt(LP_file, w_A_true.ravel(), fmt="%.6f")
        LP_file.write("\n")        
        LP_file.write("w_b_true\n")
        np.savetxt(LP_file, w_b_true.ravel(), fmt="%.6f")
        LP_file.write("\n")
        
        LP_file.write("training sample\n")
        np.savetxt(LP_file, u[0].ravel(), fmt="%.2f")
        LP_file.write("\n")
        LP_file.write("testing sample\n")
        np.savetxt(LP_file, u[1].ravel(), fmt="%.2f")
        LP_file.write("\n")

def read_InvPLP(filename, 
                verbose = 0):
    """
    read PLP instance from txt file and return the following data:
    (c, A_ub, b_ub):    baseline linear program
                        A_ub.shape = m (num. constriants), n (num. variables)

    (w_c_true, w_A_true, w_b_true): True parameters of the parametric LPs
    w_A_coeffcient:                 a list of index, len(w_A_coeffcient) = m
                                    indicating which element of each row has the parametric term w@u

    (u_train, u_test):  observable features splitted into training and testing sets.

    """
    assert verbose in [0,1,2], 'invalid value for verbose\n verbose = 0\tdont show alg log\n, verbose = 1\tshow the final alg res, verbose = 2\tshow details of alg log'

    lines=[]
    with open(filename, "rt") as fp:
        for line in fp:
            lines.append(line.rstrip('\n')) 
    
    nCons, nVar = np.fromstring(lines[1], dtype=float, sep=' ').astype(int)

    if verbose == 2: print('Reading from %s\n[num constraint %d], [num variables %d]'%(filename, nCons, nVar))
        
    temp = [lines[i+1:i+nVar+1] for i in range(len(lines)) if "c" in lines[i] and 'w' not in lines[i]]
    c = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ c\n\t',c.ravel())
        
    temp = [lines[i+1:i+nCons+1] for i in range(len(lines)) if "A_ub" in lines[i]]
    A_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ').reshape(1,-1) for row in temp[0]])
    if verbose == 2: print('+++ A_ub\n\t', A_ub)

    temp = [lines[i+1:i+nCons+1] for i in range(len(lines)) if "b_ub" in lines[i]]
    b_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ b_ub\n\t', b_ub.ravel())

    
    temp = [lines[i+1:i+1+2] for i in range(len(lines)) if "w_c_true" in lines[i]]
    w_c_true = np.hstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ w_c_true\n\t',w_c_true.ravel())
    
    temp = [lines[i+1:i+1+2] for i in range(len(lines)) if "w_A_true" in lines[i]]
    w_A_true = np.hstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])
    if verbose == 2: print('+++ w_A_true\n\t',w_A_true.ravel())

    temp = [lines[i+1:i+1+2] for i in range(len(lines)) if "w_b_true" in lines[i]]
    w_b_true = np.hstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]])    
    if verbose == 2: print('+++ w_b_true\n\t',w_b_true.ravel())
    
    line_train = [ int(i) for i in range(len(lines)) if "training sample" in lines[i]][0]
    line_test  = [ int(i) for i in range(len(lines)) if "testing sample" in lines[i]][0]

    temp    = lines[line_train+1: line_test-1]
    u_train = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp ])    
    temp    = lines[line_test+1:-1]
    u_test  = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp ])    
    if verbose == 2: 
        print('+++ u_train\n\t',u_train.ravel())
        print('+++ u_test\n\t',u_test.ravel())

    temp = [lines[i+1:i+nCons+2] for i in range(len(lines)) if "w_A_coeffcient" in lines[i]]
    w_A_coeffcient = np.hstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]]).astype(int)     
    if verbose == 2: print('+++ w_A_coeffcient\n\t',w_A_coeffcient.ravel())
    
    return (c, A_ub, b_ub), \
            (w_c_true, w_A_true, w_b_true), \
            w_A_coeffcient, (u_train, u_test)

def read_exp_result(file):
    """
    read from result file (in pkl format) and return the following data:

    w_collect: collection of parameter w at each iteration
    l_collect: collection of loss l at each iteration
    timestep:  time steps of each iteration 
    runtime:   Total runtime
    best_l:    best loss
    best_w:    parameter w associated with the best_l
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

# Class defining the corresponding parametric linear programs
class parametric_lp_vectorized(ParametricLP):
    """
    vectorized code for generating parametric LPs
    """
    def __init__(self, LP, weights, nidx_A_ub):
        super().__init__(weights)
        c, A_ub, b_ub  = LP
        m, n = A_ub.shape
        self.ref_c     = c.view(1, -1)
        self.ref_A_ub  = A_ub.unsqueeze(0)
        self.nidx_A_ub = nidx_A_ub.view(1, m)
        self.midx_A_ub = torch.arange(m).view(1, m)
        self.ref_b_ub  = b_ub.view(1, -1)
        self.nneg      = False
                
    def generate(self, u, weights):
        k = len(u)
        _, m, n = self.ref_A_ub.shape 
        w1, w2, w3, w4, w5, w6 = weights.view(-1, 6, 1).unbind(1)

        # Batch of (k, n, 1) c vectors
        plp_c = (self.ref_c + w1 + w2*u).unsqueeze(2)
        plp_c = normalize_c(plp_c)

        # Batch of (k, m, n) A_ub matrices
        kidx = torch.arange(k).view(k, 1).expand(-1, m)
        midx = self.midx_A_ub.expand(k, -1)
        nidx = self.nidx_A_ub.expand(k, -1)
        plp_A_ub = self.ref_A_ub.repeat(k, 1, 1).contiguous()
        plp_A_ub[kidx, midx, nidx] += (w3 + w4*u).expand(-1, m)  # Expand m to make same for each row of A
        
        # Batch of (k, m, 1) b_ub vectors
        plp_b_ub = (self.ref_b_ub + w5 + w6*u).unsqueeze(2)
        
        return plp_c, plp_A_ub, plp_b_ub, None, None

def _generate_InvPLP(ind_ins, 
                     nVar, 
                     nCons, 
                     directory   =  None,
                     verbose     =  False):
    """
    ind_ins: instance index
       nVar: num. of variables
      nCons: num. of constraints

    generating parametric LPs for the experiment through the following steps:
    1).  read a baseline convexhull from file (baseline LPs are generated with convexhull_generator, please see generator.py for more details)
    2).  sample cost vector c to have a complete baseline LP
    3).  sample a set of feature u
    4).  given a set of feature u, repeatedly sample parameter w s.t. resulting LPs are feasible
    5).  if step 4 failed 50 times, redo step 3 - 4.
    6).  save the PLP to txt file (see record_InvPLP for more details)
    """
    sigma_w          = 1        # used for sample w_true
    sigma_c          = 1.0      # used for sample c_true
    u_range          = 1.0      # used for sample u
    num_u            = 40       # num. of u values to sample
    train_test_ratio = 1        # ratio to split u values into training and testing sets

    def split_u(u_collect, train_test_ratio = train_test_ratio):

        test_portion = 1
        train_portion = train_test_ratio
        split_point = len(u_collect)*train_portion//(train_portion+test_portion)

        i_      =  np.random.permutation(len(u_collect))
        u_train =  u_collect[ i_[: split_point   ] ]
        u_test  =  u_collect[ i_[  split_point : ] ]

        return u_train, u_test, i_

    lp_file = directory + 'LP_baseline/%dvar%dcons/%dvar%dcons_LP%d.txt'%(nVar, nCons, nVar, nCons, ind_ins)
    A_ub, b_ub, _ = read_baseline_convexhull(lp_file)
    if verbose == True:
        print('==== read base_lp from ', lp_file)

    # baseline_lp only has the convexhull (i.e., A_ub, and b_ub) and vertices
    # generate random c to be used together with A_ub and b_ub as base lp parameters
    if verbose == True: 
        print('==== generate true c ~U(-sigmam, sigma)  with np.random.seed(%d)'%ind_ins)
    np.random.seed(ind_ins)
    c   = np.random.uniform(-sigma_c, sigma_c, size=(nVar,1)).round(6)

    # sample u_collect
    if verbose == True: 
        print('==== u_collect ~U(-u_range, u_range) with np.random.seed(%d)'%ind_ins)
    np.random.seed(ind_ins)
    u_collect = np.empty(0)
    i_u = 1
    while i_u < num_u:
        u = np.random.uniform(-u_range, u_range, 1).round(2)
        if len(u_collect) == 0:
            u_collect = u
        elif min(abs(u_collect - u))>= 1e-2:
                u_collect = np.vstack((u_collect, u))
                i_u += 1

    assert len(u_collect) == 40, 'len(u_collect) = %d'%len(u_collect)
    u_train, u_test, i_ = split_u(u_collect, train_test_ratio)
    assert len(u_train)/len(u_test) == train_test_ratio, 'WARNING: traint_test_ratio: %d, len(u_train) = %d, len(u_test) = %d'%(train_test_ratio,len(u_train), len(u_test)) 

    # Randomly generate the true parameter values
    if verbose == True: 
        print('==== generate true weights with np.random.seed(%d)'%ind_ins)    
        print('\t w_c = [0, uniform[-sigma_w, sigma_w]')    
        print('\t w_A = [0, uniform[       0, sigma_w]')    
        print('\t w_b = [0, uniform[       0, sigma_w]')  

    s = np.ones(len(u_collect))    # initialize s
    np.random.seed(ind_ins)
    trail = 0
    while sum(s) != 0:
        w_c_true = np.hstack( (np.zeros(1), 
                            np.random.uniform(-sigma_w, sigma_w, 1)) 
                            ).round(6)
        w_A_true = np.hstack( (np.zeros(1), 
                            np.random.uniform(0, sigma_w, 1)) 
                            ).round(6)
        w_b_true = np.hstack( (np.zeros(1), 
                            np.random.uniform(0, sigma_w, 1)) 
                            ).round(6)
        w_A_coef = np.random.randint(0, nVar, nCons)

        plp_true = parametric_lp_vectorized(as_tensor(c.copy(), A_ub.copy(), b_ub.copy()), 
                                            torch.cat(as_tensor(w_c_true, w_A_true, w_b_true)).view(-1), 
                                            torch.LongTensor(w_A_coef))

        c_, A_ub_, b_ub_, A_eq_, b_eq_ = plp_true.generate(tensor(u_collect).view(-1,1), plp_true.weights)
        soln_true    = homogeneous_solver(c_, A_ub_, b_ub_, A_eq_, b_eq_, tol = 1e-8)
        s = soln_true[1].detach().numpy()
        if verbose == True:  print('\t[trail %d] s = %s'%(trail+1, [int(s_) for s_ in s] ) )
        trail += 1   

        if trail == 50:
            if verbose == True: print('\t !!!Failed to find feasible w, sample a new set of u values')
            u_collect = np.empty(0)
            i_u = 1
            while i_u < num_u:
                u = np.random.uniform(-u_range, u_range, 1).round(2)
                if len(u_collect) == 0:
                    u_collect = u
                elif min(abs(u_collect - u))>= 1e-2:
                        u_collect = np.vstack((u_collect, u))
                        i_u += 1

            assert len(u_collect) == 40, 'len(u_collect) = %d'%len(u_collect)
            u_train, u_test, i_ = split_u(u_collect, train_test_ratio)
            assert len(u_train)/len(u_test) == train_test_ratio, 'WARNING: traint_test_ratio: %d, len(u_train) = %d, len(u_test) = %d'%(train_test_ratio,len(u_train), len(u_test)) 
            
            trail   = 0

    x_target = soln_true[0].detach().numpy()

    if verbose == True:
        print('\tw_ture = ', plp_true.weights.detach().numpy())
        print('\tu_train = ', u_train )
        print('\tu_test  = ', u_test )

    dir_ = directory + 'InvPLP_Ins/%dvar%dcons/'%(nVar, nCons)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    if nVar == 2:
        alpha = 0.4
        plp_plot  = deepcopy(plp_true)

        fig, ax1 = plt.subplots(1,1,figsize=(10,10))
        u_plot = u_collect[i_[:10]].copy()
        c_, A_, b_, *_ = plp_plot.generate(tensor(u_plot).view(-1,1), 
                                              plp_true.weights)
        for i in range(0, len(u_plot)):
            alpha_i = alpha + (1-alpha)*(i/(len(u_plot)-1) if len(u_plot) > 1 else 1)
            plot_linprog_hull(c_[i], A_[i], b_[i], 
                            alpha_i, c_color = 'k', cons_color='g', axes = ax1)
        
        ax1.plot(*x_target[i_[:10]].squeeze(-1).T, linestyle = 'None',
                marker = 'o', mfc='None', mec='orange',
                ms =15, mew=3, alpha=0.5)
        ax1.set_title('True InvPLP [10 obser.] - %d var %d cons, Ins %d'%(nVar, nCons, ind_ins), fontsize=14)

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        plt.ioff()
        fig.savefig(dir_+'%dvar%dcons_InvPLP%d.png'%(nVar, nCons, ind_ins), dpi=100)
        plt.close()

    filename = dir_ + '%dvar%dcons_InvPLP%d.txt'%(nVar, nCons, ind_ins)
    if verbose == True:
        print('==== record result in %s'%filename)
    record_InvPLP(filename, 
                (c, A_ub, b_ub), 
                (w_c_true, w_A_true, w_b_true), 
                w_A_coef, (u_train, u_test) )

def generate_InvPLP(lp_size,
                     nIns,
                     directory,
                     verbose):
    """
    call _generate_InvPLP to generate PLP of different size

    LP_size: a list of tuple (nVar, nCons)
    nIns: number of instances to generate
    """
    nVar, nCons = lp_size
    for ind_ins in range(nIns):
        if verbose >=1:
            print('==== Generate parametric LPs for Exp 1.a [%d var, %d cons, ins %d]'%(nVar, nCons, ind_ins))
        _generate_InvPLP(ind_ins   = ind_ins, 
                        nVar       = nVar, 
                        nCons      = nCons, 
                        directory  = directory,
                        verbose    = verbose)
        plt.close('all')


    
def inverse_parametric_linprog(u_train, 
                               x_target, 
                               plp,
                               inner_tol        =  1e-8,  # by default, use high-precision forward solver for the inner problem.
                               outer_method     =  None,
                               grad_mode        =  None,    
                               runtime_limit    =  20,
                               verbose          =  0,
                               collect_evals    =  True):
    """
    solve ILOP (with ineq. constraints only) with COBYLA or SLSQP to learn objective and ineq. constraints

    Input:
         u_train:   list of feature u for training, len(u) = N
        x_target:   list of target solutions, len(x_target) = N
             plp:   a parametric LP class (see parametric_lp_vectorized for more details)
       inner_tol:   tolerance of forward solver
    outer_method:   algorithms for solving the bi-level ILOP formulation
                            RS: random search
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
    assert verbose in [0,1,2], VERBOSE_ERROR_MESSAGE
    assert grad_mode in (None, 'numerical_grad', 'backprop', 'implicit', 'direct', 'cvxpylayer'), GRAD_MODE_ERROR_MESSAGE
    assert outer_method in ('RS', 'COBYLA', 'SLSQP'), OUTER_METHOD_MESSAGE


    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training if loss<loss_tol
    outer_tol        =  1e-15       # optimality tolerance for scipy.optimize.minimize() 
    outer_maxiter    =  10000       # max iteration for scipy.optimize.minimize() package 
                                    # (use a large number to force the algo to continue till reaching the runtime_limit)
    outer_maxeval    =  10000       # max iteration for fop evaluation (call f(w))
                                    # (use a large number to force the algo to continue till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance
    boundedness_cons =  False       

    # initialization
    num_evals      = 0
    curr_w         = None           # Current parameter vector being queried by scipy minimize
    curr_w_rep     = None           # Repeated version of curr_w, to collect Jacobian entries
    curr_lp        = None
    curr_loss      = 999
    curr_soln      = None                # The linprog results to the LPs for curr_w
    curr_target_feasibility_ub = None    # The b_ub - A_ub @ x_target residuals for curr_w
    curr_target_feasibility_eq = None    # The b_eq - A_eq @ x_target residuals for curr_w    
    best_w         = np.ones((len(plp.weights.view(-1)),))*999
    best_loss      = float(999)
    loss_collect   = []
    requires_grad  = (outer_method != 'COBYLA') and (grad_mode in ('backprop', 'implicit', 'direct', 'cvxpylayer'))
    layer          = None   # check cvxpylayer object as None, which will be initialized only if grad_mode == 'cvxpylayer'

    # update curr_w = w
    def is_w_curr(w):
        return (curr_w is not None) and np.array_equal(w, curr_w.detach().numpy())
    
    # Returning cur_lp, i.e., PLP(w, u)
    def get_curr_lp(w):
        nonlocal curr_w, curr_w_rep, curr_lp, curr_soln, curr_loss
        nonlocal curr_target_feasibility_ub
        nonlocal curr_target_feasibility_eq
        if not is_w_curr(w):
            curr_w = torch.tensor(w.ravel(), requires_grad=requires_grad)  # Deliberately copy w, enable gradient for SLSQP
            curr_w_rep = curr_w.view(1, -1).repeat(len(u_train), 1)        # copy w for |u_train| copies for faster backpropagation
            curr_lp = plp.generate(u_train, curr_w_rep)
            curr_soln = None
            curr_loss = None
            curr_target_feasibility_ub = None
            curr_target_feasibility_eq = None

            # Retain the computation graph
            if requires_grad:
                curr_w_rep.retain_grad()
        return curr_lp
    
    # Return curr_sol by solving PLP(w, u)
    def get_curr_soln(w):  # w is ndarray
        nonlocal curr_soln
        if not is_w_curr(w) or curr_soln is None:
            # print('curr_w', w)
            curr_soln = solve_lp(get_curr_lp(w), 
                                 grad_mode, 
                                 inner_tol, 
                                 plp.nneg, 
                                 layer = layer)
        return curr_soln
    
    # get the status code from the homogeneous solver
    def get_curr_status(w):
        return get_curr_soln(w)[1]
        
    def get_curr_loss(w):  # w is ndarray
        nonlocal curr_loss
        if not is_w_curr(w) or curr_loss is None:
            c = get_curr_lp(w)[0]
            x, s, *_ = get_curr_soln(w)

            l = loss_fn(c, x_target, x).view(-1)
            if boundedness_cons:
                l.masked_fill_(s >= 1, 1)  # Without this, huge loss makes SLSQP go crazy and it never converges
            curr_loss = l.mean()

            if (s==5).all():
                curr_loss.data = torch.tensor(100, dtype = DT)        

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
    
    # Reset the gradient of curr_w
    def zero_w_grad():
        if curr_w.grad is not None:
            curr_w.grad.zero_()
    
    # Reset the gradient of curr_w_rep 
    def zero_w_rep_grad():
        if curr_w_rep.grad is not None:
            curr_w_rep.grad.zero_()

    # Fills jac[start:start+size,:] with the Jacobian of target feasibility
    # residuals r with respect to w. Since residual each r[i*m+j] is a dependent
    # variable of only w_rep[i] for training cases i=0:k, we can compute k entries
    # of the Jacobian at a time in a single backward pass, for m passes total.
    def fill_jac_target_feasibility(jac, r, start, size):
        k = len(u_train)
        m = size // k     # Number of constraints (residuals per training case)
        mask = torch.zeros_like(r)
        for j in range(m):
            # Compute the relevent elements of the Jacobian of w with respect to
            # to the jth residual of each training case in a single backward pass.
            zero_w_rep_grad()
            if j > 0:
                mask[(j-1)::m] = 0  # Zero out the previous mask items
            mask[j::m] = 1          # Backprop only the jth residual of each training case
            r.backward(mask, retain_graph=True)
            jac[start+j:start+size:m,:] = curr_w_rep.grad.data

    # Fills jac[start:start+size,:] with the Jacobian of primal-dual gap residuals r
    # with respect to w. Since residual r[i] is a dependent variable of only w_rep[i],
    # we can compute the Jacobian in a single backward pass.
    def fill_jac_pdgap(jac, r, start, size):
        zero_w_rep_grad()
        r.backward(torch.ones_like(r), retain_graph=True)
        jac[start:start+size,:] = curr_w_rep.grad.data
    
    
    # Function to evaluate loss at point w in weight space
    def f(w):
        nonlocal num_evals, best_loss, best_w

        if time.time() - io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
        if num_evals >= outer_maxeval:
            raise EvaluationLimitExceeded()
        num_evals += 1
        l = float(get_curr_loss(w))

        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
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
            print('   [%d]  f(%s) \n \t  l=[%.8f]  infea=[%.4f]  pd=%s  s=%s'% (num_evals, w.ravel(), l,
                                                get_curr_target_feasibility_violation(w),
                                                get_curr_pdgap_violation(w) if boundedness_cons else None,
                                                get_curr_status_codes(w)))
        if best_loss < loss_tol:
            raise LossToleranceExceeded()
            
        return l

    # Function to evaluate gradient of loss at w
    def f_grad(w):
        nonlocal curr_lp
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] df(%s)'% (num_evals, w.ravel()))

        zero_w_grad()
        l = get_curr_loss(w)

        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
            raise UnboundedWeightsValue()

        l.backward(retain_graph=True)
        return curr_w.grad.numpy()

    # Function to evaluate ineq. constraint residuals of outer problem at w
    def g(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d]  g(%s)'% (num_evals, w.ravel()))

        r = get_curr_target_feasibility_ub(w).detach().numpy() # x_target is feasible
        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
            raise UnboundedWeightsValue()

        if boundedness_cons:
            r = np.concatenate((r, get_curr_pdgap_residual(w).detach().numpy()))
        return r

    # Function to evaluate gradient of ineq. constraint residuals of outer problem at w
    def g_grad(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] dg(%s)'% (num_evals, w.ravel()))

        r1 = get_curr_target_feasibility_ub(w)
        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
            raise UnboundedWeightsValue()

        m1 = len(r1) if r1 is not None else 0
        if boundedness_cons:
            r2 = get_curr_pdgap_residual(w)
            m2 = len(r2)
            jac = np.empty((m1 + m2, len(w)))
            fill_jac_target_feasibility(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)
            fill_jac_pdgap(jac, r2, m1, m2)               # Fill wrt boundedness
        else:
            jac = np.empty((m1, len(w)))
            fill_jac_target_feasibility(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)
        return jac

    # Function to evaluate eq. constraint residuals of outer problem at w
    def h(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('[%d]  h(%s)'% (num_evals, w.ravel()))
        r = get_curr_target_feasibility_eq(w).detach().numpy()
        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
            raise UnboundedWeightsValue()

        return r

    # Function to evaluate gradient of eq. constraint residuals of outer problem at w
    def h_grad(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('[%d] dh(%s)'% (num_evals, w.ravel()))
        r = get_curr_target_feasibility_eq(w)
        if max(abs(curr_w)) >1e5:               # detect unbounded weights
            
            raise UnboundedWeightsValue()

        jac = np.empty((len(r), len(w)))
        fill_jac_target_feasibility(jac, r, 0, len(r))
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

            layer = create_cvx_layer(plp.generate(u_train, plp.weights), plp.nneg)

        else:
            raise Exception('Invalid grad_mode for SLSQP')


    has_eq = plp.generate(u_train, plp.weights)[4]  #eq. constraints
    constraints = [{'type': 'ineq', 'fun': g, 'jac': g_jac}]
    if has_eq is not None:
        constraints.append({'type': 'eq', 'fun': h, 'jac': h_jac})

    io_tic = time.time()
    try:
        res = opt.minimize(f, plp.weights.detach(), jac = f_jac, 
                            constraints = constraints, 
                            tol = outer_tol,
                            method = outer_method,
                            options = {'maxiter': outer_maxiter})
        if not res.success:
            if verbose >=1:
                print("WARNING: scipy minimize returned failure")
        
    except (EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded, UnboundedWeightsValue) as e:
        if verbose >=1:
            print("EXCEPTION earily terminaton [%s]"%e.__class__.__name__)
        # If loss tolerance was exceeded, tack on the missing loss value that would have been
        # returned by f(w) had we not needed to rely on exception mechanism to terminate minimize()
        if collect_evals and isinstance(e, LossToleranceExceeded):
            assert len(f.rvals) == len(f.calls)-1
            f.rvals.append(best_loss)
    except RuntimeError as err:
        msg, *_ = err.args
        print("RuntimeError [%s]"%msg)


    io_toc = time.time()

    if best_loss < 999:
        plp.weights[:] = torch.from_numpy(best_w)
    
    if len(loss_collect) > 0:
        assert best_loss == loss_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_loss, loss_collect[-1])
    else:
        loss_collect = [999]

    if verbose >= 1:
        print("++++           call [f %d times], [f_grd %d times], [g %d times], [g_grd %d times]" \
                    % (len(f.calls), len(f_grad.calls), len(g.calls), len(g_grad.calls)) )
        print("++++           initial_loss [%.8f], best_loss [%.8f], runtime [%.4f s]"%(loss_collect[0], best_loss, (io_toc-io_tic)))
        if len(best_w.ravel()) > 10:
            print("++++           best_w is to long, only print first 10 elements:\n %s"%(best_w.ravel()[:10]))
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

def InvPLP_random_search(u_train, 
                         x_target,
                         plp,
                         w_init,
                         inner_tol       =  1e-8,  # by default, use high-precision forward solver.
                         runtime_limit   =  20,
                         verbose         =  0,
                         collect_evals   =  True):
    """
    solve ILOP (with ineq. constraints only) with random search to learn objective and ineq. constraints
        Randomly sample w repeatedly from the same distribution as generating the true PLP (see InvPLP_random_initialization)

    Input:
         u_train:   list of feature u for training, len(u) = N
        x_target:   list of target solutions, len(x_target) = N
             plp:   a parametric LP class (see parametric_lp_vectorized for more details)
          w_init:   initial value of parameter w

   Output:
      best_loss:  best loss value
          evals:  (if collect_evals = True), a dict contains all relevant results (e.g., collection of w and loss at every iteration)
    """    
    
    sigma_w          = 1            # used to sample w_init (same distribution function as w_true was sampled)
    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training if loss<loss_tol
    outer_maxeval    =  10000       # max iteration for evaluation fop (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance


    num_evals     =  0
    best_w        =  np.ones((6,))*999
    best_loss     =  float(999)
    loss_collect  =  []
        
    def get_curr_loss(w):  # w is ndarray
        curr_w = w.clone() # Deliberately copy w
        
        curr_lp = plp.generate(u_train, curr_w.detach())
        curr_soln = homogeneous_solver(*curr_lp, inner_tol, plp.nneg)
        
        c = curr_lp[0]
        x, s, _, _, = curr_soln
        l = loss_fn(c, x_target, x).view(-1)
        curr_loss = l.mean()
        
        return float(curr_loss), s, curr_lp
    
    # Compute constraint residual, i.e, b - A@x
    def calc_residual(b, A, x): 
        if A is None:
            return None
        # Using _bsubbmm().neg_() avoids temporaries and is slightly faster than (b - bmm(A, x))
        return _bsubbmm(b, A, x).neg_().view(-1)  # -(-b + A@x) == b - A@x

    # Check A_ub @ X <= b_ub
    def get_curr_target_feasibility_ub(curr_lp):  # w is ndarray
        _, A_ub, b_ub, _, _ = curr_lp
        curr_target_feasibility_ub = calc_residual(b_ub, A_ub, x_target)
        return curr_target_feasibility_ub

    # Check A_eq @ X == b_eq
    def get_curr_target_feasibility_eq(curr_lp):  # w is ndarray
        _, _, _, A_eq, b_eq = curr_lp
        curr_target_feasibility_eq = calc_residual(b_eq, A_eq, x_target)
        return curr_target_feasibility_eq
    
    # Return feasibility violation
    def get_curr_target_feasibility_violation(curr_lp):  # w is ndarray
        r_ub = get_curr_target_feasibility_ub(curr_lp)
        r_eq = get_curr_target_feasibility_eq(curr_lp)
        r = 0.0
        if r_ub is not None: r = max(r, -float(r_ub.detach().min()))        # Maximum inequality violation
        if r_eq is not None: r = max(r,  float(r_eq.detach().abs().max()))  # Maximum equality violation
        return r
    
    def f(w):
        nonlocal num_evals, best_loss, best_w

        w = as_tensor(w)

        if time.time() - io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
        if num_evals >= outer_maxeval:
            raise EvaluationLimitExceeded()
        num_evals += 1
        l, s, curr_lp = get_curr_loss(w)

        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        # Record the best loss for feasible w, in case we later hit evaluation limit
        
        if float(l) < best_loss and sum(s) == 0:
            v = get_curr_target_feasibility_violation(curr_lp)
            # Record loss only if outer problem constraints are satisfied and fop is feasible
            if v < violation_tol:
                best_loss = float(l)
                best_w = w.detach().numpy().copy()     
        else:
            v = -999
        loss_collect.append(best_loss)
        if verbose == 2:
            print('   [%d]  f(%s) \n \t  l=[%.8f]  infea=[%.4f]  s=%s'% (num_evals, w.detach().numpy().ravel(), float(l),
                                                v,
                                                [int(s_) for s_ in s]))
        if best_loss < loss_tol:
            raise LossToleranceExceeded()
                
    if collect_evals:
        f = collect_calls(f)

    io_tic = time.time()
    try:
        w = as_numpy(w_init)
        f(w)
        for iter_i in range(outer_maxeval-1):
            w = InvPLP_random_initialization(sigma_w, (w_init[i:i+2] for i in range(0, len(w_init),2) ) )
            f(as_numpy(w))
        
    except (EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded) as e:
        if verbose >=1:
            print("EXCEPTION earily terminaton [%s]"%e.__class__.__name__)
        # If loss tolerance was exceeded, tack on the missing loss value that would have been
        # returned by f(w) had we not needed to rely on exception mechanism to terminate minimize()
        if collect_evals and isinstance(e, LossToleranceExceeded):
            assert len(f.rvals) == len(f.calls)-1
            f.rvals.append(best_loss)
    io_toc = time.time()

    if best_loss < 999:
        plp.weights[:] = torch.from_numpy(best_w)
        
    assert best_loss == loss_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_loss == loss_collect[-1])

    if verbose >= 1:
        print("++++           call [f %d times]"% (len(f.calls) ) )
        print("++++           initial_loss [%.8f], best_loss [%.8f], runtime [%.4f s]"%(loss_collect[0], best_loss, (io_toc-io_tic)))
        if len(best_w.ravel()) > 10:
            print("++++           best_w is to long, only print first 10 elements:\n %s"%(best_w.ravel()[:10]))
        else:
            print("++++           best_w %s"%(best_w.ravel()))
        
    if collect_evals:
        evals = {
            'f': {'calls': f.calls, 'rvals': f.rvals, 'times': f.times, 'loss': loss_collect},
            'res': {'runtime': (io_tic, io_toc), 'best_l': best_loss, 'best_w':best_w}
        }
        return best_loss, evals
    
    return best_loss

def InvPLP_random_initialization(sigma_w, weights_true):
    """
    Randomly sample w
    """
    w_c, w_A, w_b = weights_true

    w_c_distri = torch.distributions.Uniform( torch.Tensor([ -sigma_w ]), torch.Tensor([ sigma_w ]) )
    w_A_distri = torch.distributions.Uniform( torch.Tensor([        0 ]), torch.Tensor([ sigma_w ]) )
    w_c_init = w_c_distri.sample(torch.Size(w_c.shape)).view(-1).type(torch.DoubleTensor) 
    w_A_init = w_A_distri.sample(torch.Size(w_A.shape)).view(-1).type(torch.DoubleTensor) 
    w_b_init = w_A_distri.sample(torch.Size(w_b.shape)).view(-1).type(torch.DoubleTensor) 

    return torch.cat( (w_c_init, w_A_init, w_b_init), 0)

def InvPLP_experiment( nVar, 
                       nCons, 
                       ind_ins, 
                       inner_tol        =  1e-8,
                       outer_method     =  None,
                       grad_mode        =  None,
                       plp_class        =  parametric_lp_vectorized,
                       runtime_limit    =  30,
                       directory        =  None,
                       verbose          =  0):
    
    """
    run ILOP experiment for a specific plp instance

    Input:
          nVar, nCons: num. variabel, num. constriants,
              ind_ins:  instance index
            inner_tol: tolerance of forward solver
         outer_method: algorithms for solving the bi-level ILOP
            grad_mode: backward methods for computing gradients
            plp_class: parametric LP class (vectorized LP generation with different w values)
        runtime_limit: runtime limit for solving ILOP (in seconds)
    """

    assert verbose in [0,1,2], VERBOSE_ERROR_MESSAGE
    assert grad_mode in (None, 'numerical_grad', 'backprop', 'implicit', 'direct', 'cvxpylayer'), GRAD_MODE_ERROR_MESSAGE
    assert outer_method in ('RS', 'COBYLA', 'SLSQP'), OUTER_METHOD_MESSAGE

    PLP_filename = directory + '%dvar%dcons_InvPLP%d.txt'\
            %(nVar, nCons, ind_ins)
    if verbose >= 1:
        print('************************************')
        print('ILOP Exp1a %dvar%dcon ins%d '%(nVar, nCons, ind_ins) )
        if outer_method =='RS':
            print('\twith Random Search [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
        else:
            if outer_method == 'COBYLA':
                print('\twith COBYLA [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
            elif outer_method == 'SLSQP':
                print('\twith SLSQP [%s] %s(tol = %.2e), '
                        %(grad_mode, 'cvx' if grad_mode == 'cvxpylayer' else 'homogeneous', inner_tol))

        print('\tloss_fn = [AOE]')

    # Initialization - find x_target from w_true
    base_PLP, weights_true, w_A_coef, u = read_InvPLP(PLP_filename)
    u_train, u_test   = as_tensor(*u)
    if verbose == 2:
        print('len(u_train) = [%d] len(u_test) = [%d]'%(len(u_train), len(u_test)))
    u_train = u_train.view(-1,1)
    
    plp     =  plp_class(as_tensor(*base_PLP),                       
                                   torch.cat(as_tensor(*weights_true)).view(-1), 
                                   torch.LongTensor(w_A_coef))


    if verbose >=1: 
        print('[Compute x_target] Initlize the InvPLP with w_true, and then compute Soln_true as the targets')
    target_res   =  homogeneous_solver(*plp.generate(u_train, plp.weights),   # computing the x_target(u_train, w_true) using homogeneous_solver with high-precision
                                    nneg = plp.nneg,
                                    tol = 1e-8)

    x_target = target_res[0].detach().clone()
    
    assert all(target_res[1]==0), 'Some FOPs of the true PLP cant be solved with homogeneous IPM, s =%s'%([int(s_) for s_ in target_res[1]])
    # Initialization - find a feasible w_init to start inverse_parametric_linprog
    if verbose == 2:
        print('Initialize with random w_init')

    # randomly initialize w_init till the corresponding PLP is feasible    
    torch.manual_seed(ind_ins)
    for iter_i in range(1, 1000):
        plp_corrupt = deepcopy(plp)
        weight_init = InvPLP_random_initialization(1, weights_true)
        init_res   =  homogeneous_solver(*plp_corrupt.generate(u_train, weight_init), 
                            tol = 1e-8, nneg = plp.nneg)
        if sum(init_res[1]) == 0:
            break
        else:
            if verbose == 2:
                print('\t infeasible w_init, status = %s'%( [ int(s_) for s_ in init_res[1].numpy()] ) )

    if verbose >= 1:
        print(' [Initialization] find a set of feasible w_init %s after %d fop evals'%(weight_init.numpy().round(4), iter_i))

    if outer_method == 'RS':
        _, res = InvPLP_random_search( u_train         =  u_train, 
                                       x_target        =  x_target, 
                                       plp             =  plp,
                                       w_init          =  weight_init,
                                       inner_tol       =  inner_tol,
                                       runtime_limit   =  runtime_limit,
                                       verbose         =  verbose,
                                       collect_evals   =  True)

    else:
        plp.weights = weight_init
        _, res = inverse_parametric_linprog(u_train          =  u_train, 
                                            x_target         =  x_target, 
                                            plp              =  plp,
                                            inner_tol        =  inner_tol,
                                            outer_method     =  outer_method,
                                            grad_mode        =  grad_mode,
                                            runtime_limit    =  runtime_limit,
                                            verbose          =  verbose,
                                            collect_evals    =  True)
 
    if outer_method == 'SLSQP':
        save_file_directory = directory + '%s_%s'%(outer_method, grad_mode)
    else:
        save_file_directory = directory + '%s'%(outer_method)
    if not os.path.exists(save_file_directory):
        os.makedirs(save_file_directory)

    _file_record = save_file_directory + '/%dvar%dcons_InvPLP%s_%s_res.pkl'\
                %(nVar, nCons, ind_ins, 'AOE')
    with open(_file_record, 'wb') as output:
        pickle.dump(res, output)
            
    if verbose >= 1:
        print('************************************\n')
    return res

def InvPLP(lp_size,
           nIns,
           runtime_limit, 
           framework,
           directory,
           verbose):
    """
    Complete experiment for inverse parametric linear programming problems with ineq. constraints only.  

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
                direct = directory + 'InvPLP_Ins/%dvar%dcons/'%(nVar, nCons)
                _ = InvPLP_experiment(nVar, nCons, ind_ins,
                                      inner_tol        =  1e-8,
                                      outer_method     =  outer_method,
                                      grad_mode        =  grad_mode,
                                      runtime_limit    =  runtime_limit,
                                      directory        =  direct,
                                      verbose          =  verbose)

def InvPLP_data_analysis(lp_size, 
                         nIns,
                         loss_fn,  
                         framework,
                         directory     =  None,
                         verbose       =  False):

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
    # loss_collect at each f(w)
    # Note, loss_collect records the best loss so far
    l_RS               =  []
    l_COBYLA           =  []
    l_SLSQP_backprop   =  []
    l_SLSQP_implicit   =  []
    l_SLSQP_direct     =  []
    l_SLSQP_cvxpylayer =  []

    # w_collect at each f(w)
    # w_collect records the w even it is a bad solution
    w_RS               =  []
    w_COBYLA           =  []
    w_SLSQP_backprop   =  []
    w_SLSQP_implicit   =  []
    w_SLSQP_direct     =  []
    w_SLSQP_cvxpylayer =  []

    # clock time of each f(w)
    timestep_RS               =  []
    timestep_COBYLA           =  []
    timestep_SLSQP_backprop   =  []
    timestep_SLSQP_implicit   =  []
    timestep_SLSQP_direct     =  []
    timestep_SLSQP_cvxpylayer =  []

    # total runtime
    runtime_RS               =  []
    runtime_COBYLA           =  []
    runtime_SLSQP_backprop   =  []
    runtime_SLSQP_implicit   =  []
    runtime_SLSQP_direct     =  []
    runtime_SLSQP_cvxpylayer =  []

    for ind_lp in range(len(lp_size)):
        nVar, nCons = lp_size[ind_lp]
        for framework_i in framework:
            outer_method, grad_mode = framework_i
            if outer_method in ('RS','COBYLA'):
                _file_path = directory + 'InvPLP_Ins/%dvar%dcons/%s'\
                                            %(nVar, nCons, outer_method)
            else:
                _file_path = directory + 'InvPLP_Ins/%dvar%dcons/%s_%s'\
                                            %(nVar, nCons, outer_method, grad_mode)

            for ind_ins in range(nIns):
                if verbose > 1:
                    print('\n=======  %d var, %d cons, Ins %d, loss = %s '
                                    %(nVar, nCons, ind_ins, loss_fn.__name__[1:-1]))
                    print('         Outer_method %s, grad_mode %s'
                                    %(outer_method, grad_mode))
                
                result_file = _file_path+'/%dvar%dcons_InvPLP%d_%s'\
                        %(nVar, nCons, ind_ins,\
                          loss_fn.__name__[1:-1])

                w_collect, l_collect, timestep, runtime, best_l, best_w = read_exp_result(result_file)

                num_evals = min((len(w_collect), len(l_collect), len(timestep), len(timestep<=30)) )

                w_collect = w_collect[:num_evals]
                l_collect = l_collect[:num_evals]
                timestep = timestep[:num_evals]

                assert sum(l_collect[-1] - best_l) <= 1e-3, 'Inconsistentency in experiment results: l_collect[-1] =%s, best_l = %s'%(l_collect[-1], best_l)

                if outer_method == 'RS':
                    l_RS.append(l_collect)
                    timestep_RS.append(np.array(timestep))
                    runtime_RS.append(runtime)
                    w_RS.append(w_collect)
                elif outer_method == 'COBYLA':
                    l_COBYLA.append(l_collect)
                    timestep_COBYLA.append(np.array(timestep))
                    runtime_COBYLA.append(runtime)
                    w_COBYLA.append(w_collect)
                elif outer_method == 'SLSQP':
                    if grad_mode == 'backprop':
                        l_SLSQP_backprop.append(l_collect)
                        timestep_SLSQP_backprop.append(np.array(timestep))
                        runtime_SLSQP_backprop.append(runtime)
                        w_SLSQP_backprop.append(w_collect)
                    elif grad_mode == 'implicit':
                        l_SLSQP_implicit.append(l_collect)
                        timestep_SLSQP_implicit.append(np.array(timestep))
                        runtime_SLSQP_implicit.append(runtime)
                        w_SLSQP_implicit.append(w_collect)  
                    elif grad_mode == 'direct':
                        l_SLSQP_direct.append(l_collect)
                        timestep_SLSQP_direct.append(np.array(timestep))
                        runtime_SLSQP_direct.append(runtime)
                        w_SLSQP_direct.append(w_collect)  
                    elif grad_mode == 'cvxpylayer':
                        l_SLSQP_cvxpylayer.append(l_collect)
                        timestep_SLSQP_cvxpylayer.append(np.array(timestep))
                        runtime_SLSQP_cvxpylayer.append(runtime)
                        w_SLSQP_cvxpylayer.append(w_collect)  

    return  (l_RS,                w_RS,                timestep_RS,                runtime_RS),\
            (l_COBYLA,            w_COBYLA,            timestep_COBYLA,            runtime_COBYLA), \
            (l_SLSQP_backprop,    w_SLSQP_backprop,    timestep_SLSQP_backprop,    runtime_SLSQP_backprop),\
            (l_SLSQP_implicit,  w_SLSQP_implicit,  timestep_SLSQP_implicit,  runtime_SLSQP_implicit),\
            (l_SLSQP_direct,  w_SLSQP_direct,  timestep_SLSQP_direct,  runtime_SLSQP_direct),\
            (l_SLSQP_cvxpylayer,  w_SLSQP_cvxpylayer,  timestep_SLSQP_cvxpylayer,  runtime_SLSQP_cvxpylayer)

def boxplot_InvPLP(ax, 
                  lp_size,      
                  loss_fn,
                  nIns,
                  end_time, 
                  data, 
                  plp_class,
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
       plp_class:  parametric LP class with vecotrized code for generating LPs
    """

    def compute_testing_error(nVar, 
                              nCons, 
                              ind_ins, 
                              w_test, 
                              l_train, 
                              loss_fn,
                              plp_class,
                              directory,
                              method_name,
                              verbose):
        """
        Compute testing loss
        when using AOE as the loss function


          lp_size: a list of tuple (nVar, nCons)
          loss_fn: function for evluation loss
          ind_ins: index of instances
           w_test: parameter w used for initialize PLPs and computing sol_test and l_test
          l_train: best training loss, if l_train = 999 that means no feasible IO solution is found, skip the function
          loss_fn: function used for evaluaiting loss
        plp_class: parametric LP class (for vectorized LP generation)
        
        """
        if verbose == True:
            print('         +++ Compute testing error for %dvar %dcons InvPLP%d'%(nVar, nCons, ind_ins) )

        if l_train >= 999:
            if verbose == True:
                print('             Failed to find feasible soln, thus, set testing error = 999' )
            l_mean = 999
            l_median = 999
        else:
            PLP_filename = directory + 'InvPLP_Ins/%dvar%dcons/%dvar%dcons_InvPLP%d.txt'\
                    %(nVar, nCons, nVar, nCons, ind_ins)

            # read true PLP from file
            base_PLP, weights_true, w_A_coef, u = read_InvPLP(PLP_filename)
            _, u_test = as_tensor(*u)
            u_test = u_test.view(-1,1)

            plp =  plp_class(as_tensor(*base_PLP),                       
                             torch.cat(as_tensor(*weights_true)).view(-1), 
                             torch.LongTensor(w_A_coef))
            c_true, A_ub_true, b_ub_true, A_eq_true, b_eq_true = plp.generate(u_test, plp.weights)

            # computing the x_target(u_test, w_best) using homogeneous_solver with high-precision
            target_res = homogeneous_solver(c_true.detach(),   
                                            A_ub_true.detach(), 
                                            b_ub_true.detach(),
                                            A_eq_true.detach() if A_eq_true is not None else None, 
                                            b_eq_true.detach() if b_eq_true is not None else None, 
                                            tol = 1e-8, nneg = plp.nneg)
            x_target = target_res[0].detach().clone()
            assert sum(target_res[1]) == 0, 'w_true is not feasible, s = %s'%target_res[1].detach().numpy()

            # initialize PLP(w_test) and compute the corresponding solution, soln_test
            plp.weights  =  tensor(w_test).view(-1)
            c_lrn, A_ub_lrn, b_ub_lrn, A_eq_lrn, b_eq_lrn = plp.generate(u_test, plp.weights)
            soln_test = homogeneous_solver(c_lrn.detach(),    # computing the x_lrn(u_test, w_best) using homogeneous_solver with high-precision
                                           A_ub_lrn.detach(), 
                                           b_ub_lrn.detach(),
                                           A_eq_lrn.detach() if A_eq_lrn is not None else None, 
                                           b_eq_lrn.detach() if b_eq_lrn is not None else None, 
                                           tol = 1e-8, nneg = plp.nneg)

            if sum(soln_test[1]) == 0: # if PLP(w_test) is feasible
                x_final  = soln_test[0].detach().clone()
                # testing loss = l(c_true, x_true, x_lrn) = c_true|x_true - x_lrn|
                l_mean   = loss_fn(c_true, x_target, x_final).mean()  
                l_median = loss_fn(c_true, x_target, x_final).median()  
                if verbose == True:
                    print('             mean testing error = [%.2e], median testing error = [%.2e]'%(float(l_mean), float(l_median)) )
            else: # if PLP(w_test) is infeasible
                if verbose == True:
                    print('             !!!!ind[%d] method[%s] w_lrn lead to infeasible LPs on u_test, s = %s'%(ind_ins, method_name, soln_test[1].detach().numpy()) )
                    print('             !!!!w_lrn = %s'%w_test)
                    
                l_mean = 999
                l_median = 999
            if l_mean < 1e-5:
                l_mean = 1e-5
            if l_median < 1e-5:
                l_median = 1e-5

        return float(l_mean), float(l_median)
    
    def _plot(loss_plot, color, axes):
        axes.boxplot(loss_plot, sym='')
        x_ticks = np.linspace(0.7, 1.3, nIns)
                # RS,  COBYLA, SQP_cvx, SQP_bprop, SQP_impl, SQP_dir
        marker = ['+', 'x',    'd',     '^',       's',      'o', ]
        for i in range(len(loss_plot)):
            axes.plot(x_ticks+i, loss_plot[i], 'o', 
                    marker=marker[i], mfc='None',
                    ms=5, mew=1, color = color[i],
                    alpha = 0.5,)
        axes.set_yscale('log')
        axes.set_ylim(5e-6,5e2)
        axes.get_xaxis().set_visible(False)
        axes.set_xlim(0,7)
    
    # initialization
    numIns = len(data[0][0])
    numMethod = len(data)
    loss_initial = np.ones((numIns,))*999          # collect initial loss
    loss_train = np.ones((numMethod, numIns))*999  # collect initial loss

    loss_test_mean = np.ones((numMethod, numIns))*999    # collect initial loss
    loss_test_median = np.ones((numMethod, numIns))*999  # collect initial loss

    # collect loss_init and loss_final
    for ind_ins in range(numIns):
        for outer_method_i in range(numMethod-1):
            #                    method            l   ins     init_l
            ini_l_1 = float(data[outer_method_i  ][0][ind_ins][ 0 ])
            ini_l_2 = float(data[outer_method_i+1][0][ind_ins][ 0 ])
            if outer_method_i != 2 and outer_method_i+1 != 2:
                # sanity check if RS, COBYLA, SQP_bprop, SQP_implic and SQP_close have the same initial loss
                # SQP_cvx uses different forward solver which will have different initial loss, thus, it is 
                # not included in this sanity check.
                assert abs(abs(ini_l_1) - abs(ini_l_2)) <=1e-5, 'ind_ins[%d], outer_method %d [%.5f] and %d [%.5f] have different initial loss'%(ind_ins, outer_method_i, ini_l_1, outer_method_i+1, ini_l_2)
                        
        l_init_ = float(data[0][0][ind_ins][0])
        loss_initial[ind_ins] = 100 if l_init_ >= 999 else l_init_
        
        method_name = ['RS', 'COBYLA', 'SLSQP_cvx', ' SLSQP_bprop', 'SLSQP_implicit', ' SLSQP_dir']

        for outer_method_i in range(numMethod):
            loss_collect    = data[outer_method_i][0][ind_ins].copy()
            time_collect    = data[outer_method_i][2][ind_ins].copy()
            assert len(time_collect) == len(loss_collect), 'len(time_collect) = %d,  len(loss_collect) = %d'%(len(time_collect), len(loss_collect))

            idx = np.where(time_collect <= end_time)[0] # find all iter that finished before reaching runtime_limit
            if len(idx) > 0:
                ind = max(idx)  
                l_train = loss_collect[ind]
                loss_train[outer_method_i][ind_ins] = float(l_train)  if l_train >1e-5 else float(1e-5) 

                weights_collect = data[outer_method_i][1][ind_ins].copy()
                w_ind = min(np.where(loss_collect == l_train)[0])
                nVar, nCons = lp_size
                l_test_mean, l_test_median = compute_testing_error(nVar      = nVar, 
                                                                nCons     = nCons,  
                                                                ind_ins   = ind_ins, 
                                                                w_test    = weights_collect[w_ind], 
                                                                l_train   = l_train, 
                                                                loss_fn   = loss_fn, 
                                                                plp_class = plp_class,
                                                                directory = directory,
                                                                method_name = method_name[outer_method_i],
                                                                verbose   = False)

                loss_test_mean[outer_method_i][ind_ins] = 100 if l_test_mean >= 999 else float(l_test_mean)         
                loss_test_median[outer_method_i][ind_ins] = 100 if l_test_mean >= 999 else float(l_test_median)         
            else: # if len(idx) == 0, it means the outer_method was not able 
                  # to finish even one iteration within the runtime_limit
                loss_train[outer_method_i][ind_ins] = float(999)   
                loss_test_mean[outer_method_i][ind_ins] = float(999)         
                loss_test_median[outer_method_i][ind_ins] = float(999)   

            if verbose > 1:
                print('init [%.5f], train [%.5f], test(mean)[%.5f], test(median)[%.5f]'
                %(loss_initial[ind_ins], l_train, l_test_mean, l_test_median))

    if verbose > 1:
        print('num l_train <= 1e-5: ', [sum(loss_train[i,:]<=1e-5) for i in range(len(loss_train))])

    loss_initial  = np.array(loss_initial)
    loss_initial = np.clip(loss_initial, 1e-5, 1e2)

    loss_train = np.array(loss_train)
    loss_train = np.clip(loss_train, 1e-5, 1e2)
    
    loss_test_mean  = np.array(loss_test_mean)
    loss_test_mean = np.clip(loss_test_mean, 1e-5, 1e2)

    loss_test_median  = np.array(loss_test_median)
    loss_test_median = np.clip(loss_test_median, 1e-5, 1e2)

    _plot([loss_train[i,:] for i in range(len(loss_train))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
          color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
          axes = ax[0])

    _plot([loss_test_mean[i,:] for i in range(len(loss_test_mean))],
                   # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
           color = ['m', 'b',      'k',    'r',      'orange',    'g', ],       
           axes = ax[1])

    _plot([loss_test_median[i,:] for i in range(len(loss_test_median))],
                   # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
           color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
           axes = ax[2])

    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[2].yaxis.set_label_position("right")
    ax[2].yaxis.tick_right()

    return loss_initial, loss_train


def fig_exp1a(lp_size, nIns, loss_fn, framework, directory, verbose = False):
    """
    Generate figures for exp1a

      lp_size:  a list of tuple (nVar, nCons)
         nIns: number of instance for each lp_size
      loss_fn: function for evaluating the loss
    framework: a list of tuple (outer_method, grad_mode)
    """

    time_range = [np.arange(1e-3, 5, 1e-3).round(3),    #   time steps for 2D.
                  np.arange(1e-3, 30, 1e-3).round(3)]   #   time steps for 10D

    res_RS, res_COBYLA, res_SLSQP_backprop,\
    res_SLSQP_implicit, res_SLSQP_direct,\
    res_SLSQP_cvxpylayer = InvPLP_data_analysis(lp_size       =  lp_size, 
                                                nIns          =  nIns,
                                                loss_fn       =  loss_fn,
                                                framework     =  framework,
                                                directory     =  directory,
                                                verbose       =  verbose)

    def subplot(lp_size, 
                nIns,
                res_rs, 
                res_c, 
                res_s_backprop, 
                res_s_implicit, 
                res_s_direct,
                res_s_cvxpylayer,
                time_range):

        """
        create plots for each lp_size
        """
        success_rate = []
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,3))
        res_ = plot_success_rate(time_range = time_range,
                            res = res_rs, 
                            ax = ax1,
                            plot_style = ['m', '-', 'Random Search'])
        success_rate.append(res_[-1])
        
        res_ = plot_success_rate(time_range = time_range,
                            res = res_c, 
                            ax = ax1,
                            plot_style = ['b', '-', 'COBYLA'])
        success_rate.append(res_[-1])
        
        res_ = plot_success_rate(time_range = time_range,
                            res = res_s_cvxpylayer, 
                            ax = ax1,
                            plot_style = ['k', '-', 'SQP_cvx'])
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                            res = res_s_backprop, 
                            ax = ax1,
                            plot_style = ['r', '-', 'SQP_bprop'])
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                            res = res_s_implicit, 
                            ax = ax1,
                            plot_style = ['orange', '-', 'SQP_impl'])
        success_rate.append(res_[-1])

        res_ = plot_success_rate(time_range = time_range,
                            res = res_s_direct, 
                            ax = ax1,
                            plot_style = ['g', '-', 'SQP_dir'])
        success_rate.append(res_[-1])

        if verbose >1:
            print('success rate: %s'%(success_rate))

        _ = boxplot_InvPLP(ax         = (ax2, ax3, ax4),             # boxplot order, RS, COBYLA, SQP+cvx, SQP+back, SQP+implicit, SQP+closed
                           lp_size    = lp_size, 
                           loss_fn    = loss_fn,
                           nIns       = nIns,
                           end_time   = time_range[-1], 
                           plp_class  = parametric_lp_vectorized,
                           data       =  (res_rs, 
                                          res_c,
                                          res_s_cvxpylayer,
                                          res_s_backprop, 
                                          res_s_implicit, 
                                          res_s_direct),
                           directory  =  directory,
                           verbose    = verbose)

        ax1.tick_params( axis='x', which='major',labelsize=12 )
        ax1.set_ylim(-5, 105)
        ax1.set_yticks(np.linspace(0,100,5))
        ax1.set_yticklabels([str(i)+'%' for i in np.linspace(0,100,5, dtype = np.int)])
        ax1.set_xlim(-0.1, time_range[-1]+0.1)

        # set xticks (in seconds) for the success_rate_curve
        if lp_size[0] == 2:
            ax1.set_xticks([0, 1, 2, 3, 4, 5])
            ax1.set_xticklabels([0, 1, 2, 3, 4, 5],fontsize=12)
        elif lp_size[0] == 10:
            ax1.set_xticks([0, 10, 20, 30])
            ax1.set_xticklabels([0, 10, 20, 30],fontsize=12)

        # plot y_ticks with log scale
        import matplotlib
        locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
        ax4.yaxis.set_major_locator(locmaj)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax4.yaxis.set_minor_locator(locmin)
        ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax4.tick_params(axis='x', which='minor', size = 12)


        return fig, ax1, ax2

    for i_ in range(len(lp_size)):
        if lp_size[i_][0] == 2: #runtime_limit = 5s for 2D instances
            t_range = time_range[0]
        elif lp_size[i_][0] == 10: #runtime_limit = 30s for 10D instances
            t_range = time_range[1]

        if verbose >=1:
            print('====== Analyze experiment results and plot figure for Exp1a %dvar%dcons'%(lp_size[i_][0], lp_size[i_][1]))
            
        
        fig, *_ = subplot(lp_size[i_],
                          nIns,
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_RS], # res_RS contains results len(lp_size)*nIns instances,
                                                                              # read results for corresponding lp_size
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_COBYLA],           # same as res_RS
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_backprop],   # same as res_RS
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_implicit],   # same as res_RS
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_direct],     # same as res_RS
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_cvxpylayer], # same as res_RS
                          t_range, 
                          )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(wspace=0.02, hspace=0)
        fig.savefig(directory+'fig_exp1a_%dvar%dcons.pdf'%(lp_size[i_]),dpi=100)


def main(directory = None):

    direct = './' if directory is None else directory


    # 6 different sizes
    LP_SIZE         = (#var, cons
                       ( 2,  4), 
                       ( 2,  8), 
                       ( 2,  16), 
                       (10,  20), 
                       (10,  36), 
                       (10,  80),
                       )       
    NUM_INS         =  100 

    VERBOSE         =  0    # 2 - print detailed information, e.g., log of each iter in the solving process, 
                            #     initialization, etc. This includes many printing functions during 
                            #     the solving process, thus, using VERBOSE = 2 will influence the experiment results. 
                            # 1 - print basic information, e.g., final results, initialization. 
                            #     all printing function  happen outside of the solving process, 
                            #     thus, does not influence the experiment results 
                            # 0 - don't display any algorithm log         

    RUN_TIME        =  (5, 30 )   # runtime limit for 2D and 10D

    FRAMEWORK       =  (# outer_method, grad_mode
                        ['RS',          None], 
                        ['COBYLA',      None], 
                        ['SLSQP',      'cvxpylayer'],
                        ['SLSQP',      'backprop'], 
                        ['SLSQP',      'implicit'],
                        ['SLSQP',      'direct'],
                       )

    # generate IO instances
    for _lp_size in LP_SIZE:
        nVar, nCons = _lp_size
        dir_ = direct + 'LP_baseline/%dvar%dcons'%(nVar, nCons)

        # generate baseline convexhull
        if not os.path.exists(dir_):
            convexhull_generator(nVar, nCons, NUM_INS)

        # generate PLP instances for IO learning
        generate_InvPLP(lp_size          = (nVar, nCons),
                         nIns             = NUM_INS,
                         directory        = direct,
                         verbose          = VERBOSE-1)

    # experiment on 2D synthetic instances
    InvPLP( lp_size            = LP_SIZE[:3],
            nIns               = NUM_INS,
            runtime_limit      = RUN_TIME[0], 
            framework          = FRAMEWORK,
            directory          = direct,
            verbose            = VERBOSE)

    # experiment on 10D synthetic instances
    InvPLP( lp_size            = LP_SIZE[3:],
            nIns               = NUM_INS,
            runtime_limit      = RUN_TIME[-1], 
            framework          = FRAMEWORK,
            directory          = direct,
            verbose            = VERBOSE)

    fig_exp1a(lp_size        = LP_SIZE,
                nIns         = NUM_INS,
                loss_fn      = _AOE_, 
                framework    = FRAMEWORK,
                directory    = direct,
                verbose      = VERBOSE)



if __name__ == "__main__":
        

    main()