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

# Function used for generateing synthetic instances with ineq and eq constraints
def generate_InvPLP_2(nVar,
                      nCons,
                      nIns,
                      directory,
                      verbose):
    """
     nVar: num. variables
    nCons: num. constraints
     nIns: num of instances

    Generate parametric LPs (with ineq. and eq. constraints) for IO task using the following steps:
    1). read ineq. constraints from baseline convexhull
    2). sample c vector
    3). sample two equality constraints. Repeat 3) till finding a feasible LP(c, A_ub, b_ub, A_eq, b_eq)
    4). sample feature u and split to training and testing
    5). given feature u's, sample parameter w till finding a feasible PLP(u, w, c, A_ub, b_ub, A_eq, b_eq)
    6). redo 4 and 5, if step 5 failed 20 times.

    Save the information of the PLP as a dict in the following form
    {'lp':{'c': c,'A_ub': A_ub,'b_ub': b_ub,'A_eq': A_eq,'b_eq': b_eq},     # baseline LP
    'weights':{'w_c': w_c_true,             # True parameter w
                'w_A_ub': w_A_ub_true,
                'w_b_ub': w_b_ub_true,
                'w_A_eq': w_A_eq_true,
                'w_b_eq': w_b_eq_true,
                'w_A_ub_coef': w_A_ub_coef,  # a list of index indicating which element of each row of A_ub has the parametric term w@u
                'w_A_eq_coef': w_A_eq_coef}, # a list of index indicating which element of each row of A_eq has the parametric term w@u
    'u':{'train': u_train, 'test': u_test}
                   }
    """

    sigma_c   = 1    # used for sampling c
    num_u     = 40   # num. of u values to sample
    u_range   = 1    # used for sampling u values
    sigma_w   = 1    # used for sampling w values

    for ind_ins in range(nIns):
        
        lp_file = directory + 'LP_baseline/%dvar%dcons/%dvar%dcons_LP%d.txt'%(nVar, nCons, nVar, nCons, ind_ins)
        if verbose >=1 :
            print('===== Generate the True PLP, %d var, %d cons, ins %d'%(nVar, nCons, ind_ins))
        A_ub, b_ub, _ = read_baseline_convexhull(lp_file)

        np.random.seed(ind_ins)
        c   = np.random.uniform(-sigma_c, sigma_c, size=(nVar,1)).round(6)

        status = np.ones((1,1))
        trail_i = 0
        while not (status == 0).all():
            A_eq = np.random.uniform(-1,1, size=(2, nVar)).round(6)
            b_eq = np.random.uniform(-0.1,0.1, size=(2, 1)).round(6)

            c, A_ub, b_ub, A_eq, b_eq = as_tensor(c, A_ub, b_ub, A_eq, b_eq)  

            soln, status, *_ = homogeneous_solver(c.view(-1, nVar, 1), 
                                                  A_ub.view(1, -1, nVar ), 
                                                  b_ub.view(1, -1, 1), 
                                                  A_eq.view(1, -1, nVar ), 
                                                  b_eq.view(1, -1, 1 ), 
                                                  tol = 1e-8,
                                                  nneg = False,
                                                  want_grad = False)
            trail_i += 1
        if verbose > 1:
            print('\t Find a feasible random lp after %d trails'%(trail_i)) 
        assert (status == 0).all()
    
        m_ub, n = A_ub.shape
        m_eq, _ = A_eq.shape

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

        assert len(u_collect) == num_u, 'len(u_collect) = %d'%len(u_collect)

        i_      =  np.random.permutation(len(u_collect))
        u_train =  u_collect[ i_[: num_u//2   ] ]
        u_test  =  u_collect[ i_[  num_u//2 : ] ]

        status = np.ones(len(u_collect))    # initialize s
        np.random.seed(ind_ins)
        trail_i = 0
        while not (status == 0).all():
            w_c_true    = np.hstack( (np.zeros(1), 
                                np.random.uniform(-sigma_w, sigma_w, 1)) 
                                ).round(6)
            w_A_ub_true = np.hstack( (np.zeros(1), 
                                np.random.uniform(0, sigma_w, 1)) 
                                ).round(6)
            w_b_ub_true = np.hstack( (np.zeros(1), 
                                np.random.uniform(0, sigma_w, 1)) 
                                ).round(6)
            w_A_eq_true = np.hstack( (np.zeros(1), 
                                np.random.uniform(0, sigma_w, 1)) 
                                ).round(6)
            w_b_eq_true = np.hstack( (np.zeros(1), 
                                np.random.uniform(0, sigma_w, 1)) 
                                ).round(6)
            w_A_ub_coef = np.random.randint(0, n, m_ub)
            w_A_eq_coef = np.random.randint(0, n, m_eq)

            plp_true = ParametricLP2((c, A_ub, b_ub, A_eq, b_eq), 
                                     torch.cat(as_tensor(w_c_true, 
                                                        w_A_ub_true, 
                                                        w_b_ub_true,
                                                        w_A_eq_true,
                                                        w_b_eq_true)).view(-1), 
                                     torch.LongTensor(w_A_ub_coef),
                                     torch.LongTensor(w_A_eq_coef))

            c_, A_ub_, b_ub_, A_eq_, b_eq_ = plp_true.generate(tensor(u_collect).view(-1,1), plp_true.weights)
            soln, status, *_    = homogeneous_solver(c_, A_ub_, b_ub_, A_eq_, b_eq_, tol = 1e-8)

            trail_i += 1
            if trail_i == 20:
                if verbose >=1:
                    print('Reaching 20 consective failure, re-sample u values')
                u_collect = np.empty(0)
                i_u = 1
                while i_u < num_u:
                    u = np.random.uniform(-u_range, u_range, 1).round(2)
                    if len(u_collect) == 0:
                        u_collect = u
                    elif min(abs(u_collect - u))>= 1e-2:
                            u_collect = np.vstack((u_collect, u))
                            i_u += 1

                assert len(u_collect) == num_u, 'len(u_collect) = %d'%len(u_collect)

                i_      =  np.random.permutation(len(u_collect))
                u_train =  u_collect[ i_[: num_u//2   ] ]
                u_test  =  u_collect[ i_[  num_u//2 : ] ]

                status = np.ones(len(u_collect))    # initialize s
        
                trail_i = 0
            
        if verbose >=1:
            print('\t Find a feasible PLP after %d trails'%(trail_i)) 

        c, A_ub, b_ub, A_eq, b_eq = as_numpy(c, A_ub, b_ub, A_eq, b_eq)
        plp_data = {'lp':{'c': c,'A_ub': A_ub,'b_ub': b_ub,'A_eq': A_eq,'b_eq': b_eq},
                    'weights':{'w_c': w_c_true, 
                               'w_A_ub': w_A_ub_true,
                               'w_b_ub': w_b_ub_true,
                               'w_A_eq': w_A_eq_true,
                               'w_b_eq': w_b_eq_true,
                               'w_A_ub_coef': w_A_ub_coef,
                               'w_A_eq_coef': w_A_eq_coef},
                    'u':{'train': u_train, 'test': u_test}
                   }
        save_file_directory = directory + 'InvPLP_ins_2/%dvar%dcons'%(nVar, nCons)
        if not os.path.exists(save_file_directory):
            os.makedirs(save_file_directory)

        _file_record = save_file_directory + '/%dvar%dcons_InvPLP%d.pkl'%(nVar, nCons, ind_ins)
        with open(_file_record, 'wb') as output:
            pickle.dump(plp_data, output)

# Class defining the corresponding parametric linear programs
class ParametricLP2(ParametricLP):
    """
    vectorized code for generating parametric LPs
    """
    def __init__(self, LP, weights, nidx_A_ub, nidx_A_eq):
        super().__init__(weights)
        c, A_ub, b_ub, A_eq, b_eq  = LP
        m_ub, n = A_ub.shape
        m_eq, _ = A_eq.shape
        
        self.ref_c     = c.view(1, -1)
        self.ref_A_ub  = A_ub.unsqueeze(0)
        self.nidx_A_ub = nidx_A_ub.view(1, m_ub)
        self.midx_A_ub = torch.arange(m_ub).view(1, m_ub)
        self.ref_b_ub  = b_ub.view(1, -1)
        
        self.ref_A_eq  = A_eq.unsqueeze(0)
        self.nidx_A_eq = nidx_A_eq.view(1, m_eq)
        self.midx_A_eq = torch.arange(m_eq).view(1, m_eq)
        self.ref_b_eq  = b_eq.view(1, -1)
        self.nneg      = False
                
    def generate(self, u, weights):
        k = len(u)
        _, m_ub, n = self.ref_A_ub.shape 
        _, m_eq, _ = self.ref_A_eq.shape 
        w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = weights.view(-1, 10, 1).unbind(1)

        # Batch of (k, n, 1) c vectors
        plp_c = (self.ref_c + w1 + w2*u).unsqueeze(2)
        plp_c = normalize_c(plp_c)

        # Batch of (k, m, n) A_ub matrices
        kidx = torch.arange(k).view(k, 1).expand(-1, m_ub)
        midx = self.midx_A_ub.expand(k, -1)
        nidx = self.nidx_A_ub.expand(k, -1)
        plp_A_ub = self.ref_A_ub.repeat(k, 1, 1).contiguous()
        plp_A_ub[kidx, midx, nidx] += (w3 + w4*u).expand(-1, m_ub)  # Expand m to make same for each row of A
        
        # Batch of (k, m, 1) b_ub vectors
        plp_b_ub = (self.ref_b_ub + w5 + w6*u).unsqueeze(2)
        
        # Batch of (k, m, n) A_ub matrices
        kidx = torch.arange(k).view(k, 1).expand(-1, m_eq)
        midx = self.midx_A_eq.expand(k, -1)
        nidx = self.nidx_A_eq.expand(k, -1)
        plp_A_eq = self.ref_A_eq.repeat(k, 1, 1).contiguous()
        plp_A_eq[kidx, midx, nidx] += (w7 + w8*u).expand(-1, m_eq)  # Expand m to make same for each row of A
        
        # Batch of (k, m, 1) b_ub vectors
        plp_b_eq = (self.ref_b_eq + w9 + w10*u).unsqueeze(2)
               
        return plp_c, plp_A_ub, plp_b_ub, plp_A_eq, plp_b_eq

def read_pkl_file(file):
    with open(file, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        
    return data

def read_InvPLP_2(file_name):
    """
    read PLP instance from file
    """
    lp_file = read_pkl_file(file_name)
    
    c    = lp_file['lp']['c']
    A_ub = lp_file['lp']['A_ub']
    b_ub = lp_file['lp']['b_ub']
    A_eq = lp_file['lp']['A_eq']
    b_eq = lp_file['lp']['b_eq']

    w_c    = lp_file['weights']['w_c']
    w_A_ub = lp_file['weights']['w_A_ub']
    w_b_ub = lp_file['weights']['w_b_ub']
    w_A_eq = lp_file['weights']['w_A_eq']
    w_b_eq = lp_file['weights']['w_b_eq']
    w_A_ub_coef = lp_file['weights']['w_A_ub_coef']
    w_A_eq_coef = lp_file['weights']['w_A_eq_coef']
    
    u_train = lp_file['u']['train']
    u_test  = lp_file['u']['test']
    
    return (c, A_ub, b_ub, A_eq, b_eq),\
           (w_c, w_A_ub, w_b_ub, w_A_eq, w_b_eq, w_A_ub_coef, w_A_eq_coef),\
           (u_train, u_test)

# sample w_init for experiment
def InvPLP_random_initialization(sigma_w, weights_true):
    """
    Randomly sample w
    """
    w_c, w_A_ub, w_b_ub, w_A_eq, w_b_eq = as_tensor(*weights_true)

    w_c_init = torch.ones_like(w_c).uniform_(-sigma_w, sigma_w)
    w_A_ub_init = torch.ones_like(w_A_ub).uniform_(0, sigma_w)
    w_b_ub_init = torch.ones_like(w_b_ub).uniform_(0, sigma_w)
    w_A_eq_init = torch.ones_like(w_A_eq).uniform_(0, sigma_w)
    w_b_eq_init = torch.ones_like(w_b_eq).uniform_(0, sigma_w)

    return (w_c_init, w_A_ub_init, w_b_ub_init, w_A_eq_init, w_b_eq_init)

def transform_constraints(A, b, nidx_A, x_target, u_train):            
    """
    rewrite A(u, w) x_target <= b(u, w) to A(u, x_target) w <= b(u, x_target)
         or G(u, w) x_target == h(u, w) to G(u, x_target) w == h(u, x_target)

    Note, x_target is known, w is unknown.
    """
    _, m, n = A.shape
    k = len(u_train)
    
    assert isinstance(A, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(nidx_A, torch.LongTensor)
    assert u_train.shape == (k,1), 'u_train.shape%s'%([i_ for i_ in  u_train.shape])
    assert b.shape == (1, m, 1), 'b.shape%s'%([i_ for i_ in  b.shape])
    
    u_train = u_train.repeat(1, m).view(-1,1)
    nidx_A  = nidx_A.view(-1)

    b = b.repeat(k, 1, 1)
    A = A.repeat(k, 1, 1)
    b_  = (b - A@x_target).view(-1,1)

    coln1 = x_target[:,nidx_A].view(-1,1)
    coln2 = x_target[:,nidx_A].view(-1,1) * u_train
    coln3 = torch.ones_like(coln2)*-1
    coln4 = torch.ones_like(u_train)*u_train*-1
    
    A_ = torch.cat((coln1, coln2, coln3, coln4), 1)
    
    return A_, b_

# function used for training InvPLP
def inverse_parametric_linprog(u_train, 
                               x_target, 
                               plp,
                               inner_tol        =  1e-8,  # by default, use high-precision forward solver for the inner problem.
                               outer_method     =  None,
                               grad_mode        =  None,    
                               runtime_limit    =  30,
                               verbose          =  0,
                               collect_evals    =  True):

    """
    solve ILOP (with ineq. and eq. constraints only) with COBYLA or SLSQP to learn objective, ineq. and eq. constraints

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

    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training if loss<loss_tol
    outer_tol        =  1e-15       # tolerance for scipy.opt package
    outer_maxiter    =  10000       # iteration for scipy.opt package (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    outer_maxeval    =  10000       # max iteration for evaluation fop (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance
    boundedness_cons =  False       
    
    assert verbose in [0,1,2], VERBOSE_ERROR_MESSAGE
    assert grad_mode in (None, 'numerical_grad', 'backprop', 'implicit', 'direct', 'cvxpylayer'), GRAD_MODE_ERROR_MESSAGE
    assert outer_method in ('RS', 'COBYLA', 'SLSQP'), OUTER_METHOD_MESSAGE

    num_evals      = 0
    rankd_G        = None
    curr_s         = None  # Current parameter vector being queried by scipy minimize
    curr_s_rep     = None  # Repeated version of curr_w, to collect Jacobian entries
    curr_w_recover = None  
    curr_lp        = None
    curr_loss      = 999
    curr_soln      = None                # The linprog results to the LPs for curr_w
    curr_target_feasibility_ub = None    # The b_ub - A_ub @ x_target residuals for curr_w
    curr_target_feasibility_eq = None    # The b_eq - A_eq @ x_target residuals for curr_w    
    best_s         = None
    best_loss      = float(999)
    s_collect      = []
    loss_collect   = []
    requires_grad  = (outer_method != 'COBYLA') and (grad_mode in ('backprop', 'implicit', 'direct', 'cvxpylayer'))
    layer          = None   # check cvxpylayer object as None, which will be initialized only if grad_mode == 'cvxpylayer'

    # update curr_w = w
    def is_s_curr(s):
        return (curr_s is not None) and np.array_equal(s, curr_s.detach().numpy())
    
    # see "Equality constraints affinely-dependent in w" in Appendix B of the paper
    # eliminate affinely-dependent equality constraint sets by reparametrizing the ILOP search over a lower-dimensional space
    # P@s = w + G@h, 
    def transform_eq_constrain(u_train, 
                               plp,
                               x_target):
        """
        compute P, inverse(P), G and inverse(G)
        """
        k = len(u_train)
        _, m_eq, _ = plp.ref_A_eq.shape   
        n_w = 4
        # transform A_eq(u, w)x_target = b_eq(u, w) to G(u,x_target)@w = h(u,x_target)
        G, h = transform_constraints(A = plp.ref_A_eq,
                                     b = plp.ref_b_eq.unsqueeze(2),
                                     nidx_A   = plp.nidx_A_eq,
                                     u_train  = u_train.view(-1,1),
                                     x_target = x_target)
        assert G.shape == (k*m_eq, n_w)
        assert h.shape == (k*m_eq, 1)

        n_w_     = n_w - torch.matrix_rank(G)
        G_pinv = torch.pinverse(G)
        assert G_pinv.shape == (n_w, k*m_eq)

        P = (torch.eye(n_w, ).type(G.dtype) - G_pinv@G)[:,:n_w_]
        assert P.shape == (n_w, n_w_)

        P_pinv = torch.pinverse(P)
        assert P_pinv.shape == (n_w_, n_w)
        
        return G.unsqueeze(0), G_pinv.unsqueeze(0), \
               P_pinv.unsqueeze(0), P.unsqueeze(0), \
               h.unsqueeze(0), torch.matrix_rank(G)
        
    # Vectorized code to work with N rows of weights
    # reparameterize w as: s = P_pinv(w - G_pinv@h )
    def reparametrize_w(w):     
        w_  = w[:, -4:].view(-1, 4,1)
        s_ = P_pinv@(w_ - G_pinv@h_)
        s = torch.cat((w[:,:6].view(-1,6), s_.squeeze(2)), 1)
        return s    
    
    # recover w from s, using  w = G_pinv@h + P@s
    def recover_w(s):
        len_s = 4 - rank_G
        if len_s >0:
            s_ = s[:, -len_s:].view(-1, 4, 1)
        else:
            s_ = torch.empty((len(s),0,1))
        w_ = G_pinv@h_ + P@s_
        w_recovered =  torch.cat((s[:,:6].view(-1,6), w_.squeeze(2)),1 )
        return w_recovered

    # the algorithm search in s space
    # Returning cur_lp, i.e., PLP(w, u)
    def get_curr_lp(s):
        nonlocal curr_s, curr_s_rep, curr_lp, curr_soln, curr_loss, curr_w_recover
        nonlocal curr_target_feasibility_ub
        nonlocal curr_target_feasibility_eq
        if not is_s_curr(s):
            curr_s     = torch.tensor(s.ravel(), requires_grad=requires_grad)  # Deliberately copy s, enable gradient for SLSQP
            curr_s_rep = curr_s.clone().view(1, -1).repeat(len(u_train), 1)    # copy s for |u_train| copies for faster backpropagation
            curr_w_recover = recover_w(curr_s_rep) #recover w from s, and generate PLP
            curr_lp   = plp.generate(u_train, curr_w_recover)
            curr_soln = None
            curr_loss = None
            curr_target_feasibility_ub = None
            curr_target_feasibility_eq = None

            # Retain the computation graph
            if requires_grad:
                curr_s_rep.retain_grad()
        return curr_lp
    
    # Return curr_sol by solving PLP(w, u)
    def get_curr_soln(s):  # w is ndarray
        nonlocal curr_soln
        if not is_s_curr(s) or curr_soln is None:
            curr_soln = solve_lp(get_curr_lp(s), 
                                 grad_mode, 
                                 inner_tol, 
                                 plp.nneg,
                                 layer = layer)
        return curr_soln
    
    # get the status code from the homogeneous solver
    def get_curr_status(s):
        return get_curr_soln(s)[1]
        
    def get_curr_loss(s):  # w is ndarray
        nonlocal curr_loss
        if not is_s_curr(s) or curr_loss is None:
            c = get_curr_lp(s)[0]
            x, status, *_ = get_curr_soln(s)
            l = loss_fn(c, x_target, x).view(-1)
            if boundedness_cons:
                l.masked_fill_(status >= 1, 1)
            curr_loss = l.mean()
        return curr_loss

    # Capture the primal-dual gap
    def get_curr_pdgap_residual(s):  # w is ndarray
        return -get_curr_soln(s)[3] + 1e-5

    # Compute constraint residual, i.e, b - A@x
    def calc_residual(b, A, x): 
        if A is None:
            return None
        # Using _bsubbmm().neg_() avoids temporaries and is slightly faster than (b - bmm(A, x))
        return _bsubbmm(b, A, x).neg_().view(-1)  # -(-b + A@x) == b - A@x

    # Check A_ub @ X <= b_ub
    def get_curr_target_feasibility_ub(s):  # w is ndarray
        nonlocal curr_target_feasibility_ub
        if not is_s_curr(s) or curr_target_feasibility_ub is None:
            _, A_ub, b_ub, _, _ = get_curr_lp(s)
            curr_target_feasibility_ub = calc_residual(b_ub, A_ub, x_target)
        return curr_target_feasibility_ub

    # Check A_eq @ X == b_eq
    def get_curr_target_feasibility_eq(s):  # w is ndarray
        nonlocal curr_target_feasibility_eq
        if not is_s_curr(s) or curr_target_feasibility_eq is None:
            _, _, _, A_eq, b_eq = get_curr_lp(s)
            curr_target_feasibility_eq = calc_residual(b_eq, A_eq, x_target)
        return curr_target_feasibility_eq
    
    # Return feasibility violation
    def get_curr_target_feasibility_violation(s):  # w is ndarray
        r_ub = get_curr_target_feasibility_ub(s)
        r_eq = get_curr_target_feasibility_eq(s)
        r = 0.0
        if r_ub is not None: r = max(r, -float(r_ub.detach().min()))        # Maximum inequality violation
        if r_eq is not None: r = max(r,  float(r_eq.detach().abs().max()))  # Maximum equality violation
        return r
    
    # Return primal-dual gap violation
    def get_curr_pdgap_violation(s):  # w is ndarray
        r = get_curr_pdgap_residual(s)
        return float(r.detach().max())

    def get_curr_status_codes(s):
        return [int(s) for s in get_curr_status(s).view(-1)]
    
    # Reset the gradient of curr_w
    def zero_w_grad():
        if curr_s.grad is not None:
            curr_s.grad.zero_()
    
    # Reset the gradient of curr_w_rep 
    def zero_w_rep_grad():
        if curr_s_rep.grad is not None:
            curr_s_rep.grad.zero_()

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
            jac[start+j:start+size:m,:] = curr_s_rep.grad.data

    # Fills jac[start:start+size,:] with the Jacobian of primal-dual gap residuals r
    # with respect to w. Since residual r[i] is a dependent variable of only w_rep[i],
    # we can compute the Jacobian in a single backward pass.
    def fill_jac_pdgap(jac, r, start, size):
        zero_w_rep_grad()
        r.backward(torch.ones_like(r), retain_graph=True)
        jac[start:start+size,:] = curr_s_rep.grad.data
    
    
    # Function to evaluate loss at point w in weight space
    def f(s):
        nonlocal num_evals, best_loss, best_s

        if time.time() - io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
        if num_evals >= outer_maxeval:
            raise EvaluationLimitExceeded()
        num_evals += 1
        l = float(get_curr_loss(s))

        if max(abs(curr_s)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        # Record the best loss for feasible w, in case we later hit evaluation limit
        if l < best_loss and (get_curr_status(s) == 0 ).all():
            v = get_curr_target_feasibility_violation(s)
            # Record loss only if outer problem constraints are satisfied and fop is feasible
            if v < violation_tol:
                best_loss = float(l)
                best_s = s.copy()     
        loss_collect.append(best_loss)
        s_collect.append(best_s)

        if verbose == 2:
            print('   [%d]  f(%s) \n \t  l=[%.8f]  infea=[%.4f]  pd=%s  s=%s'% (num_evals, s.ravel().round(4), l,
                                                get_curr_target_feasibility_violation(s),
                                                get_curr_pdgap_violation(s) if boundedness_cons else None,
                                                get_curr_status_codes(s)))
        if best_loss < loss_tol:
            raise LossToleranceExceeded()
            
        return l

    # Function to evaluate gradient of loss at w
    def f_grad(s):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] df(%s)'% (num_evals, s.ravel().round(4)))

        zero_w_grad()
        l = get_curr_loss(s)

        if max(abs(curr_s)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        l.backward(retain_graph=True)
        return curr_s.grad.numpy()

    # Function to evaluate ineq. constraint residuals of outer problem at w
    def g(s):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d]  g(%s)'% (num_evals, s.ravel().round(4)))

        r = get_curr_target_feasibility_ub(s).detach().numpy() # x_target is feasible
        if max(abs(curr_s)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        if boundedness_cons:
            r = np.concatenate((r, get_curr_pdgap_residual(s).detach().numpy()))
        return r

    # Function to evaluate gradient of ineq. constraint residuals of outer problem at w
    def g_grad(s):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] dg(%s)'% (num_evals, s.ravel().round(4)))

        r1 = get_curr_target_feasibility_ub(s)
        if max(abs(curr_s)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        m1 = len(r1) if r1 is not None else 0
        if boundedness_cons:
            r2 = get_curr_pdgap_residual(s)
            m2 = len(r2)
            jac = np.empty((m1 + m2, len(s)))
            fill_jac_target_feasibility(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)
            fill_jac_pdgap(jac, r2, m1, m2)               # Fill wrt boundedness
        else:
            jac = np.empty((m1, len(s)))
            fill_jac_target_feasibility(jac, r1,  0, m1)  # Fill wrt training feasibility (inequalities only)
        return jac

    # compute P, inverse(P), G, inverse(G) and h only once since they do not change during the search
    G_, G_pinv, P_pinv, P, h_, rank_G = transform_eq_constrain(u_train, 
                                                                plp,
                                                                x_target)
    if verbose >=1:
        print('    ++++ w_init\n', plp.weights.view(-1).detach().numpy())
    s_init = reparametrize_w(plp.weights.view(1,-1))
    if verbose >=1:
        print('    ++++ s_init after reparameterization\n', s_init.view(-1).detach().numpy())
    
    best_s  = np.ones((len(s_init.view(-1)),))*999


    if collect_evals:
        f = collect_calls(f)
        g = collect_calls(g)
        f_grad = collect_calls(f_grad)
        g_grad = collect_calls(g_grad)

    if outer_method == 'COBYLA':
        f_jac = None
        g_jac = None
    elif outer_method == 'SLSQP':
        if grad_mode == 'numerical_grad':
            f_jac = None
            g_jac = None
        elif grad_mode in ('backprop', 'implicit', 'direct'):
            f_jac = f_grad
            g_jac = g_grad    
        elif grad_mode == 'cvxpylayer':
            f_jac = f_grad
            g_jac = g_grad    

            # initialize an optimizatin layer with cvxpylayer once
            layer = create_cvx_layer(plp.generate(u_train, plp.weights), plp.nneg)

        else:
            raise Exception('Invalid grad_mode for SLSQP')


    constraints = [{'type': 'ineq', 'fun': g, 'jac': g_jac}]

    io_tic = time.time()
    try:
        res = opt.minimize(f, s_init.detach().numpy().ravel(), 
                           jac     = f_jac, 
                           constraints = constraints, 
                           tol     = outer_tol,
                           method  = outer_method,
                           options = {'maxiter': outer_maxiter})
        if not res.success:
            if verbose >=1:
                print("WARNING: scipy minimize returned failure")
        
    except (EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded, UnboundedWeightsValue) as e:
        if verbose >=1 :
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

    # recover best_w from best_s
    if (best_s == 999).any(): # failed to find a feasible solution w, return dummy parameter w
        w_lrn_recover = np.ones((10,))*999
    else: # recover feasible solution w from best_s
        w_lrn_recover = recover_w(tensor(best_s).view(-1, len(best_s))).detach().numpy().ravel()
    

    # recover w_collect from s_collect
    if len(s_collect) > 0:
        s_collect = tensor(np.vstack(s_collect))
        s_collect_1 = s_collect[(s_collect == 999).any(1) == True] # w_collect == 999 if s_collect == 999
        s_collect_2 = s_collect[(s_collect == 999).any(1) == False] # recover w_collect if corresponding s_collect is not 999
        w_collect_2_recover = recover_w(tensor(s_collect_2).view(-1, len(curr_s))).detach().numpy()
        w_collect = np.vstack((np.ones( (len(s_collect_1), 10))*999, w_collect_2_recover))
        
        # sanity check, w recovered from best_s must be the same as the last elemnt of w_collect recovered from s_collect
        assert (w_collect[-1] == w_lrn_recover).all(), 'w_collect[-1] = %s\n w_lrn_recover = %s'%(w_collect[-1], w_lrn_recover)
        # sanity check, best_loss much be the same as the last element of loss_collect
        assert best_loss == loss_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_loss, loss_collect[-1])
    else: # failed to find a feasible solution s
        assert (best_s == 999).all()
        assert len(loss_collect) ==0
        w_collect = best_s
        loss_collect.append(best_loss) 

    if verbose >= 1:
        print("++++           call [f %d times], [f_grd %d times], [g %d times], [g_grd %d times]" \
                    % (len(f.calls), len(f_grad.calls), len(g.calls), len(g_grad.calls)) )
        print("++++           initial_loss [%.8f], best_loss [%.8f], runtime [%.4f s]"%(loss_collect[0], best_loss, (io_toc-io_tic)))
        print("++++           best_s %s"%(best_s.ravel()))
        print("++++           best_w %s"%(w_lrn_recover))
    if collect_evals:
        evals = {
            'f': {'calls': f.calls, 'rvals': f.rvals, 'times': f.times, 'loss': loss_collect, 'w': w_collect},
            'g': {'calls': g.calls, 'rvals': g.rvals, 'times': g.times},
            'f_grad': {'calls': f_grad.calls, 'rvals': f_grad.rvals, 'times': f_grad.times},
            'g_grad': {'calls': g_grad.calls, 'rvals': g_grad.rvals, 'times': g_grad.times},
            'res': {'runtime': (io_tic, io_toc), 'best_l': best_loss, 'best_w':w_lrn_recover},
        }

        return best_loss, evals
    
    return best_loss

def InvPLP_random_search(u_train, 
                         x_target,
                         plp,
                         w_init,
                         inner_tol       =  1e-8,  # by default, use high-precision forward solver.
                         runtime_limit   =  30,
                         verbose         =  0,
                         collect_evals   =  True):
    """
    solve ILOP (with ineq. and eq. constraints) with random search to learn objective, ineq. and eq. constraints

    Input:
        Randomly sample w repeatedly from the same distribution as generating the true PLP (see InvPLP_random_initialization)

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
    best_w        =  np.ones((10,))*999
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
        for iter_i in range(outer_maxeval):
            w_random = InvPLP_random_initialization(sigma_w, [w_init[i:i+2] for i in range(0, len(w_init),2) ] )
            w = np.hstack(as_numpy(*w_random))
            f(w)
        
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
            'f': {'calls': f.calls, 'rvals': f.rvals, 'times': f.times, 'loss': loss_collect, 'w': f.calls},
            'res': {'runtime': (io_tic, io_toc), 'best_l': best_loss, 'best_w':best_w}
        }
        return best_loss, evals
    
    return best_loss

def InvPLP_experiment(nVar,
                      nCons,
                      ind_ins,
                      inner_tol,
                      outer_method,
                      grad_mode,
                      plp_class, 
                      runtime_limit,
                      directory,
                      verbose):
    """
    run ILOP experiment for a specific plp instance

    Input:
          nVar, nCons: num. variabel, num. constriants,
              ind_ins:  instance index
            inner_tol: tolerance of forward solver
         outer_method: algorithms for solving the bi-level ILOP
            grad_mode: backward methods for computing gradients
            plp_class: parametric LP class (vectorized code for generating LPs)
        runtime_limit: runtime limit for solving ILOP (in seconds)
    """

    PLP_filename = directory + '%dvar%dcons_InvPLP%d.txt'\
            %(nVar, nCons, ind_ins)
    if verbose >= 1:
        print('************************************')
        print('ILOP Exp1b %dvar, %d ineq. con, 2 eq. cons, ins%d '%(nVar, nCons, ind_ins) )
        if outer_method =='RS':
            print('\twith Random Search [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
        else:
            if outer_method == 'COBYLA':
                print('\twith COBYLA [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
            elif outer_method == 'SLSQP':
                print('\twith SLSQP [%s] %s(tol = %.2e), '
                        %(grad_mode, 'cvx' if grad_mode == 'cvxpylayer' else 'homogeneous', inner_tol))

        print('\tloss_fn = [AOE]' )

    # Initialization - find x_target from w_true
    file_name = directory + 'InvPLP_ins_2/%dvar%dcons/%dvar%dcons_InvPLP%d.pkl'%(nVar, nCons,nVar, nCons,ind_ins)
    lp_, w_, u_ = read_InvPLP_2(file_name)
    u_train, u_test = as_tensor(*u_)
    w_true = w_[:-2]
    
    w_A_ub_coef = w_[-2]
    w_A_eq_coef = w_[-1]
    
    if verbose == 2:
        print('len(u_train) = [%d] len(u_test) = [%d]'%(len(u_train), len(u_test)))
    u_train  = u_train.view(-1,1)
    
    plp_true = plp_class(as_tensor(*lp_), 
                         torch.cat(as_tensor(*w_true)).view(-1), 
                         torch.LongTensor(w_A_ub_coef).view(-1),
                         torch.LongTensor(w_A_eq_coef).view(-1))
                         
    if verbose >=1: 
        print('[Compute x_target] Initlize the InvPLP with w_true, and then compute Soln_true as the targets')
    soln_true, status, *_   =  homogeneous_solver(*plp_true.generate(u_train, plp_true.weights), 
                                                nneg = plp_true.nneg,
                                                tol = 1e-8)

    x_target = soln_true.detach().clone()
    assert (status == 0).all(), 'Some FOPs of the true PLP cant be solved with linprog, s =%s'%(status.numpy().ravel())

    # Initialization - find a feasible w_init to start inverse_parametric_linprog
    if verbose == 2:
        print('Initialize with random w_init')
        
    torch.manual_seed(ind_ins)
    for iter_i in range(1, 1000, 1):
        w_init = InvPLP_random_initialization(1, w_true)
        plp_init = plp_class(as_tensor(*lp_), 
                             torch.cat(as_tensor(*w_init)).view(-1), 
                             torch.LongTensor(w_A_ub_coef).view(-1),
                             torch.LongTensor(w_A_eq_coef).view(-1))
        
        soln_init, status, *_   =  homogeneous_solver(*plp_init.generate(u_train, plp_init.weights),
                                                      tol = 1e-8, 
                                                      nneg = plp_init.nneg)
    
        if (status == 0).all():
            break
        else:
            if verbose == 2:
                print('\t infeasible w_init, status = %s'%( status.numpy() ) )

    if verbose >= 1:
        print(' [Initialization] find a set of feasible w_init %s after %d fop evals'
                  %( torch.cat(w_init,0).numpy().round(4), iter_i))

    if outer_method == 'RS':
        _, res = InvPLP_random_search( u_train         = u_train, 
                                       x_target        =  x_target, 
                                       plp             =  plp_init,
                                       w_init          =  plp_init.weights,
                                       inner_tol       =  inner_tol,
                                       runtime_limit   =  runtime_limit,
                                       verbose         =  verbose,
                                       collect_evals   =  True)

    else:
        _, res = inverse_parametric_linprog(u_train          =  u_train, 
                                            x_target         =  x_target, 
                                            plp              =  plp_init,
                                            inner_tol       =  inner_tol,
                                            outer_method     =  outer_method,
                                            grad_mode        =  grad_mode,
                                            runtime_limit    =  runtime_limit,
                                            verbose          =  verbose,
                                            collect_evals    =  True)

    if outer_method == 'SLSQP':
        save_file_directory = directory + '/InvPLP_ins_2/%dvar%dcons/%s_%s'%(nVar, nCons, outer_method, grad_mode)
    else:
        save_file_directory = directory + '/InvPLP_ins_2/%dvar%dcons/%s'%(nVar, nCons,outer_method)
    if not os.path.exists(save_file_directory):
        os.makedirs(save_file_directory)

    _file_record = save_file_directory + '/%dvar%dcons_InvPLP%s_%s_res.pkl'\
                %( nVar, nCons, ind_ins, 'AOE')
    with open(_file_record, 'wb') as output:
        pickle.dump(res, output)
            
    if verbose >= 1:
        print('************************************\n')
    return res

def InvPLP(lp_size, 
           nIns, 
           plp_class, 
           runtime_limit,
           framework,
           verbose,
           directory):

    """
    Complete experiment for inverse parametric linear programming problems with ineq. and eq. constraints.  

    Input:
          lp_size:  a list of tuple (nVar, nCons)
             nIns: number of instances for each lp_size
    runtime_limit: runtime limit for solving the IO problem
        framework: a list of tuple(outer_method, grad_mode)
    """

    nVar, nCons = lp_size
    for framework_i in framework:
        outer_method, grad_mode = framework_i
        for ind_ins in range(nIns):
            _ = InvPLP_experiment(nVar          = nVar,
                                  nCons         = nCons,
                                  ind_ins       = ind_ins,
                                  inner_tol     = 1e-8,
                                  outer_method  = outer_method,
                                  grad_mode     = grad_mode,
                                  plp_class     = ParametricLP2,
                                  runtime_limit = runtime_limit,
                                  directory     = directory,
                                  verbose       = verbose)

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
    
    w_collect  =   np.vstack(as_numpy(res['f']['w']))
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

def InvPLP_data_analysis(lp_size, 
                         nIns,
                         loss_fn,  
                         framework,
                         directory     =  None,
                         plot_callback =  None,
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
    l_RS                =  []
    l_COBYLA            =  []
    l_SLSQP_backprop    =  []
    l_SLSQP_implicit    =  []
    l_SLSQP_direct      =  []
    l_SLSQP_cvxpylayer  =  []

    # w_collect at each f(w)
    # w_collect records the w even it is a bad solution
    w_RS                =  []
    w_COBYLA            =  []
    w_SLSQP_backprop    =  []
    w_SLSQP_implicit    =  []
    w_SLSQP_direct      =  []
    w_SLSQP_cvxpylayer  =  []

    # clock time of each f(w)
    timestep_RS                =  []
    timestep_COBYLA            =  []
    timestep_SLSQP_backprop    =  []
    timestep_SLSQP_implicit    =  []
    timestep_SLSQP_direct      =  []
    timestep_SLSQP_cvxpylayer  =  []

    runtime_RS                =  []
    runtime_COBYLA            =  []
    runtime_SLSQP_backprop    =  []
    runtime_SLSQP_implicit    =  []
    runtime_SLSQP_direct      =  []
    runtime_SLSQP_cvxpylayer  =  []

    for ind_lp in range(len(lp_size)):
        nVar, nCons = lp_size[ind_lp]
        for framework_i in framework:

            outer_method, grad_mode = framework_i
            if outer_method in ('RS','COBYLA'):
                _file_path = directory + 'InvPLP_ins_2/%dvar%dcons/%s'\
                                            %(nVar, nCons, outer_method)
            else:
                _file_path = directory + 'InvPLP_ins_2/%dvar%dcons/%s_%s'\
                                            %(nVar, nCons, outer_method, grad_mode)  

            for ind_ins in range(nIns):
                if verbose > 1:
                    print('\n=======  %d var, %d cons, Ins %d, loss = %s '
                                    %(nVar, nCons, ind_ins, 'AOE'))
                    print('         Outer_method %s, grad_mode %s'
                                    %(outer_method, grad_mode))
                
                result_file = _file_path+'/%dvar%dcons_InvPLP%d_%s'\
                        %(nVar, nCons, ind_ins,\
                          loss_fn.__name__[1:-1])

                w_collect, l_collect, timestep, runtime, best_l, best_w = read_exp_result(result_file)
                num_evals = min((len(w_collect), len(l_collect), len(timestep)) )

                w_collect = w_collect[:num_evals]
                l_collect = l_collect[:num_evals]
                timestep = timestep[:num_evals]

                assert sum(l_collect[-1] - best_l) <= 1e-3, 'Inconsistentency in experiment record: l_collect[-1] =%s, best_l = %s'%(l_collect[-1], best_l)

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

    return  (l_RS,               w_RS,               timestep_RS,               runtime_RS),\
            (l_COBYLA,           w_COBYLA,           timestep_COBYLA,           runtime_COBYLA), \
            (l_SLSQP_backprop,   w_SLSQP_backprop,   timestep_SLSQP_backprop,   runtime_SLSQP_backprop),\
            (l_SLSQP_implicit,   w_SLSQP_implicit,   timestep_SLSQP_implicit,   runtime_SLSQP_implicit),\
            (l_SLSQP_direct,     w_SLSQP_direct,     timestep_SLSQP_direct,     runtime_SLSQP_direct),\
            (l_SLSQP_cvxpylayer, w_SLSQP_cvxpylayer, timestep_SLSQP_cvxpylayer, runtime_SLSQP_cvxpylayer)




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
                              verbose,
                              method_name
                             ):
        if verbose > 1:
            print('         +++ Compute testing error for %dvar %dcons InvPLP%d'%(nVar, nCons, ind_ins) )

        if l_train >= 999:
            if verbose > 1:
                print('             Failed to find feasible soln, thus, testing error = 999' )
            l_mean = 999
            l_median = 999
        else:
            PLP_filename = directory + 'InvPLP_ins_2/%dvar%dcons/%dvar%dcons_InvPLP%d.pkl'\
                    %(nVar, nCons, nVar, nCons, ind_ins)

            lp_, w_, u_ = read_InvPLP_2(PLP_filename)
            _, u_test = as_tensor(*u_)
            w_true = w_[:-2]

            w_A_ub_coef = w_[-2]
            w_A_eq_coef = w_[-1]

            if verbose == 2:
                print('len(u_test) = [%d]'% len(u_test))
            u_test  = u_test.view(-1,1)

            plp = ParametricLP2(as_tensor(*lp_), 
                                     torch.cat(as_tensor(*w_true)).view(-1), 
                                     torch.LongTensor(w_A_ub_coef).view(-1),
                                     torch.LongTensor(w_A_eq_coef).view(-1))
            
            
            c_true, A_ub_true, b_ub_true, A_eq_true, b_eq_true = plp.generate(u_test, plp.weights)
            target_res   =  homogeneous_solver(c_true.detach(),   # computing the x_target(u_test, w_best) using homogeneous_solver with high-precision
                                               A_ub_true.detach(), 
                                               b_ub_true.detach(),
                                               A_eq_true.detach() if A_eq_true is not None else None, 
                                               b_eq_true.detach() if b_eq_true is not None else None, 
                                               tol = 1e-8, 
                                               nneg = plp.nneg)
            x_target = target_res[0].detach().clone()
            
            assert sum(target_res[1]) == 0, 'ind[%d] w_true is not feasible, s = %s'%(ind_ins, target_res[1].detach().numpy())
 
            if (w_test == 999).any():
                l = 999
            else:
                plp.weights  =  tensor(w_test).view(-1)
                                
                c_lrn, A_ub_lrn, b_ub_lrn, A_eq_lrn, b_eq_lrn = plp.generate(u_test, plp.weights)
                soln_test = homogeneous_solver(c_lrn.detach(),    # computing the x_lrn(u_test, w_best) using homogeneous_solver with high-precision
                                               A_ub_lrn.detach(), 
                                               b_ub_lrn.detach(),
                                               A_eq_lrn.detach() if A_eq_lrn is not None else None, 
                                               b_eq_lrn.detach() if b_eq_lrn is not None else None, 
                                               tol = 1e-8, 
                                               nneg = plp.nneg)

            if sum(soln_test[1]) == 0:
                x_final  = soln_test[0].detach().clone()
                l_mean   = loss_fn(c_true, x_target, x_final).mean()  # testing loss = l(c_true, x_true, x_lrn) = c_true|x_true - x_lrn|
                l_median = loss_fn(c_true, x_target, x_final).median()  # testing loss = l(c_true, x_true, x_lrn) = c_true|x_true - x_lrn|
                if verbose >1:
                    print('             Testing error = [%.2e]'%(float(l)) )
            else:
                if verbose >1 :
                    print('             !!!!ind[%d] method[%s] w_lrn lead to infeasible LPs on u_test, s = %s'%(ind_ins, method_name, soln_test[1].detach().numpy()) )
                    print('             !!!!w_lrn = %s'%w_test)

                l_mean = 999
                l_median = 999

            if l_mean < 1e-5:
                l_mean = 1e-5
            if l_median < 1e-5:
                l_median = 1e-5

        return float(l_mean), float(l_median)
    
    def plot_(loss_plot, color, axes):
        axes.boxplot(loss_plot, sym='')
        x_ticks = np.linspace(0.7, 1.3, nIns)
                 # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
        marker = ['+', 'x',     'd',     '^',       's',        'o', ]
        for i in range(len(loss_plot)):
            axes.plot(x_ticks+i, loss_plot[i], 'o', 
                    marker=marker[i], mfc='None',
                    ms=5, mew=1, color = color[i],
                    alpha = 0.5,)
        axes.set_yscale('log')
        axes.set_ylim(5e-6,5e2)
        axes.get_xaxis().set_visible(False)
        axes.set_xlim(0,7)
    """
            
    data = [outer_method][loss/weights/time]
    """
    
    numIns = len(data[0][0])
    numMethod = len(data)
    loss_initial = np.ones((numIns,))*999  # collect initial loss
    loss_train = np.ones((numMethod, numIns))*999  # collect initial loss

    loss_test_mean = np.ones((numMethod, numIns))*999  # collect initial loss
    loss_test_median = np.ones((numMethod, numIns))*999  # collect initial loss

    # collect loss_init and loss_final
    for ind_ins in range(numIns):
        for outer_method_i in range(1, numMethod-1):
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
                ind = max(idx)  # find the last iter before reaching runtime_limit
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
                                                                verbose   = False,
                                                                method_name = method_name[outer_method_i])


                loss_test_mean[outer_method_i][ind_ins] = float(l_test_mean)         
                loss_test_median[outer_method_i][ind_ins] = float(l_test_median)    
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
    loss_train = np.array(loss_train)
    loss_train = np.clip(loss_train, 1e-5, 1e2)
    
    loss_test_mean  = np.array(loss_test_mean)
    loss_test_mean = np.clip(loss_test_mean, 1e-5, 1e2)

    loss_test_median  = np.array(loss_test_median)
    loss_test_median = np.clip(loss_test_median, 1e-5, 1e2)
  
    plot_([loss_train[i,:] for i in range(len(loss_train))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
           color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
           axes = ax[0])
    plot_([loss_test_mean[i,:] for i in range(len(loss_test_mean))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
           color = ['m', 'b',      'k',    'r',      'orange',    'g', ],        
           axes = ax[1])
    plot_([loss_test_median[i,:] for i in range(len(loss_test_median))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
           color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
           axes = ax[2])
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[2].yaxis.set_label_position("right")
    ax[2].yaxis.tick_right()
    return loss_initial, loss_train


def fig_exp1b(lp_size, nIns, loss_fn, framework, directory, verbose = False):
    """
    Generate figures for exp1b

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

        if verbose > 1:
            print('success rate: %s'%(success_rate))

        _ = boxplot_InvPLP(ax          = (ax2, ax3, ax4), 
                           lp_size     = lp_size, 
                           loss_fn     =  loss_fn,
                           nIns        = nIns,
                           end_time  =  time_range[-1], 
                           plp_class   = ParametricLP2,
                           data        =  (res_rs, 
                                           res_c, 
                                           res_s_cvxpylayer,
                                           res_s_backprop, 
                                           res_s_implicit, 
                                           res_s_direct),
                           directory   =  directory,
                           verbose     = verbose)

        ax1.tick_params( axis='x', which='major',labelsize=12 )
        ax1.set_ylim(-5, 105)
        ax1.set_yticks(np.linspace(0,100,5))
        ax1.set_yticklabels([str(i)+'%' for i in np.linspace(0,100,5, dtype = np.int)])
        ax1.set_xlim(-0.1, time_range[-1]+0.1)
        if lp_size[0] == 2:
            ax1.set_xticks([0, 1, 2, 3, 4, 5])
            ax1.set_xticklabels([0, 1, 2, 3, 4, 5],fontsize=12)
        elif lp_size[0] == 10:
            ax1.set_xticks([0, 10, 20, 30])
            ax1.set_xticklabels([0, 10, 20, 30],fontsize=12)

        import matplotlib
        locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
        ax4.yaxis.set_major_locator(locmaj)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax4.yaxis.set_minor_locator(locmin)
        ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        return fig, ax1, ax2

    for i_ in range(len(lp_size)):
        if lp_size[i_][0] == 2: #runtime_limit = 5s for 2D instances
            t_range = time_range[0]
        elif lp_size[i_][0] == 10: #runtime_limit = 30s for 20D instances
            t_range = time_range[1]

        if verbose >=1:
            print('====== Analyze experiment results and plot figure for Exp1b %dvar%dcons'%(lp_size[i_][0], lp_size[i_][1]))
            
        
        fig, *_ = subplot(lp_size[i_],
                          nIns,
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_RS],  # res_RS contains results len(lp_size)*nIns instances,
                                                                               # read results for corresponding lp_size
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_COBYLA],
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_backprop], 
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_implicit], 
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_direct], 
                          [data_i[i_*nIns:(i_+1)*nIns] for data_i in res_SLSQP_cvxpylayer], 
                          t_range)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.subplots_adjust(wspace=0.02, hspace=0)
        fig.savefig(directory+'fig_exp1b_%dvar%dcons.pdf'%(lp_size[i_]),dpi=100)
        
        

def main(directory = None):

    direct = './' if directory is None else directory

    LP_SIZE         = (10,80)        # 6 classes of LPs
    NUM_INS         =  100    
    VERBOSE         =  0    # 2 - print detailed information, e.g., log of each iter in the solving process, 
                            #     initialization, etc. This includes many printing functions during 
                            #     the solving process, thus, using VERBOSE = 2 will influence the experiment results. 
                            # 1 - print basic information, e.g., final results, initialization. 
                            #     all printing function  happen outside of the solving process, 
                            #     thus, does not influence the experiment results 
                            # 0 - don't display any algorithm log           
    
    RUN_TIME            =  30            # runtime limit for experiment

    FRAMEWORK       =  (# outer_method, grad_mode
                        ['RS',          None], 
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

    generate_InvPLP_2(nVar      = nVar,
                      nCons     = nCons,
                      nIns      = NUM_INS,
                      directory = direct,
                      verbose   = VERBOSE)

    InvPLP(lp_size       = (nVar, nCons), 
           nIns          = NUM_INS, 
           plp_class     = ParametricLP2, 
           runtime_limit = RUN_TIME,
           framework     = FRAMEWORK,
           verbose       = VERBOSE,
           directory     = direct)

           
    fig_exp1b(lp_size   = ((nVar, nCons),), 
              nIns      = NUM_INS, 
              loss_fn   = _AOE_,
              framework = FRAMEWORK,
              directory = direct,
              verbose   = VERBOSE)

if __name__ == "__main__":

    main()