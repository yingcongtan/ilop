# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for reproducing the results of NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


import os
import sys

# from util.py importing pre-define functions which are shared for all experiments
sys.path.append("..")
from util import T, tensor, as_tensor, as_numpy, _bsubbmm, ParametricLP
from util import _AOE_, normalize_c, HiddenPrints, create_cvx_layer
from util import EvaluationLimitExceeded, LossToleranceExceeded, RuntimeLimitExceeded, UnboundedWeightsValue
from util import VERBOSE_ERROR_MESSAGE, GRAD_MODE_ERROR_MESSAGE, OUTER_METHOD_MESSAGE

from util import collect_calls, compute_loss_over_time, plot_success_rate
from linprog_solver import linprog_scipy_batch, linprog_batch_std

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import pickle
import scipy.optimize as opt
from math import pi

import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

import warnings
warnings.simplefilter('once')
DT = torch.float64

# define the Nguyen-Dupuis Graph
"""
      flow: defines the flow conservation
        SD: pairs of Source and Destination
node_coord: defines the coordinates of each node for ploting purpose only
"""
graph_ND = {  'flow':    [ # from      to
                          [ 1,       2], # arc 1
                          [ 2,       7], # arc 2
                          [ 1,       4], # arc 3
                          [ 2,       5], # arc 4
                          [ 3,       4], # arc 5
                          [ 4,       5], # arc 6
                          [ 5,       6], # arc 7
                          [ 6,       7], # arc 8
                          [ 3,       8], # arc 9
                          [ 4,       8], # arc 10
                          [ 5,       9], # arc 11
                          [ 6,      10], # arc 12
                          [ 7,      11], # arc 13
                          [ 8,       9], # arc 14
                          [ 9,      10], # arc 15
                          [10,      11], # arc 16
                          [ 8,      12], # arc 17
                          [10,      13], # arc 18
                          [12,      13], # arc 19
                         ],
              'SD':      [[1,11], [1,13], [3,11], [3,13]] ,
              'node_coord': [[ 0,    0],     # node 1
                             [ 1,    0],     # node 2
                             [-1,   -1],     # node 3
                             [ 0,   -1],     # node 4
                             [ 1,   -1],     # node 5
                             [ 2,   -1],     # node 6
                             [ 3,   -1],     # node 7
                             [ 0,   -2],     # node 8
                             [ 1,   -2],     # node 9
                             [ 2,   -2],     # node 10
                             [ 3,   -2],     # node 11
                             [ 1,   -3],     # node 12
                             [ 2,   -3],     # node 13
                            ],
            
          }


def homogeneous_solver(c, 
                       A_ub, 
                       b_ub, 
                       A_eq, 
                       b_eq, 
                       tol, 
                       nneg = False, 
                       want_grad = False, 
                       std_form = False):
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
    b_ub = b_ub.squeeze(2) if b_ub is not None else None
    b_eq = b_eq.squeeze(2) if b_eq is not None else None
    

    if std_form == True:
        x_sol, *rest = linprog_batch_std(c, A_eq, b_eq, 
                                         tol=tol, check_cond = True,
                                         want_pdgap = True, want_grad = want_grad)
    else:
        x_sol, *rest = linprog_scipy_batch(c, A_ub, b_ub, A_eq, b_eq, 
                                          tol = tol, nneg = nneg, check_cond = True,
                                          want_pdgap = True, want_grad = want_grad)
    x_sol.unsqueeze_(2)
    return [x_sol] + rest


def multi_commodity_min_cost_lp_model(graph, 
                                      verbose = False):

    """
    Generate LP mode for minimum cost multi-commodity flow problem

    Input:
       graph: a dict with all relevant info of the graph, including flow, source-destination and arc cost

    Output:
       c, A_ub, b_ub, A_eq, b_eq 
    """
    SD = np.array(graph['SD'])
    num_commodity = len(SD)

    s = SD[:,0]
    d = SD[:,1]
    
    cost = np.array(graph['arc_cost']).reshape(-1,1)
    flow = graph['flow']

    num_node = len(graph['node_coord'] if graph['node_coord'] else len(np.unique(flow)))
    num_arc  = len(cost)
    
    A_eq = np.zeros(( (num_node-1)*num_commodity,   num_arc*num_commodity)  )
    b_eq = np.zeros(( (num_node-1)*num_commodity,   1)  )
    
    def flow_conservation(num_arc, num_node, s, d, flow):
        """
        construct flow conservation constraints for each pair of source-destination
        """
        A_eq = np.zeros((num_node, num_arc))
        b_eq = np.zeros((num_node,1))
        for i in range(len(flow)):
            start, end = flow[i]
            A_eq[start-1, i] = -1
            A_eq[end-1, i] = 1
                  
        b_eq[s-1], b_eq[d-1] = -1, 1
    
        # remove redundant constraints (flow-conservation constraints) of the shortest path lp 
        A_eq = np.delete(A_eq, np.array(d)-1, axis = 0)
        b_eq = np.delete(b_eq, np.array(d)-1, axis = 0)
        
        if verbose == 1:   print('[A_eq|b_eq]\n', np.hstack((A_eq, b_eq)))    
        if verbose == 2:       
            print('[A_eq|b_eq]')
            print('    #Arc:   %s   | b_eq'%',  '.join([str(indx_+1) for indx_ in range(num_arc)]) )
            i_ = 0
            for node_i in range(num_node):
                if node_i+1 != d:
                    print('Node [%d]: %s | %s'%(node_i+1, A_eq[i_],b_eq[i_]))
                    i_ += 1

        return A_eq, b_eq
    
    for i_ in range(num_commodity):
        s_, d_ = s[i_], d[i_]
        A_eq_, b_eq_ = flow_conservation(num_arc, num_node, s_, d_, flow)
        A_eq[i_*(num_node-1): (i_+1)*(num_node-1), i_*num_arc: (i_+1)*num_arc] = A_eq_
        b_eq[i_*(num_node-1): (i_+1)*(num_node-1)] = b_eq_
    
    if verbose >=1:
        print('[c]\n', cost.ravel())
    # create seperate set of c for each pair of source-destination
    c = np.vstack([cost for i_ in range(num_commodity)])
    
    
    A_ub = np.zeros((  num_arc,      num_arc*num_commodity)  )
    b_ub = np.array(graph['capacity']).reshape(-1,1)
    for sd_i in range(len(SD)):
        for i_ in range(num_arc):
            A_ub[i_, i_ + sd_i*num_arc] = 1
            
    return c[None,:,:], A_ub[None,:,:], b_ub[None,:,:], A_eq[None,:,:], b_eq[None,:,:]


def plot_opt_path(graph, x, ax, tol):

    """
    plot graph and solution x
    """

    if ax is None:
        ax = plt.gca()
        
    coord = np.array(graph['node_coord'])
    arc = np.vstack(graph['flow'])-1
    num_arc = len(arc)
    num_sd = len(graph['SD'])
    def path(graph, x, coord, arc, *args, **kwargs):
        for i in range(len(arc)):
            if x[i] >= tol:
                ax.plot(*np.vstack(coord[arc[i]]).T,
                                   alpha = float(x[i]) if x[i]<1 else 1,
                                   *args, **kwargs)
    
    num_arc = len(graph['flow'])
    
    ax.plot(*coord.T, 'o', mfc='None', mec = 'grey', mew = 4, ms= 15, alpha=0.5)
    for i in range(num_arc):
        ax.plot(*np.vstack(coord[arc[i]]).T, '-', color = 'grey', alpha=0.5, linewidth = 5)
        
    # assign different colors and linestyles for each pair of source-destination
    color = ['r','b','g','k']
    linestyle = ['-', '--', '-.', ':']
    linewidth = [12, 10, 6, 6]
    for i_ in range(num_sd):
        path(graph, x[i_*num_arc:(i_+1)*num_arc], coord, arc,  
             color = color[i_], 
             linewidth = linewidth[i_], 
             linestyle = linestyle[i_])
        
    ax.axis('off')
    ax.set_aspect('equal')


def read_pkl_file(file):
    with open(file, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        
    return data

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

class ParametricND(ParametricLP):
    """
    parametric MCF class with vecotrized code for PLP generation

    three features: length (graph['arc_cost]), toll (graph['toll']), t (time)
                    Note, length and toll are arc-specific features which are fixed (a local feature), 
                          and t varies (a global feature)
    five parameters: w1, w2, w3, w4, w5, s.t. w3+w4+w5 = 1

    u = [1, t, length], w_c = [w3, w4, w5]
    c = length + w1*toll + w2*length*( sin(w3 + w4*t+w5*length) + 1 ) 

    b = 1 + w6 + w7*length
    """

    def __init__(self, LP, weights):
        super().__init__(weights)
        _, A_ub, _, A_eq, b_eq  = LP
        self.ref_A_ub = A_ub
        self.ref_A_eq = A_eq
        self.ref_b_eq = b_eq
        self.nneg     = True
        self.w_normalize_idx = [2,3,4]

    def generate(self, 
                 t, 
                 weights, 
                 graph, 
                 plot_param = False):
        
        _, m, n = self.ref_A_ub.shape
        num_sd  = n//m
        k       = len(t)
        w1, w2, w3, w4, w5, w6, w7 = weights.view(-1, 7, 1).unbind(1)
        t       = t.view(k, 1, 1).repeat(1, n, 1)

        length  = torch.cat([tensor(graph['arc_cost']) for i in range(num_sd)])
        length  = length.view(-1, n, 1).repeat(k, 1, 1)
        toll    = torch.cat([tensor(graph['toll']) for i in range(num_sd)])
        toll    = toll.view(-1, n, 1).repeat(k, 1, 1)

        
        def parametric_c(w, t, length, toll): 
            k, n, _ = length.shape
            w1, w2, w3, w4, w5 = w
            w_sin = torch.cat((w3, w4, w5), axis=1).view(-1,3,1)
            w_sin = w_sin/w_sin.norm(p=1, dim=1, keepdim=True)

            u = torch.cat((torch.ones_like(t),t, length), 2)
            sin_fn = torch.sin(u@w_sin*2*pi)+1

            plp_c = toll@w1.unsqueeze(2) + (length*sin_fn)@w2.unsqueeze(2)   +length          
                
            return plp_c

        def parametric_b_ub(w, length):
            k, n, _ = length.shape
            w = torch.cat(w, axis = 1).view(-1,2,1)

            u = torch.cat((torch.ones_like(length), length), 2)

            plp_b_ub = u@w +1 

            return plp_b_ub

        plp_c = parametric_c([w1, w2, w3, w4, w5], t, length, toll)
        plp_b_ub=parametric_b_ub([w6, w7], length[:, :m, :])
        
        plp_A_ub = self.ref_A_ub.repeat(k,1,1) if self.ref_A_ub is not None else None
        plp_A_eq = self.ref_A_eq.repeat(k,1,1) if self.ref_A_eq is not None else None
        plp_b_eq = self.ref_b_eq.repeat(k,1,1) if self.ref_b_eq is not None else None
        
        if plot_param == True:
            fig, ax = plt.subplots(1, 2, figsize=(6,3))
            
            ax[0].plot(plp_c.squeeze(2).detach()) 
            ax[0].set_title('plp_c')
            ax[1].plot(plp_b_ub.squeeze(2).detach())
            ax[1].set_title('plp_b$_{ub}$')
            for i in range(len(ax)):
                ax[i].set_xticks([0, k//2, k])
                ax[i].set_xticklabels([0, 12, 24])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle('parametric function', fontsize=20)
            
        return plp_c, plp_A_ub, plp_b_ub, plp_A_eq, plp_b_eq
    
def sampling_w_true(seed, verbose = 0):

    """
    sample true parameters w = [w1, w2, w3, w4, w5, w6, w7], s.t., w3+w4+w5 = 1 and w5=1
    """
    if seed is not None:
        torch.manual_seed(seed)

    w = torch.zeros((7,)).uniform_(0,1).view(-1,1).type(torch.DoubleTensor)
    w1, w2, w3, w4, w5, w6, w7 = w
    w_sin = torch.cat((w3, w4, w5) )
    w_sin[-1]  = 1
    w_sin = w_sin/w_sin.norm(p=1, dim=0, keepdim=True)
    
    w = torch.cat((w1, w2, w_sin, w6, w7)).view(-1)
    if verbose >= 1:
        print('\tw_true = %s'%w.detach().numpy().ravel())

    return w.detach()

def sampling_w_random(seed, verbose = 0):
    """
    sample random parameters w = [w1, w2, w3, w4, w5, w6, w7], s.t., w3+w4+w5 = 1
    """
    if seed is not None:
        torch.manual_seed(seed)

    w = torch.zeros((7,)).uniform_(0,1).view(-1,1).type(torch.DoubleTensor)
    w1, w2, w3, w4, w5, w6, w7 = w
    w_sin = torch.cat((w3, w4, w5))
    w_sin = w_sin/w_sin.norm(p=1, dim=0, keepdim=True)
    
    w = torch.cat((w1, w2, w_sin, w6, w7)).view(-1)
    if verbose >= 1:
        print('\tw_random = %s'%w.detach().numpy().ravel())

    return w.detach()

def generate_plp_multi_commodity(graph, 
                                 nIns, 
                                 num_obs, 
                                 directory,
                                 plp_class,
                                 show_plot = False, 
                                 verbose   = 0):
    """
    Input:
         graph: basic graph contains the source-destination pairs and flow convervation
          nIns: number of instances to generate
       num_obs: number of observations (i.e., time points through out a day)

    Generate IO instances for MCF problem through the following steps:
    1). sample local feature lenght and toll price
    2). sample global feature time for training and testing
    3). sample true parameter w
    4). solve MCF(w, length, toll, time) and obtain x_target
    5). save each MCF instance in a pkl file

    """

    for ind_ins in range(nIns):
        if verbose >=1:
            print('==== Generate Parametric Multi-Commodity instance %d'%(ind_ins))
        graph = graph.copy()
    
        num_sd = len(graph['SD'])
        num_arc = len(graph['flow'])
        np.random.seed(0)
        length  = np.random.uniform(0,1, size=(num_arc,)).round(4) #sample local feature length
        toll    = np.random.uniform(0,1, size=(num_arc,)).round(4) #sample local feature toll price
        
        t_train = np.sort(np.random.uniform(0,1, size=(num_obs//2)).round(4)) #sample global feature time for training
        t_test = np.sort(np.random.uniform(0,1, size=(num_obs//2)).round(4)) #sample global feature time for testing
        t_collect = np.sort(np.hstack((t_train, t_test)))
        if verbose == True:
            print('    ++++ t_train', t_train)
            print('    ++++ t_test', t_test)
        
        # dummy capacity used for generating baselineLP = (c, A_ub, b_ub, A_eq, b_eq)
        # baselineLP are then used for generating parametricLP (plp_c, plp_A_ub, plp_b_ub, plp_A_eq, plp_b_eq)
        # NOTE, c and b_ub from baselineLP were never used in generating parametricLP
        graph['capacity'] = np.ones((num_arc))*4
        graph['arc_cost'] = length
        graph['toll'] = toll
    
        w_true = sampling_w_true(seed    = ind_ins,
                                 verbose = verbose)

        plp_true = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)),
                             w_true)
        assert plp_true.nneg == True
        lp_formulation_true = plp_true.generate(t        = tensor(t_collect),
                                                weights  = plp_true.weights,
                                                graph    = graph,
                                               )


        assert np.allclose(sum(w_true.detach().numpy().ravel()[plp_true.w_normalize_idx]), 1),\
                        'sum(w_true)[plp_true.w_normalize_idx]) = %s'\
                        %sum(w_true.detach().view(-1)[plp_true.w_normalize_idx])
        assert (lp_formulation_true[2] <=3).all() 
        assert (lp_formulation_true[2] >=1).all() 
        assert (lp_formulation_true[0] >=0).all()

        soln, status, *rest = homogeneous_solver(*lp_formulation_true,
                                                 nneg = plp_true.nneg,
                                                 tol  = 1e-8, 
                                                 std_form  = False)
        assert (soln   >=0).all()
        assert (status ==0).all()
        assert (status !=4).all()
        
        soln    = soln.detach().squeeze(2)
        
        graph['w_true']  = w_true.detach().numpy().ravel()
        graph['t_train'] = t_train
        graph['t_test']  = t_test
        
        save_file_directory = directory + 'parametricND/multi-commodity-flow'

        if not os.path.exists(save_file_directory):
            os.makedirs(save_file_directory)
        _file_record = save_file_directory + '/Nguyen-Dupuis_ins%d.pkl'%(ind_ins)
        with open(_file_record, 'wb') as output:
            pickle.dump(graph, output)


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
                                                      want_grad = True,
                                                      std_form  = False)

        ctx.mark_non_differentiable(s, niter, pdgap)  # mark non-differentiable to speed up the process
        ctx.lp_grad = grad  # save the gradient function for the backward process
        ctx.aoe_closed_form = aoe_closed_form
        x.requires_grad_()  # enable gradients of x only if one of the lp parameters also requires gradients
        
        return x, s, niter, pdgap
            
    @staticmethod
    def backward(ctx, grad_x, grad_s, grad_niter, grad_pdgap):
        # use the gradient function and grad_x to compute the gradient of each lp parameter using the implicit differentiation formula
        dc, dA_ub, db_ub, dA_eq, db_eq = ctx.lp_grad(grad_x.squeeze(-1),
                                                     homogeneous_mode = False,
                                                     aoe_closed_form = ctx.aoe_closed_form)
        if dA_eq.nelement() == 0:
            dA_eq = None 
            db_eq = None    
        return dc.unsqueeze(2), dA_ub, db_ub.unsqueeze(2), dA_eq, db_eq, None, None, None

def call_cvxpylayer(lp, layer, tol, verbose = True):

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
                            solver_args = {'n_jobs_forward': 1, 'n_jobs_backward': 1, 'eps': tol, 'max_iters': int(1e8)})
        else:
            with HiddenPrints():
                soln, = layer(c.squeeze(-1), A_ub, b_ub.squeeze(-1), 
                            solver_args = {'n_jobs_forward': 1, 'n_jobs_backward': 1, 'eps': tol, 'max_iters': int(1e8)})
        status = torch.zeros((N), dtype = DT)
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
        grad_mode:  numerical_grad  - forward: homogeneous_solver
                                     backward: infinite difference
                    None            - forward: homogeneous_solver
                                     backward: None
                    cvxpylayer      - forward: cvxpylayer
                                     backward: cvxpylayer
                    bprop           - forward: homogeneous_solver
                                     backward: backpropagation (PyTorch automatic differentiation)
                    implicit        - forward: homogeneous_solver
                                     backward: implicit differentiation (see backward in LinprogImplicitDiff)
                    direct          - forward: homogeneous_solver
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
        return homogeneous_solver(*lp, 
                                  tol = tol, 
                                  nneg = nneg, 
                                  want_grad= False)
    elif grad_mode =='cvxpylayer':
        assert layer is not None
        return call_cvxpylayer(lp, layer, tol, False)
    else:
        raise ValueError('Wrong grad_mode value')

def inverse_parametric_linprog(graph, 
                               u_train, 
                               x_target, 
                               plp,
                               inner_tol        =  1e-8,  # by default, use high-precision forward solver for the inner problem.
                               outer_method     =  None,
                               grad_mode        =  None,  
                               runtime_limit    =  30,
                               verbose          =  0,
                               collect_evals    =  True):
    """
    solve inverse MCF problem with COBYLA or SLSQP to learn objective and ineq. constraints

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
    assert grad_mode in['backprop', 'implicit', 'direct', 'cvxpylayer', None], GRAD_MODE_ERROR_MESSAGE
    assert outer_method in['SLSQP','COBYLA'], OUTER_METHOD_MESSAGE

    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training is loss<loss_tol
    outer_tol        =  1e-15       # tolerance for scipy.opt package
    outer_maxiter    =  10000       # iteration for scipy.opt package (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    outer_maxeval    =  10000       # max iteration for evaluation fop (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance
    boundedness_cons =  False       

    num_evals      = 0
    curr_w         = None  # Current parameter vector being queried by scipy minimize
    curr_w_rep     = None  # Repeated version of curr_w, to collect Jacobian entries
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
        if not is_w_curr(w) or curr_lp is None:
            curr_w = torch.tensor(w.ravel(), requires_grad=requires_grad)  # Deliberately copy w, enable gradient for SLSQP
            curr_w_rep = curr_w.view(1, -1).repeat(len(u_train), 1)        # copy w for |u_train| copies for faster backpropagation
            curr_lp = plp.generate(u_train, 
                                   curr_w_rep, 
                                   graph = graph,
                                  )
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
            curr_soln = solve_lp(lp        = get_curr_lp(w),
                                 grad_mode = grad_mode,
                                 tol       = inner_tol,
                                 nneg      = plp.nneg,
                                 layer     = layer)
            

        return curr_soln
    
    # get the status code from the homogeneous solver
    def get_curr_status(w):
        return get_curr_soln(w)[1]
        
    def get_curr_loss(w):  # w is ndarray
        nonlocal curr_loss
        if not is_w_curr(w) or curr_loss is None:
            c = get_curr_lp(w)[0]
            x, s, *_ = get_curr_soln(w)
            # assert (s != 4).all(), 'homogeneous solver status =%s'%status.numpy()
            
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


        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        # Record the best loss for feasible w, in case we later hit evaluation limit
        if l < best_loss and (get_curr_status(w) == 0 ).all():
            v = get_curr_target_feasibility_violation(w)
            sum_w = np.sum(w[plp.w_normalize_idx])
            
            # Record loss only if outer problem constraints are satisfied and fop is feasible
            if v < violation_tol and abs(sum_w - 1) < violation_tol and (w > -violation_tol).all():
                best_loss = float(l)
                best_w = w.copy()     
        loss_collect.append(best_loss)
        if verbose == 2:
            print('   [%d]  f(%s) \n \t  l=[%.8f]  infea=[%.4f]  sum_w = [%4f], pd=%s  s=%s'\
                  % (num_evals, 
                     w.ravel(), 
                     l,
                     get_curr_target_feasibility_violation(w),
                     np.sum(w[plp.w_normalize_idx]),
                     get_curr_pdgap_violation(w) if boundedness_cons else None,
                     get_curr_status_codes(w)))
            
        if best_loss < loss_tol:
            raise LossToleranceExceeded()
            
        return l

    # Function to evaluate gradient of loss at w
    def f_grad(w):
        nonlocal curr_w
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d] df(%s)'% (num_evals, w.ravel()))

        zero_w_grad()
        l = get_curr_loss(w)
        
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        l.backward(retain_graph=True)
        if verbose == 2:
            print('\tjac in f_grad(w) ',curr_w.grad.numpy())
            
        return curr_w.grad.numpy()

    # Function to evaluate ineq. constraint residuals of outer problem at w
    def g(w):
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()

        if verbose == 2:
            print('   [%d]  g(%s)'% (num_evals, w.ravel()))
            
        r = get_curr_target_feasibility_ub(w).detach().numpy() # x_target is feasible

        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()


        r = np.concatenate((r, w))
        
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
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
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
        
        jac = np.concatenate((jac, np.eye(len(w)) ))
            
        return jac
    # Function to evaluate eq. constraint residuals of outer problem at w
    def h(w):
        
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
            
        if verbose == 2:
            print('   [%d]  h(%s)'% (num_evals, w.ravel()))
            
        r = np.sum(w[plp.w_normalize_idx])-1
        
        
        if max(abs(w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
            
        return r

    # Function to evaluate gradient of eq. constraint residuals of outer problem at w
    def h_grad(w):
        
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
            
        if verbose == 2:
            print('   [%d] dh(%s)'% (num_evals, w.ravel()))

        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()

        jac = np.zeros((len(w),))
        jac[plp.w_normalize_idx] = 1
            
        return jac
    
    # Function to evaluate eq. constraint residuals of outer problem at w
    def h_cobyla(w):
        
        if time.time()-io_tic>runtime_limit: # check if reaching the runtime_limit
            raise RuntimeLimitExceeded()
            
        if verbose == 2:
            print('   [%d]  h_cobyla(%s)'% (num_evals, w.ravel()))
            
        r = np.array([np.sum(w[plp.w_normalize_idx])-1, 1-np.sum(w[plp.w_normalize_idx])])
                
        if max(abs(curr_w)) >1e5:               # check if weights values is getting too large/small
            raise UnboundedWeightsValue()
            
        return r

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
        
        constraints = [{'type': 'ineq', 'fun': g, 'jac': g_jac}]
        constraints.append({'type': 'ineq', 'fun': h_cobyla, 'jac': h_jac})
        options = {'maxiter': outer_maxiter}
        
    elif outer_method =='SLSQP':
        f_jac = f_grad
        g_jac = g_grad    
        h_jac = h_grad
        
        constraints = [{'type': 'ineq', 'fun': g, 'jac': g_jac}]
        constraints.append({'type': 'eq', 'fun': h, 'jac': h_jac})
        
        options = {'maxiter': outer_maxiter}

        if grad_mode == 'cvxpylayer':
            f_jac = f_grad
            g_jac = g_grad    
            h_jac = h_grad
            
            layer = create_cvx_layer(plp.generate(u_train, plp.weights, graph = graph), plp.nneg)
        
    io_tic = time.time()
    
    try:
        res = opt.minimize(f, plp.weights.detach(), jac = f_jac, 
                           constraints = constraints, 
                           tol = outer_tol,
                           method = outer_method,
                           options = options)
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
    io_toc = time.time()

    if best_loss < 999:
        plp.weights[:] = torch.from_numpy(best_w)
        
    assert best_loss == loss_collect[-1], 'best_loss = %.2e, loss_collect[-1] = %.2e'%(best_loss == loss_collect[-1])

    if verbose >= 1:
        print('++++ Result Summary from outer_method [%s] grad_mode [%s]'%(outer_method, grad_mode))
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

def random_search(graph, 
                  u_train, 
                  x_target,
                  plp,
                  w_init,
                  inner_tol       =  1e-8,  # by default, use high-precision forward solver.
                  runtime_limit   =  60,
                  verbose         =  0,
                  collect_evals   =  False):

    """
    solve inverse MCF with random search to learn objective, ineq. constraints
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

    loss_fn          =  _AOE_       # loss function for training 
    loss_tol         =  1e-5        # termiante training is loss<loss_tol
    outer_maxeval    =  10000       # max iteration for evaluation fop (use a huge number, so the algo will not terminate till reaching the runtime_limit)
    violation_tol    =  1e-3        # constraint violation tolerance

    num_evals     =  0
    best_w        =  np.ones((6,))*999
    best_loss     =  float(999)
    loss_collect  =  []
        
    def get_curr_loss(w):  # w is ndarray
        curr_w = w.clone() # Deliberately copy w
        
        curr_lp = plp.generate(u_train, 
                               curr_w.detach(),
                               graph = graph
                              )
        curr_soln = homogeneous_solver(*curr_lp, 
                                       tol = inner_tol, 
                                       nneg = plp.nneg,
                                       std_form = False)
        
        c = curr_lp[0]
        x, s, _, _, = curr_soln
        l = loss_fn(c, x_target, x)
        curr_loss = l.mean().detach()
        
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
            
            sum_w = sum(w[plp.w_normalize_idx]).detach().numpy()
            # Record loss only if outer problem constraints are satisfied and fop is feasible
            if v < violation_tol and abs(sum_w - 1) < violation_tol and (w>0).all():
                best_loss = float(l)
                best_w = w.detach().numpy().copy()   
                
        loss_collect.append(best_loss)
        if verbose == 2:
            print('   [%d]  f(%s) \n \t  l=[%.8f]  infea=[%.4f]   sum_w = [%.4f]  s=%s'
                  % (num_evals, w.detach().numpy().ravel(), float(l),
                     get_curr_target_feasibility_violation(curr_lp),
                     sum(w[plp.w_normalize_idx]).detach().numpy(),
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
            w = sampling_w_random(seed    = None,
                                  verbose = verbose-1)
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

def experiment(nIns, 
               inner_tol,
               plp_class,
               outer_method,
               grad_mode,
               runtime_limit, 
               verbose,
               directory):

    """
    run experiment for a specific MCF instance

    Input:
          nVar, nCons: num. variabel, num. constriants,
              ind_ins:  instance index
            inner_tol: tolerance of forward solver
         outer_method: algorithms for solving the bi-level ILOP
            grad_mode: backward methods for computing gradients
            plp_class: parametric LP class (vectorized LP generation with different w values)
        runtime_limit: runtime limit for solving ILOP (in seconds)
    """

    RESULT = []
    BEST_LOSS = []
    for ind_ins in range(nIns):

        torch.manual_seed(ind_ins+100)
        if verbose >=1:
            print('************************************')
            print('ILOP Exp2 Parametirc multi-commodity Min Cost on ND Graph ins %s'%ind_ins)
            if outer_method =='RS':
                print('\twith Random Search [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
            else:
                if outer_method == 'COBYLA':
                    print('\twith COBYLA [No Gradient], homogeneous(tol = %.2e)'%inner_tol)
                elif outer_method == 'SLSQP':
                    print('\twith SLSQP [%s] + %s(tol = %.2e)'
                            %(grad_mode, 'cvx' if grad_mode == 'cvxpylayer' else 'homogeneous', inner_tol))

            print('\tloss_fn = [AOE]' )
        
        file = directory+'parametricND/multi-commodity-flow/Nguyen-Dupuis_ins%d.pkl'%(ind_ins)
        graph = read_pkl_file(file)

        w_true = graph['w_true']
        if verbose > 0: 
            print('w_true = %s'%w_true)

        t_train = tensor(graph['t_train'])

        plp_true = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), w_true)
        assert plp_true.nneg == True
        plp_formulation_true = plp_true.generate(t          = t_train,
                                                 weights    = plp_true.weights,
                                                 graph      = graph,
                                                 plot_param = verbose-1)

        x_target, s_true, *_ = homogeneous_solver(*plp_formulation_true, 
                                                    nneg      = plp_true.nneg,
                                                    tol       = 1e-8, 
                                                      std_form  = False)

        assert (s_true == 0).all(), 'homogeneous solver status =%s'%status.numpy().ravel()
        assert (s_true != 4).all(), 'homogeneous solver status =%s'%status.numpy().ravel()

        plp_init = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), w_true)

        for trail_i in range(50):
            w_init = sampling_w_random(seed    = None,
                                       verbose = 0)


            if verbose >1:
                print('trail [%d], w_random = %s'%(trail_i+1, w_init.numpy().ravel()))
            plp_init = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), w_init)
            assert plp_init.nneg == True

            plp_formulation_init = plp_init.generate(t       = t_train,
                                                     weights = plp_init.weights,
                                                     graph = graph
                                                    )

            _, s_init, *_ = homogeneous_solver( *plp_formulation_init,
                                               nneg = plp_init.nneg,
                                               tol  = 1e-8, 
                                               std_form  = False)
            assert (s_init != 4).all(), 'homogeneous solver status =%s'%status.numpy()
            if (s_init == 0).all():
                if verbose > 1:
                    print('\t After %d Trail, found a feasbile w_init = %s'
                          %(trail_i+1, w_init.detach().numpy().ravel()) )

                break
        if outer_method == 'RS':
            best_l, result = random_search(graph           =  graph,
                                           u_train         =  t_train, 
                                           x_target        =  x_target,
                                           plp             =  plp_init,
                                           w_init          =  w_init,
                                           inner_tol       =  inner_tol,
                                           runtime_limit   =  runtime_limit,
                                           verbose         =  verbose,
                                           collect_evals   =  True)
        else:
            best_l, result = inverse_parametric_linprog(graph            = graph,
                                                        u_train          =  t_train, 
                                                        x_target         =  x_target, 
                                                        plp              =  plp_init,
                                                        outer_method     =  outer_method,
                                                        grad_mode        =  grad_mode,    # backprop, implicit, implicit_hlf
                                                        inner_tol        =  inner_tol,
                                                        runtime_limit    =  runtime_limit,
                                                        verbose          =  verbose,
                                                        collect_evals    =  True)

        assert best_l == result['res']['best_l']

        if verbose >=1:
            print('*********************************\n')

        plp_lrn = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), result['res']['best_w'])

        assert plp_lrn.nneg == True
        plp_formulation_lrn = plp_lrn.generate(t       = t_train,
                                               weights = plp_lrn.weights,
                                               graph  = graph,
                                              )
        
        soln_lrn, status, *_ = homogeneous_solver(*plp_formulation_lrn,
                                                  nneg      = plp_lrn.nneg,
                                                  tol       = 1e-8,
                                                  std_form  = False)

        assert (status == 0).all(), 'homogeneous solver status =%s'%status.numpy()
        assert (status != 4).all(), 'homogeneous solver status =%s'%status.numpy()
        
        soln = soln_lrn.detach().squeeze(2)
        
        if outer_method == 'RS':
            save_file_directory = directory + 'parametricND/multi-commodity-flow/RS'
        elif outer_method == 'SLSQP':
                save_file_directory = directory + 'parametricND/multi-commodity-flow/SLSQP_%s'%grad_mode
        elif outer_method =='COBYLA':
            save_file_directory = directory + 'parametricND/multi-commodity-flow/COBYLA'
        if not os.path.exists(save_file_directory):
            os.makedirs(save_file_directory)


        
        _file_record = save_file_directory + '/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        with open(_file_record, 'wb') as output:
            pickle.dump(result, output)


        RESULT.append(result)
        BEST_LOSS.append(best_l)
        
        
    return (RESULT, BEST_LOSS)

def parametric_MCF_exp(nIns,
                       plp_class,
                       loss_fn,
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
    for framework_i in framework:
        outer_method, grad_mode = framework_i
        
        
        _ = experiment(nIns          = nIns,
                       inner_tol     = 1e-8,
                       plp_class     = plp_class,
                       outer_method  = outer_method,
                       grad_mode     = grad_mode,
                       runtime_limit = runtime_limit,
                       verbose       = verbose,
                       directory     = directory)
    
    
def parametricND_data_analysis(nIns,
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


    for framework_i in framework:
        
        outer_method, grad_mode = framework_i
        if outer_method in ('RS','COBYLA'):
            _file_path = directory + 'parametricND/multi-commodity-flow/%s'\
                                        %(outer_method)
        else:
            _file_path = directory + 'parametricND/multi-commodity-flow/%s_%s'\
                                        %(outer_method, grad_mode)
        for ind_ins in range(nIns):
            if verbose == True:
                print('\n=======  Parametric ND, Ins %d, loss = %s '
                                %(ind_ins, loss_fn.__name__[1:-1]))
                print('         Outer_method %s, grad_mode %s'
                                %(outer_method, grad_mode))

            result_file = _file_path+'/Nguyen-Dupuis_ins%d'%(ind_ins)

            data_ = read_pkl_file(result_file+'_res.pkl')
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

    return  (l_RS,                w_RS,                timestep_RS,                runtime_RS),\
            (l_COBYLA,            w_COBYLA,            timestep_COBYLA,            runtime_COBYLA), \
            (l_SLSQP_backprop,    w_SLSQP_backprop,    timestep_SLSQP_backprop,    runtime_SLSQP_backprop),\
            (l_SLSQP_implicit,  w_SLSQP_implicit,  timestep_SLSQP_implicit,  runtime_SLSQP_implicit),\
            (l_SLSQP_direct, w_SLSQP_direct, timestep_SLSQP_direct, runtime_SLSQP_direct),\
            (l_SLSQP_cvxpylayer,  w_SLSQP_cvxpylayer,  timestep_SLSQP_cvxpylayer,  runtime_SLSQP_cvxpylayer)

def boxplot(ax, nIns, plp_class, directory, verbose = 0):

    """
    plot boxplot of loss.  

    Input:
            nIns: num. of instances
       plp_class: parametric LP class (i.e., ParametricND) with vecotrized code for generating LPs

    """

    def collect_training_loss(rs_result,
                              cobyla_result,
                              slsqp_backprop_result, 
                              slsqp_implicit_result,
                              slsqp_direct_result,
                              slsqp_cvxpylayer_result):

        """
        collect training loss

        Input: 
                        rs_result: Random search results, a dict
                    cobyla_result: COBYLA results, a dict
            slsqp_backprop_result: SQP_bprop results, a dict
            slsqp_implicit_result: SQP_impl results, a dict
              slsqp_direct_result: SQP_dir results, a dict
          slsqp_cvxpylayer_result: SQP_cvx results, a dict

        Output:
             initial_loss: Initial loss instance
            training_loss: best training loss of each methods
        """
        # sanity check on initial loss
        assert abs( abs(slsqp_backprop_result['f']['loss'][0]) - abs(cobyla_result['f']['loss'][0])) < 1e-3,\
                'SLSQP(backprop) init loss [%.2e], COBYLA init loss [%.2e]'%(slsqp_backprop_result['f']['loss'][0], cobyla_result['f']['loss'][0])
        
        assert abs( abs(slsqp_backprop_result['f']['loss'][0]) - abs(rs_result['f']['loss'][0])) < 1e-3,\
                'SLSQP(backprop) init loss [%.2e], RS init loss [%.2e]'%(slsqp_backprop_result['f']['loss'][0], rs_result['f']['loss'][0])
        
        assert abs( abs(slsqp_backprop_result['f']['loss'][0]) - abs(slsqp_implicit_result['f']['loss'][0])) < 1e-3,\
                'SLSQP(backprop) init loss [%.2e], implicit init loss [%.2e]'%(slsqp_backprop_result['f']['loss'][0], slsqp_implicit_result['f']['loss'][0])
        
        assert abs( abs(slsqp_backprop_result['f']['loss'][0]) - abs(slsqp_direct_result['f']['loss'][0])) < 1e-3,\
                'SLSQP(backprop) init loss [%.2e], implicit_AOE init loss [%.2e]'%(slsqp_backprop_result['f']['loss'][0], slsqp_direct_result['f']['loss'][0])
               
        initial_loss  = float(slsqp_backprop_result['f']['loss'][0])
        training_loss = ([float(rs_result['res']['best_l']),
                          float(cobyla_result['res']['best_l']), 
                          float(slsqp_cvxpylayer_result['res']['best_l']),
                          float(slsqp_backprop_result['res']['best_l']),
                          float(slsqp_implicit_result['res']['best_l']),    
                          float(slsqp_direct_result['res']['best_l']) ]) 

        return initial_loss, training_loss
    
    def collect_testing_loss(plp_class, 
                             rs_result, 
                             cobyla_result,
                             slsqp_backprop_result, 
                             slsqp_implicit_result, 
                             slsqp_direct_result, 
                             slsqp_cvxpylayer_result, 
                             directory):
        """
        compute and collect testing loss

        Input: 
                        plp_class: parametric LP class (vectorized LP generation with different w values)
                        rs_result: Random search results, a dict
                    cobyla_result: COBYLA results, a dict
            slsqp_backprop_result: SQP_bprop results, a dict
            slsqp_implicit_result: SQP_impl results, a dict
              slsqp_direct_result: SQP_dir results, a dict
          slsqp_cvxpylayer_result: SQP_cvx results, a dict

        Output:
             initial_loss: Initial loss instance
            training_loss: best training loss of each methods
        """
        if verbose >0:
            print('Computing Testing Loss = c_true(x_true - x_pred)')
        rs_best_w = rs_result['res']['best_w']
        cobyla_best_w = cobyla_result['res']['best_w']
        slsqp_backprop_best_w = slsqp_backprop_result['res']['best_w']
        slsqp_implicit_best_w = slsqp_implicit_result['res']['best_w']
        slsqp_direct_best_w = slsqp_direct_result['res']['best_w']
        slsqp_cvxpylayer_best_w = slsqp_cvxpylayer_result['res']['best_w']

        file = directory+'parametricND/multi-commodity-flow/Nguyen-Dupuis_ins%d.pkl'%(ind_ins)
        graph = read_pkl_file(file)
        w_true = graph['w_true']
        if verbose >0:
            print('\tw_true = %s'%w_true)
            print('\tCOBYLA best_w = %s'%np.array(cobyla_best_w))
            print('\tSLSQP (backprop) best_w = %s'%np.array(slsqp_backprop_best_w))
            print('\tSLSQP (implicit) best_w = %s'%np.array(slsqp_implicit_best_w))
            print('\tSLSQP (direct) best_w = %s'%np.array(slsqp_direct_best_w))
            print('\tSLSQP (cvxpylayer) best_w = %s'%np.array(slsqp_cvxpylayer_best_w))
        t_test = tensor(graph['t_test'])

        plp_ND_true = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), w_true)
        assert plp_ND_true.nneg == True
        plp_formulation_true = plp_ND_true.generate(t       = t_test,
                                                    weights = plp_ND_true.weights,
                                                    graph   = graph,
                                                   )
        soln_true, status, *_ = homogeneous_solver(*plp_formulation_true,
                                                    nneg      = plp_ND_true.nneg,
                                                    tol       = 1e-8, 
                                                    std_form  = False)
        assert (status==0).all()
        
        testing_loss_mean = []
        testing_loss_median = []
        for best_w_ in [rs_best_w, 
                        cobyla_best_w,
                        slsqp_cvxpylayer_best_w,
                        slsqp_backprop_best_w, 
                        slsqp_implicit_best_w,
                        slsqp_direct_best_w]:

            plp_ND_lrn = plp_class(as_tensor(*multi_commodity_min_cost_lp_model(graph)), best_w_)
            assert plp_ND_lrn.nneg == True
            plp_formulation_lrn = plp_ND_lrn.generate(t       = t_test,
                                                      weights = plp_ND_lrn.weights,
                                                      graph   = graph,
                                                     )
            soln_lrn, status, *_ = homogeneous_solver(*plp_formulation_lrn, 
                                                          nneg      = plp_ND_lrn.nneg,
                                                          tol       = 1e-8, 
                                                          std_form  = False)      
            assert (status==0).all()
            assert (status!=4).all()
            
            c_true = plp_formulation_true[0]
            l_mean = _AOE_(c_true.detach(), soln_true.detach(), soln_lrn.detach()).mean().numpy()
            l_median = _AOE_(c_true.detach(), soln_true.detach(), soln_lrn.detach()).median().numpy()

            testing_loss_mean.append(float(l_mean))
            testing_loss_median.append(float(l_median))
        return testing_loss_mean, testing_loss_median

    def plot_(loss_plot, color, axes):
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
        axes.set_ylim(5e-6,1e1)
        axes.get_xaxis().set_visible(False)
        
    initial_loss  = []
    training_loss = []
    l_test_mean  = []
    l_test_median  = []
    
    for ind_ins in range(nIns):

        file = directory+'parametricND/multi-commodity-flow/RS/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        rs_result = read_pkl_file(file)  
        
        file = directory+'parametricND/multi-commodity-flow/COBYLA/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        cobyla_result = read_pkl_file(file)  

        file = directory+'parametricND/multi-commodity-flow/SLSQP_backprop/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        slsqp_backprop_result = read_pkl_file(file)

        file = directory+'parametricND/multi-commodity-flow/SLSQP_implicit/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        slsqp_implicit_result = read_pkl_file(file) 
        
        file = directory+'parametricND/multi-commodity-flow/SLSQP_direct/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        slsqp_direct_result = read_pkl_file(file)   

        file = directory+'parametricND/multi-commodity-flow/SLSQP_cvxpylayer/Nguyen-Dupuis_ins%d_res.pkl'%(ind_ins)
        slsqp_cvxpylayer_result = read_pkl_file(file)   

        l_init, l_train = collect_training_loss(rs_result               = rs_result,
                                                cobyla_result           = cobyla_result,
                                                slsqp_backprop_result   = slsqp_backprop_result, 
                                                slsqp_implicit_result   = slsqp_implicit_result,
                                                slsqp_direct_result     = slsqp_direct_result,
                                                slsqp_cvxpylayer_result = slsqp_cvxpylayer_result)

        initial_loss.append(l_init)
        training_loss.append(l_train)
        
        l_mean, l_median = collect_testing_loss(plp_class     = plp_class, 
                                                rs_result     = rs_result,
                                                cobyla_result = cobyla_result, 
                                                slsqp_backprop_result   = slsqp_backprop_result, 
                                                slsqp_implicit_result   = slsqp_implicit_result,
                                                slsqp_direct_result     = slsqp_direct_result, 
                                                slsqp_cvxpylayer_result = slsqp_cvxpylayer_result, 
                                                directory               = directory)
        l_test_mean.append(l_mean)
        l_test_median.append(l_median)
        
    initial_loss  = np.array(initial_loss)
    initial_loss = np.clip(initial_loss,1e-5, 1e3)
    
    training_loss = np.array(training_loss)
    training_loss = np.clip(training_loss,1e-5, 1e3)

    if verbose >= 1:
        print('num l_train <= 1e-5: ', [sum(training_loss[i,:]<=1e-5) for i in range(len(training_loss))])
    
    l_test_mean  = np.array(l_test_mean)
    l_test_mean = np.clip(l_test_mean,1e-5, 1e3)
    
    l_test_median  = np.array(l_test_median)
    l_test_median = np.clip(l_test_median,1e-5, 1e3)
    
    assert ax is not None
    
    ax1, ax2, ax3 = ax
    



    plot_([training_loss[:,i] for i in range(len(training_loss[0]))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
          color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
          axes = ax1)

    plot_([l_test_mean[:,i] for i in range(len(l_test_mean[0]))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
          color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
          axes = ax2)

    plot_([l_test_median[:,i] for i in range(len(l_test_median[0]))],
                  # RS, cobyla, sqp_cvx, sqp_bprop, sqp_implic, sqp_close
          color = ['m', 'b',      'k',    'r',      'orange',    'g', ],
          axes = ax3)
    
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax1.set_xlim(0,7)
    ax2.set_xlim(0,7)
    ax3.set_xlim(0,7)
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    
    
def fig_exp2(nIns, 
            loss_fn, 
            plp_class,
            framework,
            directory,
            verbose = 0):
    """
    Generate figures for exp1a

    Input:
    plp_class: parametric LP class (vectorized LP generation with different w values)
         nIns: number of instance for each lp_size
      loss_fn: function for evaluating the loss
    framework: a list of tuple (outer_method, grad_mode)

    """

    import matplotlib.gridspec as gridspec


    time_range = np.arange(1e-3, 30, 1e-3).round(3)   #   time steps for 10D
    
    res_RS, res_COBYLA, res_SLSQP_backprop,\
    res_SLSQP_implicit, res_SLSQP_direct, \
    res_SLSQP_cvxpylayer = parametricND_data_analysis(nIns       = nIns,
                                                      loss_fn = loss_fn,
                                                      framework = framework,
                                                      directory  =  directory,
                                                      verbose    =  False)

    """
    create plots for each lp_size
    """
    success_rate = []

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,3),
                                             gridspec_kw={'width_ratios': [1, 1, 1, 1]})
                                
    
    res_= plot_success_rate(time_range = time_range,
                      res        = res_RS, 
                      ax         = ax1,
                      plot_style = ['m', '-', 'RS'])
    success_rate.append(res_[-1])

    res_= plot_success_rate(time_range = time_range,
                      res        = res_COBYLA, 
                      ax         = ax1,
                      plot_style = ['b', '-', 'COBYLA'])
    success_rate.append(res_[-1])

    res_= plot_success_rate(time_range = time_range,
                      res        = res_SLSQP_cvxpylayer, 
                      ax         = ax1,
                      plot_style = ['k', '-', 'SLSQP+cvxpylayer'])
    success_rate.append(res_[-1])

    res_= plot_success_rate(time_range = time_range,
                      res        = res_SLSQP_backprop, 
                      ax         = ax1,
                      plot_style = ['r', '-', 'SLSQP+backprop'])
    success_rate.append(res_[-1])

    res_= plot_success_rate(time_range = time_range,
                      res        = res_SLSQP_implicit, 
                      ax         = ax1,
                      plot_style = ['orange', '-', 'SLSQP+implicit'])
    success_rate.append(res_[-1])

    res_= plot_success_rate(time_range = time_range,
                      res        = res_SLSQP_direct, 
                      ax         = ax1,
                      plot_style = ['g', '-', 'SLSQP+closed'])
    success_rate.append(res_[-1])

    boxplot(ax        = (ax2, ax3, ax4), 
            nIns      = nIns, 
            plp_class = plp_class, 
            directory = directory,
            verbose = 0)

    ax1.tick_params( axis='x', which='major', labelsize=12 )
    ax1.set_xlim(-0.1, time_range[-1]+0.1)
    ax1.set_ylim(-5, 105)
    ax1.set_yticks(np.linspace(0,100,5))
    ax1.set_yticklabels([str(i)+'%' for i in np.linspace(0,100,5, dtype = np.int)])
    ax1.xaxis.set_label_position('bottom') 
    
    fig.subplots_adjust(wspace=0.02, hspace=0)
    
    fig.savefig('fig_exp2.pdf',dpi=150)
    
def main(directory = None):

    direct = './' if directory is None else directory

    NUM_INS             =  100   
    LOSS_FN             =  _AOE_         # loss function for IO evaluation

    VERBOSE         =  0    # 2 - print detailed information, e.g., log of each iter in the solving process, 
                            #     initialization, etc. This includes many printing functions during 
                            #     the solving process, thus, using VERBOSE = 2 will influence the experiment results. 
                            # 1 - print basic information, e.g., final results, initialization. 
                            #     all printing function  happen outside of the solving process, 
                            #     thus, does not influence the experiment results 
                            # 0 - don't display any algorithm log         
    RUN_TIME            =  30            # runtime limit for 2D and 10D

    FRAMEWORK       =  (# outer_method, grad_mode
                        ['RS',          None], 
                        ['COBYLA',      None], 
                        ['SLSQP',      'cvxpylayer'],
                        ['SLSQP',      'backprop'], 
                        ['SLSQP',      'implicit'],
                        ['SLSQP',      'direct'],
                       )

    dir_ = direct+'parametricND'    
    if not os.path.exists(dir_):
        generate_plp_multi_commodity(graph     = graph_ND,
                                     nIns      = NUM_INS, 
                                     num_obs   = 40, 
                                     directory = './',
                                     plp_class = ParametricND,
                                     show_plot = True, 
                                     verbose   = VERBOSE)

    parametric_MCF_exp(nIns          = NUM_INS,
                       plp_class     = ParametricND,
                       loss_fn       = LOSS_FN,
                       runtime_limit = RUN_TIME,
                       framework     = FRAMEWORK,
                       verbose       = VERBOSE,
                       directory     = direct)

    fig_exp2(nIns     = NUM_INS,
            loss_fn   = LOSS_FN,
            framework = FRAMEWORK,
            plp_class = ParametricND,
            directory = direct)

if __name__ == "__main__":
        

    main()