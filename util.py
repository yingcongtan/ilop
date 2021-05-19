# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


# Utility Functions

import torch
import numpy as np
from warnings import warn
from torch import bmm, baddbmm
from copy import deepcopy
import time
import sys, os
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

def tensor(data, **kwargs):
    """Returns a torch.DoubleTensor, unlike torch.tensor which returns FloatTensor by default"""
    return torch.DoubleTensor(data, **kwargs)

def detach(*args):
    r = tuple(arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in args)
    return r if len(r) > 1 else r[0]

def as_tensor(*args, **kwargs):
    r = tuple(torch.DoubleTensor(arg, **kwargs) if arg is not None else None for arg in args)
    return r if len(r) > 1 else r[0]

def as_numpy(*args):
    r = tuple(arg.detach().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args)
    return r if len(r) > 1 else r[0]

def as_str(*args, precision=5, indent=0, flat=False):
    r = tuple(np.array2string(as_numpy(x) if not flat else as_numpy(x).ravel(),
                              precision=4, max_line_width=100,
                              threshold=5, floatmode='fixed',
                              suppress_small=True, prefix=' '*indent) if x is not None else 'None'
              for x in args)
    return r if len(r) > 1 else r[0]

def build_tensor(x):
    """Builds a tensor from a list of lists x that preserves gradient flow
    through to whatever tensors appear inside x, which does not happen if
    you just call torch.tensor(). Used by ParametricLP base class.

    Useful for writing code like:
    
       a = tensor([1.5], requires_grad=True)
       b = tensor([3.0], requires_grad=True)
    
       x = build_tensor([[a*b, 6.0],
                         [0.0, a+1]])
    
       y = x.sum()
       y.backward()  # gradient of y flows back to a and b
    
       print(a.grad, b.grad)
    
    Builds a tensor out of a list-of-lists representation, but does so using
    torch.cat and torch.stack so that, if any components of the resuling matrix are
    tensors requiring gradients, the final tensor will be dependent on them and als
    require gradients.
    
    If you just use torch.tensor(...) using a list-of-lists, it will simply copy any
    tensors and the resulting tensor will be detached with no gradient required.

    If x is already a tensor, no change. If x is None, returns None.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(tuple(torch.cat(tuple(xij.view(-1) if isinstance(xij, torch.Tensor) else tensor((xij,)) for xij in xi)) for xi in x))

class ParametricLP:
    '''
    ParametricLP is a base class for defining a parametric
    linear program. The subclass is responsible for implementing
    the generate(u, w) function as depicted in the 1D example below.

        class Simple1D(io.ParametricLP):

            """min c(u) s.t. a(u) x <= b(u)"""
        
            def generate(self, u, w):
                w0, w1, w2, w3, w4, w5 = w

                c = [[w0 + w1*u]]
                A = [[w2 + w3*u]]
                b = [[w4 + w5*u]]
                
                return c, A, b, None, None   # None for A_eq or b_eq
            
        f = Simple1D([ 1.0, 0.0,   # c(u) = [ 1.0 + 0.0*u]
                      -1.0, 0.0,   # A(u) = [-1.0 + 0.0*u]
                       1.0, 1.0])  # b(u) = [ 1.0 + 1.0*u]

        params = f(u=2)          # params is [1.0, -1.0, -3.0, None, None]
        x = io.linprog(*params)  # x is -3.0

    '''
    def __init__(self, weights, requires_grad=True):
        self.weights = as_tensor(weights)
        if requires_grad:
            self.weights.requires_grad_()
                
    def zero_grads(self):
        if self.weights.grad is not None:
            self.weights.grad.detach_()
            self.weights.grad.data.zero_()
            
    def generate(self, u, w):
        raise NotImplementedError("Should be implemented by subclass")
    
    def __call__(self, u):
        # If called with a single scalar value, must be in a list to become a tensor
        if isinstance(u, (int, float)):
            u = [u]
            
        u = as_tensor(u)
            
        # Call build_tensor on each 
        return list(map(build_tensor, self.generate(u, self.weights)))
    
    def __str__(self):
        return "weights=%s" % as_str(self.weights, flat=True)

def norm(a):
    """Returns a (k,1,1)-tensor of k Frobenius from a (k,n,1)-tensor of n-dimensional vectors."""
    return torch.norm(a, None, dim=1, keepdim=True)
    
def T(x):
    """Returns a (k,n,m)-tensor of k transposed matricies where x is a (k,m,n)-tensor."""
    return torch.transpose(x, -2, -1)

def sym_cond(A):
    """Returns a (k,)-vector of the condition numbers of (k,n,n)-tensor A."""
    # Using symeig is more accurate than inverting A and is faster and uses less
    # less memory than svd(A). (Note: even with svd(A, compute_uv=False), PyTorch 1.5
    # apply_svd wastefully allocates U,V tensors of zeros, with full strides,
    # even though MKL doesn't touch the data -- argh!)
    S = A.symeig().eigenvalues
    return (S[:,-1] / S[:,0])

class _Baddbmm(torch.autograd.Function):
    """
    Same as baddbmm(c, a, b, beta=1, alpha=1) except forward+backward together
    is 1.2x faster than it is for built-in baddbmm. Does this by
    avoiding multiply of grad_a and grad_b by alpha. PyTorch 1.4.0 only does
    this optimization for x and for grad_c with current derivatives.yaml.
    """

    @staticmethod
    def forward(ctx, c, A, B):
        x = c.baddbmm(A, B)
        ctx.save_for_backward(A, B)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A, B = ctx.saved_tensors
        need_c, need_A, need_B = ctx.needs_input_grad
        grad_c = grad_x           if need_c else None
        grad_A = grad_x.bmm(T(B)) if need_A else None
        grad_B = T(A).bmm(grad_x) if need_B else None
        return grad_c, grad_A, grad_B
    
_baddbmm = _Baddbmm.apply

class _Bsubbmm(torch.autograd.Function):
    """
    Same as baddbmm(c, a, b, beta=-1, alpha=1). See _Baddbmm.
    """

    @staticmethod
    def forward(ctx, c, A, B):
        x = c.baddbmm(A, B, beta=-1)
        ctx.save_for_backward(A, B)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A, B = ctx.saved_tensors
        need_c, need_A, need_B = ctx.needs_input_grad
        grad_c = -grad_x           if need_c else None
        grad_A = grad_x.bmm(T(B))  if need_A else None
        grad_B = T(A).bmm(grad_x)  if need_B else None
        return grad_c, grad_A, grad_B
    
_bsubbmm = _Bsubbmm.apply

class _Addcdiv(torch.autograd.Function):
    """
    Same as addcdiv(c, a, b, value=1) except forward+backward together
    is 1.5x faster than for addcdiv directly. Does this by avoiding multiply
    with with 'value' and by using inplace ops.
    Note that, without this, addcdiv is actually slower for forward+backward
    than simply using regular ops (c + a / b), so there's no point
    in using addcdiv for backprop without this.
    """

    @staticmethod
    def forward(ctx, c, a, b):
        x = c.addcdiv(a, b)
        ctx.save_for_backward(a, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        a, b = ctx.saved_tensors
        need_c, need_a, need_b = ctx.needs_input_grad
        grad_c = grad_x                               if need_c else None
        grad_a = grad_x.div(b)                        if need_a else None
        grad_b = grad_x.mul(a).div_(b).div_(b).neg_() if need_b else None
        return grad_c, grad_a, grad_b

_addcdiv = _Addcdiv.apply

class _Subcdiv(torch.autograd.Function):
    """
    Same as addcdiv(c, a, b, value=-1) except forward+backward together
    is 1.4x faster. Cannot yet be scripted, though.
    """

    @staticmethod
    def forward(ctx, c, a, b):
        x = c.addcdiv(a, b, value=-1)
        ctx.save_for_backward(a, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        a, b = ctx.saved_tensors
        need_c, need_a, need_b = ctx.needs_input_grad
        grad_c = grad_x                        if need_c else None
        grad_a = grad_x.div(b).neg_()          if need_a else None
        grad_b = grad_x.mul(a).div_(b).div_(b) if need_b else None
        return grad_c, grad_a, grad_b

_subcdiv = _Subcdiv.apply

class _Addcmul(torch.autograd.Function):
    """
    Same as addcmul(c, a, b, value=1) except forward+backward together
    is 1.2x faster than for addcmul directly. Does this by avoiding
    multiply with 'value'.
    """

    @staticmethod
    def forward(ctx, c, a, b):
        x = c.addcmul(a, b)
        ctx.save_for_backward(a, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        a, b = ctx.saved_tensors
        need_c, need_a, need_b = ctx.needs_input_grad
        grad_c = grad_x        if need_c else None
        grad_a = grad_x.mul(b) if need_a else None
        grad_b = grad_x.mul(a) if need_b else None
        return grad_c, grad_a, grad_b

_addcmul = _Addcmul.apply

class _Subcmul(torch.autograd.Function):
    """
    Same as addcmul(c, a, b, value=-1) except forward+backward together
    is 1.2x faster. Cannot yet be scripted, though.
    """

    @staticmethod
    def forward(ctx, c, a, b):
        x = c.addcmul(a, b, value=-1)
        ctx.save_for_backward(a, b)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        a, b = ctx.saved_tensors
        need_c, need_a, need_b = ctx.needs_input_grad
        if need_a or need_b:
            neg_grad_x = -grad_x
        grad_c = grad_x             if need_c else None
        grad_a = neg_grad_x.mul(b)  if need_a else None
        grad_b = neg_grad_x.mul_(a) if need_b else None
        return grad_c, grad_a, grad_b

_subcmul = _Subcmul.apply


VERBOSE_ERROR_MESSAGE = ('Invalid value for verbose\n'
                        'verbose = 0\tdont show alg log\n,'
                        'verbose = 1\tshow the final alg res,\n'
                        'verbose = 2\tshow details of alg log')

GRAD_MODE_ERROR_MESSAGE = ('Invalid value for grad_mode\n'
                          'grad_mode = None\tfor non-gradient method\n'
                          'grad_mode = numerical_grad\tuse numerical gradient\n'
                          'grad_mode = backprop\tuse backpropagation for computing gradient\n'
                          'grad_mode = implicit\tuse implicit differentiation (w.r.t KKT) formula for computing gradient\n'
                          'grad_mode = direct\tuse closed-form formula for computing gradient\n'
                          'grad_mode = cvxpylayer\tuse cvxpylayer for computing gradient')

OUTER_METHOD_MESSAGE = ('Invalid entry for outer_method\n'
                        'outer_method = COBYLA\tnon-gradient numerical optimization method\n'
                        'outer_method = SLSQP\tgradient-based optimization method')

#  Define some basic utility functions
#  Normalize c vector
def normalize_c(c, degree = 1):
    """
    c.shape = (nIns, nVar, 1)
    Normalize c in axis 1
    """
    assert len(c.shape) == 3, 'len(c.shape) = [%d]'%len(c.shape)
    if degree == 2:
        n_c = c.norm(dim=1, keepdim=True)
    elif degree == 1:
        n_c = c.norm(p=1, dim=1, keepdim=True)
    return c/n_c  
  

#  Build a 3d tensor which enables backpropogation computaion graph
def build_3d_tensor(x): 
    return torch.stack(
            tuple(torch.stack(
                tuple(torch.cat(
                    tuple(xijk.view(-1) if isinstance(xijk, torch.Tensor) else torch.DoubleTensor((xijk,))
                          for xijk in xij))
                      for xij in xi))
                  for xi in x))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def create_cvx_layer(lp, nneg):
    """
    create a torch layer using cvxpylayer
    lp   = (c, A_ub, b_ub, A_eq, b_eq)
    nneg = True/False (decision variables are non-negative or not)
    """
    _, A_ub, _, A_eq, _ = lp
    N, m, n = A_ub.shape
    cvx_x = cp.Variable( (n), nonneg = nneg )
    cvx_c = cp.Parameter((n))
    cvx_A_ub = cp.Parameter((m, n))
    cvx_b_ub = cp.Parameter((m))

    obj = cp.Minimize(cvx_c @ cvx_x)
    cons = [cvx_A_ub@cvx_x <= cvx_b_ub]

    if A_eq is not None:
        _, m_eq, _ = A_eq.shape
        cvx_A_eq = cp.Parameter((m_eq, n))
        cvx_b_eq = cp.Parameter((m_eq))
        cons += [cvx_A_eq@cvx_x == cvx_b_eq]
        prob = cp.Problem(obj, cons)
        layer = CvxpyLayer(prob, [cvx_c, cvx_A_ub, cvx_b_ub, cvx_A_eq, cvx_b_eq], [cvx_x])
    else:
        prob = cp.Problem(obj, cons)
        layer = CvxpyLayer(prob, [cvx_c, cvx_A_ub, cvx_b_ub], [cvx_x])

    return layer

#  Define Loss Functions
def _AOE_(c, x_target, x_predict, mask= None):
    """
    x_predict, x_target and c has the shape of (nIns, nVar, 1)
    To do matrix multiplication for AOE, transpose axis 1 and 2 of c matrix
    ***    _aoe_      has the shape of (nIns, 1, 1)
    ***    mean(aoe)  returns a scalar as loss
    """

    aoe = torch.bmm(T(c), x_target - x_predict).abs_().sum(axis=1)
    return aoe   # return AOE for all u's, without aggregation

class EvaluationLimitExceeded(Exception):
    """Raise this to terminate a search procedure."""
    pass

class LossToleranceExceeded(Exception):
    """Raise this to terminate a search procedure."""
    pass

class RuntimeLimitExceeded(Exception):
    """Raise this to terminate a search procedure."""
    pass

class UnboundedWeightsValue(Exception):
    """Raise this to terminate a search procedure."""
    pass


class collect_calls:
    """
    Decorates a function with a 'calls' attribute containing
    a copy of each argument that was passed to each invocation.
    A parallel list of 'times' records the start and end time
    of each invocation.
    """
    __slots__ = ('func', 'calls', 'rvals', 'times')
    
    def __init__(self, func):
        self.func = func
        self.calls = []
        self.rvals = []
        self.times = []
        
    def __call__(self, *args):
        self.calls.append(deepcopy(args))
        tic = time.time()
        try:
            res = self.func(*args)
        finally:  # Make sure that if an exception occurs we still add the time
            toc = time.time()
            self.times.append((tic, toc))
        self.rvals.append(deepcopy(res))
        return res

# Find the loss value at each time point in time_range
def compute_loss_over_time(time_range, time_data, loss_data): 
    """
    time_range: a list of pre-defined time steps, e.g., time_range = np.linspace(0,30, 1e-3)
    time_data:  time of each iteration of the solving process
    loss_data:  loss at each iteration of the solving process

    return loss_at_each_time_step = loss at each time steps of time_range
    """
    loss_at_each_time_step = np.zeros((len(time_data), len(time_range)) )

    for ind_ins in range(len(time_data)):  # for each instance
        assert len(loss_data[ind_ins]) > 0
        if len(loss_data[ind_ins]) == 1:
            loss_at_each_time_step[ind_ins] = float(loss_data[ind_ins][0] )
        
        else:
            time_max = max(time_range)
            time_    = time_data[ind_ins].round(3)   # recorded time steps of each f(w) in the algorithm
            loss_    = loss_data[ind_ins]

            time_    = time_[np.where(time_<=time_max)]
            loss_    = loss_[np.where(time_<=time_max)]

            loss_[np.where(loss_<1e-5)] = 1e-5   

            index    = np.nonzero(np.in1d(time_range, time_))[0]     # find each time point in time_range is in time_
            interval = [(index[ind_], index[ind_+1]) for ind_ in range(len(index)-1)]

            if len(index) > 1:
                start_i, _ = interval[0]
                loss_at_each_time_step[ind_ins, :start_i] = float(999) # set loss before the first record to be 999
                _, end_i   = interval[-1]   # set loss after the last record to be the same as the last record
                loss_at_each_time_step[ind_ins, end_i:] = loss_[-1]
                for i in range(len(interval)): 
                    start_i, end_i = interval[i]
                    loss_at_each_time_step[ind_ins, start_i:end_i] = loss_[i]
            elif len(index) == 1:
                start_i, end_i = index[0], index[0]
                loss_at_each_time_step[ind_ins, :start_i] = float(999)
                loss_at_each_time_step[ind_ins, end_i:] = loss_[-1]
    return loss_at_each_time_step

def plot_success_rate(time_range,
                      res, 
                      ax,
                      plot_style):
    """
    time_range: a list of pre-defined time steps, e.g., time_range = np.linspace(0,30, 1e-3)
    res:  a list of raw experiment results 
         (loss_at_each_iteration, weights_at_each_iteration, time_of_each_iteration, total_runtime)
    plot_style:  color, linestyle, label

    plot the success rate curve: percentage of instances reaching loss<=1e-5 at each time step.
    """
    color, linestyle, label = plot_style
    assert len(res) == 4, print('res has invalid format, len(res) = %d'%len(res))
    res_l, _, res_time, _ = res    

    # collect the loss value at each time points in time_range
    loss_over_time = compute_loss_over_time(time_range,  res_time, res_l)

    success_rate = np.sum((loss_over_time<=1e-5), 0)/len(res_l)
    ax.plot(time_range, success_rate*100, alpha = 0.8, linewidth = 2,
            color = color, label = label, linestyle = linestyle, )
    ax.set_xlim([0, time_range[-1]])

    return success_rate
