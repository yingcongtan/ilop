# Copyright (C) to Yingcong Tan, Daria Terekhov, Andrew Delong. All Rights Reserved.
# Script for reproducing the results of NeurIPS paper - 'Learning Linear Programs from Optimal Decisions'


# Functions for generating baseline convexhull

import numpy as np
from scipy.spatial import ConvexHull
import os
import matplotlib.pyplot as plt


def plot_LP(hull, points, vertices, axes = None):
    for simplex in hull.simplices:
        plt.plot(*points[simplex].T, 'k-', linewidth=2)
    
    axes.plot(*np.array(vertices).T, 'o', 
            mfc='None', mec='orange', ms =10, mew=3, 
            alpha=1, label='vertices')

def record_baseline_convexhull(filename, A, b, vertices):
    """
    Save LP instance in txt file in the following format:
    m,n  # num of rows and columns
    A_ub
    b_ub
    vertex
    """
    m,n = A.shape
    with open(filename, 'w') as LP_file:
        LP_file.write("Instance Size\n")
        np.savetxt(LP_file, np.array(A.shape).reshape((1,2)), fmt="%.d")
        LP_file.write("\n")

        LP_file.write("A_ub\n")
        np.savetxt(LP_file, A, fmt="%.6f")
        LP_file.write("\n")
        LP_file.write("b_ub\n")
        np.savetxt(LP_file, b, fmt="%.6f")
        LP_file.write("\n")        
        
        LP_file.write("%d vertices\n"% len(vertices) )
        np.savetxt(LP_file, vertices, fmt="%.6f")
        LP_file.write("\n")

def read_baseline_convexhull(filename):
    """
    Read baselin_LP instance in txt file in the following format:
    nCons, nVar  # num of rows and columns
    A_ub
    b_ub
    vertex
    """
    
    
    lines=[]    
    with open(filename, "rt") as fp:
        lines=fp.readlines()
        for line in fp:
            lines.append(line.rstrip('\n')) 
            
    nCons, nVar = np.fromstring(lines[1], dtype=float, sep=' ').astype(np.int)
    
    temp = [lines[j+1:j+1+nCons] for j in range(len(lines)) if "A_ub" in lines[j]]
    A_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]]) 
    
    temp = [lines[j+1:j+1+nCons] for j in range(len(lines)) if "b_ub" in lines[j]]
    b_ub = np.vstack([np.fromstring(row, dtype=float, sep=' ') for row in temp[0]]) 
    
    temp = [lines[j+1:-1] for j in range(len(lines)) if "vertices" in lines[j]]
    vertices = np.vstack([np.fromstring(row, dtype=float, sep=' ')for row in temp[0]]) 
    
    return A_ub, b_ub, vertices

def _convexhull_generator(nIns, 
                          nVar, 
                          nCons,
                          nPoints,
                          directory,
                          verbose):
    ind_ins = 0
    np.random.seed(0)
    while ind_ins < nIns: 
        points = np.random.normal(0,1, (nPoints, nVar)) 
        # Note that this means there will be *AT MOST* nVar constraints
        hull = ConvexHull(points) 
        m, numVars = hull.equations.shape
        if m == nCons:
            # print('************************************')
            assert numVars-1 == nVar, "Number of variables after convex hull function not equal to the original number of vars"
            A = -1 * hull.equations[:,0:nVar]
            b = hull.equations[:,nVar]
            # important: the convexhull generates Ax >= b (not Ax <= b as linprog requires)
            # convert A and b s.t. the constraints have the form as: Ax <= b
            A = np.round(np.negative(A),6)
            b = np.round(np.negative(b),6)
            if verbose >=1:
                print("==== Generate convexhull [%d Var, %d Cons, Ins %d] "%(nVar, nCons, ind_ins))

            # Generate points
            # vertices == use built in function
            vertex = points[hull.vertices,:] 
            if verbose == True:
                print("\t %d vertices"%vertex.shape[0])
            
            dir_ = directory + "LP_baseline/%dvar%dcons/"%(nVar, nCons)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            filename = dir_ + "%dvar%dcons_LP%s.txt"%(nVar, nCons, ind_ins)
            if verbose == True:
                print('\tbaseline_lp saved to %s'%filename)
            record_baseline_convexhull(filename, A, b, vertex)

            # plot the
            if plot_LP and nVar <=2:
                fig_save_dir = filename = dir_ + "%dvar%dcons_LP%s.png"%(nVar, nCons, ind_ins)
                fig, ax = plt.subplots(1,1, figsize=(5,5))
                plot_LP(hull, points, vertex, ax)
                ax.set_title('Baseline convexhull [%d var %d cons ins %d]'%(nVar, nCons, ind_ins), fontsize=14)
                ax.set_xlim(-3,3)
                ax.set_ylim(-3,3)
                plt.show(block=False)
                fig.savefig(fig_save_dir, dpi=100)


            ind_ins += 1
            # print('************************************')
            plt.close('all')


def convexhull_generator(nVar, 
                         nCons, 
                         nIns   = 100, 
                         directory = None, 
                         verbose   = 0):

    direct = './' if directory is None else directory

    var_range = np.array([2,10])  # num of variables

    cons_range = np.array([[ 4,  8, 16], # num constraints for 2D case
                           [20, 36, 80]]) # num constraints for 10D case

    # number of random generated point which will be used for constructing convexhull 
    # using scipy.spatial.convexhull function
    # These are suggested values for efficient LP generation, you might also use other values
    numPoints = [[5,  8,  150], # corresponds to 2D cases
                 [12, 12, 13]]  # corresoonds to 10D cases

    assert nVar  is not None
    assert nCons is not None
    assert nVar  in var_range, 'nVar = %s, var_range = %s'%(nVar, var_range)
    assert nCons in cons_range 


    var_i = int(np.where(var_range == nVar)[0])
    cons_i = int(np.where(cons_range[var_i].ravel() == nCons)[0])

    nPoints = numPoints[var_i][cons_i]

    _convexhull_generator(nIns  = nIns,
                          nVar  = nVar,
                          nCons = nCons,
                          nPoints   = nPoints, 
                          directory = direct, 
                          verbose   = verbose)

if __name__ == "__main__":
    print("Arguments count: %d"%len(sys.argv))
    for i, arg in enumerate(sys.argv):
        print("Argument %d: %s"%(i, arg))
        

    convexhull_generator(*sys.argv[1:])                          