# Learning Linear Programs from Optimal Decisions

This branch contains the code for reproducing the results of our NeurIPS paper "Learning Linear Programs from Optimal Decisions".
 

--- 
---

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> the main required packages and python environment are the follows:  
>  - cvxpylayer
> - PyTorch v1.6 nightly build
> - SciPy v1.4.1, 
> - python 3.7.7

## Experiment 1_a
To run the experiment 1a, run this command

```exp1a
python exp1a.py
```

The script does the following:
> 1. Generate 100 baseline convexhull for six different LP sizes (if not exist).
> 2. Generate corresponding synthetic parametric linear programs for the experiment.<br>
   100 PLP instances for each LP size (inequality constraints only)
> 3. run experiment with <img src="https://render.githubusercontent.com/render/math?math=\text{Random Search},\text{COBYLA},\text{SQP}_\text{cvx},\text{SQP}_\text{bprop},\text{SQP}_\text{impl},\text{SQP}_\text{dir}">
> 4. Generate figures <br>
   Note. **fig_exp1a_10var80cons.pdf** is Figure 3 in the main paper, and the rest of the figures are Figure 7 in the Appendix.

**NOTE.** The complete Exp1a (including experiments for all six classes of PLP instances) might take several hours to complete. If you only want to reproduce Figure 3 in the main paper, you need to change the *LP_SIZE* variable in the *main()* function.

## Experiment 1_b
To run the experiment 1b, run this command

```exp1b
python exp1b.py
```

> The script does the following:
> 1. Generate 100 baseline convexhull with <img src="https://render.githubusercontent.com/render/math?math=D=10, M_1 = 80">  (if not exist) 
> 2. Generate synthetic parametric linear programs for the experiment. <br> 
    100 PLP instances with <img src="https://render.githubusercontent.com/render/math?math=D=10, M_1 = 80, M_2 = 2"> ( inequality and equality constraints)
> 3. run experiment with <img src="https://render.githubusercontent.com/render/math?math=\text{Random Search},\text{COBYLA},\text{SQP}_\text{cvx},\text{SQP}_\text{bprop},\text{SQP}_\text{impl},\text{SQP}_\text{dir}">
> 4. Generate a figure called **fig_exp1b_10var80cons.pdf**, which is Figure 8 in the Appendix.

## Experiment 1_c
To run the experiment 1c, run this command

```exp1c
python exp1c.py
```

> The script does the following:
> 1. Generate 100 baseline convexhull with <img src="https://render.githubusercontent.com/render/math?math=D=10, M_1 = 80"> (if not exist) 
> 2. Generate synthetic linear programs for experiment/ <br> 
    100 LP instances with <img src="https://render.githubusercontent.com/render/math?math=D=10, M_1 = 80">  (inequality constraints only).
> 3. Run experiment with <img src="https://render.githubusercontent.com/render/math?math=\text{COBYLA},\text{SQP}_\text{cvx},\text{SQP}_\text{bprop},\text{SQP}_\text{impl},\text{SQP}_\text{dir}">
> 4. Generate a figure called **fig_exp1c_10var80cons.pdf** which is Figure 9 in the Appendix.


## Experiment 2
To run experiment 2, run this command

```exp2
python exp2.py
```

> The script will do the following:
> 1. Generate 100 minimum cost multi-commodity flow (MCF) instances.
> 2. Run experiment with<img src="https://render.githubusercontent.com/render/math?math=\text{Random Search},\text{COBYLA},\text{SQP}_\text{cvx},\text{SQP}_\text{bprop},\text{SQP}_\text{impl},\text{SQP}_\text{dir}">
> 3. Generate a figure called **fig_exp2.pdf** which is Figure 5 in the Appendix.

<br>

--- 
---

<br>

## Results

The results of our experiment are presented as figures which can be found in the main paper and Appendix:


| Experiment         | Figure in the paper |
| ------------------ |----------------    | 
| Experiment 1(a)    |    Figure 3 in main paper      | 
|                    |    Figure 7 in Appendix       | 
| Experiment 1(b)    |    Figure 8 in Appendix       | 
| Experiment 1(c)    |    Figure 9 in Appendix     | 
| Experiment 2       |    Figure 5 (MCF on Nguyen-Dupuis Graph)       | 


