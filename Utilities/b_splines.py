import torch
import numpy as np

def set_knots(anz_splines,x):
    dist = (max(x)-min(x))/(anz_splines-3)
    knots = np.linspace(min(x)-3*dist,max(x)+3*dist,anz_splines+4)
    return knots

def design_matrix(knots,x,deg=3):
    design_matrix=torch.zeros((len(x),len(knots)-deg-1))
    for i in range(len(x)):
        for j in range(len(knots)-deg-1):
            design_matrix[i][j]=spline(t=j,deg=deg,knots=knots,x=x[i])
    return design_matrix.type(torch.float64)

def spline(t,deg,knots,x):
    # t index of spline
    # deg is degree
    # knots ordered list of knots
    # x: point of evaluation
    if deg==0:
        if x>=knots[t] and x<knots[t+1]:
            return 1
        else:
            return 0
    else:
        return (x-knots[t])/(knots[t+deg]-knots[t])*spline(t,deg-1,knots,x)+(knots[t+deg+1]-x)/(knots[t+deg+1]-knots[t+1])*spline(t+1,deg-1,knots,x)