import numpy as np
import torch
from torch import sin, cos 

def f(x):
    return 4*pow(x,2)+3*sin(x)

x = torch.tensor([1.,2.],requires_grad=True,dtype=float)
deltax =torch.tensor([0.1, 0.01, 0.001, 0.0001],dtype=float)

def NumSolDiff(arges):
    y1 = f(x+h)
    y2 = f(x-h)
    if arges == 1:
        yDiff = (y1-y2)/(2*h)
    elif arges == 2:
        yDiff = (y1+y2-2*y)/(pow(h,2))
    else:
        yDiff = ("That's wrong, please input 1 or 2 !")
    return yDiff


######## Analytical Solution ########
# yDiff1 = 8 * x + 3*cos(x) 
# yDiff2 = 8 - 3*sin(x)
y = f(x)
yDiff1 = torch.autograd.grad(y,x,grad_outputs=torch.ones(x.shape),create_graph=True,retain_graph=True)
yDiff2 = torch.autograd.grad(yDiff1[0],x, grad_outputs=torch.ones(x.shape), create_graph=False)
print("Analytical solution:  y = ", y ,"\tyDiff1 = ", yDiff1 ,"\tyDiff2 = ", yDiff2, "\n")


# ######## Numerical Solution ########
for h in deltax:
    yDiff1 = NumSolDiff(1)
    yDiff2 = NumSolDiff(2)
    print("numerical solution:  y = ", y ,"\tyDiff1 = ", yDiff1 ,"\tyDiff2 = ", yDiff2)


    
