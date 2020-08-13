# This model optimizes prices for N consecutive time periods each of which has its own demand function.
# It is assumed that the stock level of the product is limited and
# the goal is to sell out the stock in time maximizing the revenue.

import sympy as sy
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from numpy import linspace

def tabprint(msg, A):
    print(msg)
    print(tabulate(A, tablefmt="fancy_grid"))


#Example: we want to sell at a range from 0.049cents/kwh to 0.089cents / kwh 

price_floor=0.04
price_ceiling=1
plevels = linspace(price_floor,price_ceiling,20) # allowed price levels    
C = 1500                     # stock level

price = sy.symbols("price")

def rectified(f):
    return sy.Piecewise( (0, f < 0), (f, True))

# Demand functions estimated for different periods
#demands = [rectified(900 - 10*price), # period 1 High Demand
           # rectified(10 - 60*price),# period 2 Middle Demand
           # rectified(10 - 70*price),# period 3] Low Demand
           # rectified(80),
           # rectified(800)] #period 4 Low Demand, demand is fixed number
           
demands = [rectified(900 - 10*price), # period 1 High Demand
            rectified(10 - 60*price),# period 2 Middle Demand
            rectified(10 - 70*price)]# period 3] Low Demand

          

0
# Evaluate values of demand functions for each price level (Non-negative)
D = np.array([[q.subs(price, p) for p in plevels] for q in demands])
tabprint("D =", D)

# Evaluate revenue for each demand function and each price level (Demand * Price)
R = np.array([[p*q.subs(price, p) for p in plevels] for q in demands])
tabprint("R =", R)

# Now we solve the following optimization problem:
# (q is demand, P is price, T is the number of time periods, and K is the number of price levels)


L = len(demands)*len(plevels)

# First, we generate a binary mask to ensure that all z's 
# in one time interval sum up to 1.0, that is z.M = B
#This fulfils constraint 1 in word document
M = np.array([[
    1 if i >= len(plevels)*j and i < len(plevels)*(j+1) else 0
    for i in range(L)
] for j in range(len(demands))])

tabprint("M = ", M)

B = [1 for i in range(len(demands))]

# Second, we ensure that the sum of all demands is less than the availbale stock level,
# that is z.Df <= C
Df = np.array(D).reshape(1, L)

res = linprog(-np.array(R).flatten(), 
              A_eq=M, 
              b_eq=B, 
              A_ub=Df, 
              b_ub=np.array([C]), 
              bounds=(0, None))

print("Revenue value: $", -res.fun)

# Each column of the solution matrix corresponds to a time period (one week).  
# Each row corresponds to z value that can be interpreted as the percentage 
# of time z't price level should be used in the corresponding time period. 

tabprint("Price schedule:", np.array(res.x).reshape(len(demands), len(plevels)).T)

# DmOriginal = (np.array(res.x).reshape(len(plevels), len(demands)).T)

# DmNew = (np.array(res.x).reshape(len(plevels), len(demands)).T)
# Dm1 = plevels* DmOriginal
# m1=np.asmatrix(DmOriginal)
# m2=np.asmatrix(plevels)
# print(DmOriginal)
# print(m1)
# # print(np.shape(m1))
# m3= np.reshape(m2,(len(plevels),1))
# print(m3)
# # print(np.shape(m3))
# m4 = m1.dot(m3)
# print(m4)

dm1= (np.array(res.x).reshape(len(demands), len(plevels)).T)
# print(dm1)
# print(np.shape(dm1))
dm2 = np.reshape(plevels,(1,len(plevels),)) #plevels reshape
# print(dm2)
# print(np.shape(dm2))
DmFinal = dm2.dot(dm1)

# print(DmFinal)

tabprint("Price schedule:", DmFinal)
