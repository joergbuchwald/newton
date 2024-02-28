#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:23:14 2024

@author: buchwalj
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
#from scipy.sparse.linalg import inv
#from scipy.sparse import csc_matrix
from scipy import linalg
import time

# physical parametrization
T1 = 273
T2 = 400
K = 0.1

class Solution:
    def __init__(self, T0, Tleft, Tright, Ntime, Nspace, dx) -> None:
        self.solution = np.zeros((Ntime, Nspace))
        self.Ntime = Ntime
        self.Nspace = Nspace
        self.Tleft = Tleft
        self.Tright = Tright
        self.solution[0,:] = T0
        self.timestep = 0
        self.dx = dx
        self.applyRB()


    def newTime(self):
        self.timestep = self.timestep + 1
        try:
            self.solution[self.timestep, :] = self.solution[self.timestep-1, :]
        except IndexError:
            print("maximum timestep reached")
        self.applyRB()

    def applyRB(self):
        self.solution[self.timestep, 0] = self.Tleft
        self.solution[self.timestep, -1] = self.Tright

    def getVal(self, spaceiter):
        if spaceiter < 0:
            spaceiter = 0
        elif spaceiter > self.Nspace-1:
            spaceiter = self.Nspace-1
        return self.solution[self.timestep, spaceiter]

    def getPreVal(self, spaceiter):
        if spaceiter < 0:
            spaceiter = 0
        elif spaceiter > self.Nspace-1:
            spaceiter = self.Nspace-1
        return self.solution[self.timestep-1, spaceiter]

    def setVal(self, spaceiter, value):
        if spaceiter <= 0:
            pass
        elif spaceiter >= self.Nspace-1:
            pass
        else:
            self.solution[self.timestep, spaceiter] = value[0,0]

    def getTimestep(self):
        return self.solution[self.timestep, :]

    def getPreTimestep(self):
        return self.solution[self.timestep-1, :]

def F(solution, dt, dTleft=0, dTmiddle=0, dTright=0):
    R = []
    for spaceiter in range(0, solution.Nspace):
        Tnewleft = solution.getVal(spaceiter-1) + dTleft
        Tnewright = solution.getVal(spaceiter+1) + dTright
        Tnewmiddle = solution.getVal(spaceiter) + dTmiddle
        Toldmiddle = solution.getPreVal(spaceiter) 
        R.append(K*dt * (Tnewleft-2*Tnewmiddle+Tnewright)/(solution.dx**2) - Tnewmiddle+Toldmiddle)
    R[-1] = 0
    R[0] = 0
    return np.matrix(R).transpose()

def JF(solution, dt):
    dT = 1e-6
    J = np.matrix(np.zeros((solution.Nspace,solution.Nspace)))
    for i in range(0, solution.Nspace):
        for j in range(0, solution.Nspace):
            if (j == i-1):
                J[i,j] = (F(solution, dt, dTleft=dT)[i,0] - F(solution, dt, dTleft=0.0)[i,0] ) / dT
            elif j == i:
                J[i,j] = (F(solution, dt, dTmiddle=dT)[i,0] - F(solution, dt, dTmiddle=0.0)[i,0] ) / dT
            elif (j == i + 1):
                J[i,j] = (F(solution, dt, dTright=dT)[i,0] - F(solution, dt, dTright=0.0)[i,0] ) / dT
            else:
                J[i,j] = 0.0
    J[:,0] = 0
    J[0,:] = 0
    J[:,-1] = 0
    J[-1,:] = 0
    J[0,0] = 1
    J[-1,-1] = 1
    return J

def f(solution, dt):
    return 0.5 * np.linalg.norm(F(solution, dt))

def alpha(delta, nablaf, J):
    alpha_0 = delta/np.linalg.norm(nablaf)
    alpha_1 = np.linalg.norm(nablaf)**2 / np.matmul(
                           nablaf.transpose(), np.matmul(
                           np.matmul(J.transpose(), J), nablaf))[0,0]
    return np.min([alpha_0, alpha_1])


def tau(delta, s_nt, s_cy):
    p_0 = np.matmul((s_nt-s_cy).transpose(), s_nt-s_cy)
    p_1 = np.matmul(2*s_cy.transpose(), s_nt-s_cy)
    p_2 = np.matmul(s_cy.transpose(), s_cy) - delta**2
    assert p_0.shape == (1, 1)
    assert p_1.shape == (1, 1)
    assert p_2.shape == (1, 1)
    taupm = np.roots([p_0[0,0], p_1[0,0], p_2[0,0]])
    tau = np.max(taupm)
    return tau

def rho(R, J, p, sol_temp, dt):
    f0 = 0.5 * np.matmul(R.transpose(), R)[0,0]
    gn = np.matmul(J.transpose(),R)
    Bn = np.matmul(J.transpose(),J)
    g = 0.5 * np.linalg.norm(F(sol_temp, dt))**2
    mp = f0 + np.matmul(gn.transpose(), p)[0,0] + 0.5 * np.matmul(p.transpose(), np.matmul(Bn,p))[0,0]
    return (f0 - g )/(f0-mp)

# numerical Definitions:
Ntime = 2
Nspace = 5
dx = 0.1
dt = 0.5


sol = Solution(T1, T1, T2, Ntime, Nspace, dx)

# newton tr settings 
damping = 1
max_iter = 2
tol = 1e-10
delta_0 = 0.2
eta_1 = 0.001
eta_2 = 0.25
eta_3 = 0.75
t1 = 0.25
t2 = 2.0
delta_tol = 1e-12

for i in range(0, Ntime-1):
    niter = 0
    error = 9e9
    print(f"Time t = {i*dt}, time step {i}")
    sol.newTime()
    xnorm = np.linalg.norm(sol.getTimestep())
    delta = delta_0 * xnorm

    while ((error > tol) and (niter <= max_iter)):   
        dx_pre = np.sum( np.abs( sol.getTimestep()-sol.getPreTimestep() ))

        
        J = JF(sol, dt)       
        R = F(sol, dt) 
        s_nt = np.matrix(linalg.solve(J, -R))
        #scale s, delta
        nablaf =  np.matmul(J.transpose(), R)
        norms_nt = np.linalg.norm(s_nt)
        print(f"Debug Newton update: {s_nt}")
        while True:
            if norms_nt <= delta:
                sn = s_nt
            else:
                s_cy = - alpha(delta, nablaf, J) * nablaf
                print(f"Debug Cauchy update: {s_nt}")
                if np.linalg.norm(s_cy) > delta:
                    sn = s_cy
                else:
                    sn = s_cy + tau(delta, s_nt, s_cy) * (s_nt-s_cy)
            sol_temp = Solution(T1, T1, T2, Ntime, Nspace, dx)
            sol_temp.timestep = 1
            sol_temp.solution = np.concatenate(([copy.deepcopy(sol.getPreTimestep())], [copy.deepcopy(sol.getTimestep())]), axis=0)
            update = np.matrix(sol_temp.getTimestep().transpose()).transpose() + sn
            for spaceiter in range(0, sol.Nspace):
                sol_temp.setVal(spaceiter, update[spaceiter])
            rho_ = rho(R, J, sn, sol_temp, dt)
            print("rho:", rho_)
            #rescale sn, delta
            #precheck
            print(f"Debug residual: {R}")
            print(f"Debug Jacobian: {J}")  
            print(f"Debug update: {sn}")
            realupdate = np.matrix(sol.getTimestep().transpose()).transpose() + sn
            #postcheck 
            for spaceiter in range(0, sol.Nspace):
                sol.setVal(spaceiter, realupdate[spaceiter])
            Rnew = F(sol, dt) 
            if rho_ < eta_2:
                print("rho too small, decrease trust radius")
                delta = t1 * delta
            elif rho_ > eta_3:
                print("rho too big, increase trust radius")
                delta = t2 * delta
            print("trust radius:", delta)
            if rho_ > eta_1:
                break
            elif delta < delta_tol:
                print("trust radius smaller than delta_tol")
                break
            
        error = np.linalg.norm(F(sol, dt))
        niter = niter + 1
        print(f"Iteration {niter}: error: {error} tol: {tol}")


print(sol.solution)
timesteps = [0,1]
for it in timesteps:
    plt.plot(sol.solution[it,:], label=f"t = {dt*it} s")
plt.legend()
plt.xlabel("x / m")
plt.ylabel("T / K")
plt.show()