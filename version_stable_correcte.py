#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:50:20 2021

@authors: lagree_emmanuel & marion_antoine

Final project : Computational Neuroscience methods (Universite de Paris, ENS Paris)
This scrip implements Fig2 of Sussilo & Abbott (2009, *Generating Coherent patterns of
Activity from Chaotic Neural Networks) 
"""


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10645)

N_G = 1000 ### mauvaise valeur
g_GG = 1
p_GG = 0.1
p_z = 1
g_Gz = 1
g_GF = 0
alpha = 1
tau = 10

def define_initial_weights(N_G, p_z):  
    return np.random.normal(0, np.sqrt(1/N_G), N_G)

def define_connection_matrix_J(N, proba):
    """ Given the size and a probability for each entry to be set to zero,
    returns a N*N random matrix. Used to define J_GG """
    
    J = np.random.normal(0,np.sqrt(1/(N*proba)), [N, N])
    phi = lambda x : x if np.random.random() < proba else float(0) # Sets x to zero with a probability 1 - proba, else does nothing
    set_to_zero = np.vectorize(phi)
    return set_to_zero(J)

def update_x_t_by_euler_method(x, r, z, dt, tau, N_G, g_GG, J_GG, g_Gz, J_Gz):
    """ This function computes x(t+dt) knowing x(t) for every i
    This holds for Figure 2 and a simpler version of the differential equation
    x_in, r_in and z are functions of t ; tau, N_G, g_GG, J_GG, g_Gz and J_Gz are constants"""
    
    x_out = []
    for i in range(N_G):
        variation = -x[i] + g_GG * np.dot(J_GG[i], r) + g_Gz * J_Gz[i] * z
        variation = variation / tau
        x_out.append(x[i] + variation * dt)
    
    return np.array(x_out)

def compute_z(w, r):
    """ w and r as numpy arrays of same size. They are the weights and the 
    firing rates at time t. The function returns the dot
    product of the two, which is the output of the network"""
    return np.dot(w, r)

def compute_error(w, r, f):
    """ The function returns the error (scalar)"""
    return np.dot(w, r) - f

def modification_rule_weights(w, e_minus, P, r):
    """ w, e_minus and r are column arrays, P is a matrix 
    w is taken at time t - Delta_t ; the others at time t """
    w = w - e_minus * np.dot(P, r)
    return w

def modification_rule_matrix(P, r):
    """ P is a matrix, taken at time t - Delta_t
    r is the synaptic array taken at time t """
    D = np.outer(r, r)
    D = np.dot(D, P)
    D = np.dot(P, D)
    Pr = np.dot(P, r)
    a = np.dot(r, Pr)
    return P - D/(1 +a)

D = 4000 #duration in ms
dt = 1 #time step in ms
time = np.arange(0, D, dt)

J_GG = define_connection_matrix_J(N_G, p_GG)
J_Gz = np.random.uniform(-1, 1, N_G)

def computations(index, Fi):
    """ This function computes the activity of the neurons for one target 
    function (Fi), updates the weights during the training and stores
    everything we are interested in. It is the core of the program. """
    
    Fx=[]
    Fy=[]
    DW=[]
    Z = np.zeros(int(D/dt))
    F = np.zeros(int(D/dt))
    DW = np.ones(int(D/dt))*(-3)

    # Define initial variables
    P = np.eye(N_G)/alpha
    w = define_initial_weights(N_G, p_z)
    x = np.random.uniform(0, 1, N_G)
    r = np.tanh(x)
    z = compute_z(w,r)

    # Before updating the weigths
    W=[]
    X=[]
    for i in range (0,4):
        W.append(np.zeros(int(D/dt)))
        X.append(np.zeros(int(D/dt)))
    mod = 1
    init=True
    finish=True

    for i in range (0,len(time)):
        t=time[i]
        f=0

        x = update_x_t_by_euler_method(x, r, z, dt, tau, N_G, g_GG, J_GG, g_Gz, J_Gz)
        r = np.tanh(x)
        z = compute_z(w,r)
    
        Z[i]=(z)
        e_minus = compute_error(w, r, f)
        if t>D/4 and t<3*D/4 and t%10==0:
        
            if init==True:
                print('End of the spontaneaous activity #' + str(index+1))
                init=False
            Fx.append(t)
            f = Fi(t) 
            Fy.append(f)
        
            P = modification_rule_matrix(P, r)
            e_minus = compute_error(w, r, f)
            w = modification_rule_weights(w, e_minus, P, r)
            DW[i]=np.sum(np.absolute(np.dot(e_minus*P,r)))-3
            
        if t - mod/10*D > 0:
            print('*', end = " ")
            mod+=1
        if t>3*D/4 and t%10==0 and finish==True:
            print('End of the learning phasis #' + str(index+1))
            finish=False
    
        for j in range(0,4):
            W[j][i]=w[j]
            X[j][i]=x[j]
        """
        Pr = np.dot(P, r)
        a = np.dot(r, Pr)
        e_plus = e_minus*(1-a)
        """
    print('\n') 
    return Z, F, Fx, Fy, DW, W, X

def settings_all_plots(d=D):
    """ Settings we use for every plot """
    plt.axvline(d/4, -2, 2, color = 'gray', linewidth = 0.7)
    plt.axvline(3*d/4, -2, 2, color = 'gray', linewidth = 0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Relative scale')
    plt.legend()

def display_result_one_function(Z, F, Fx, Fy, DW, W, X, function_name, s_a_p=settings_all_plots):
    """ This function displays the results obtained by computations() :
        it is called for each of the three functions we target """
    
    
    # Main plot
    plt.figure(figsize = (16,4))
    plt.plot(time,Z, color = 'red', label = 'Output')
    plt.plot(Fx,Fy, color = 'blue', label = 'Target')
    plt.plot(time,DW, color = 'green', label = 'variation of weights')
    plt.title('Evolution of the network output over time. Target is '+function_name)
    s_a_p()
    plt.yticks([])
    plt.show()
    
    # Plot the activity of four specific weights
    plt.figure(figsize = (16,4))
    for i in range(0,4):
        plt.plot(W[i], label= 'Weight #'+str(i+1))
    s_a_p()
    plt.title("Evolution of four specific weights")
    plt.show()

    # Plot the activity of four specific activities
    plt.figure(figsize = (16,4))
    for i in range(0,4):
        plt.plot(np.array(X[i]) + 5*i, label= 'Neuron #'+str(i+1))
    s_a_p()
    plt.yticks([])
    plt.title("Evolution of four specific neurons")
    plt.show()
    

# Definition of the targets
def sin(t, period = 600):
    """ Sinus function"""
    return np.sin(2*np.pi*t/period)

def triangle(t, period = 600):
    """triangle function"""
    a = 3/period
    b = period/4*a
    j = t%period - period/2
    return a*j+b if j<0 else -a*j+b

def creneau(t, period = 600):
    """discontinuous function"""
    j = t%period-period/2
    return 0.5 if j<0 else -0.5    

list_functions = [sin, triangle, creneau]
dict_functions = {sin: 'sin', triangle: 'triangle', creneau : 'creneau'}

# Here we go !
if __name__ == '__main__':
    for index, Fi in enumerate(list_functions):
        Z, F, Fx, Fy, DW, W, X = computations(index, Fi)
        display_result_one_function(Z, F, Fx, Fy, DW, W, X, dict_functions[Fi])
    print('\n*end*')
