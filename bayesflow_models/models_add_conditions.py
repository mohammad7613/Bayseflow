import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import bayesflow as bf
import torch
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
from time import time
import matplotlib.pyplot as plt
import random
# from bayesflow.simulation import Configurator

RNG = np.random.default_rng(2023)

# Discrete time-to-arrival (TTA) flags used to condition the simulators/adapters.
CONDITIONS = np.array([2.0, 3.5, 5.0])

def prior_1a():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0.0, 0.3),
        'varsigma': RNG.uniform(0.0, 0.3),
    }

def prior_1b():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'mu_tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0.0, 0.3),
        'varsigma': RNG.uniform(0.0, 0.3),
    }

def prior_1c():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.4),
        'tau_m': RNG.uniform(0.06, 0.6),
        'sigma': RNG.uniform(0, 0.1),
        'varsigma': RNG.uniform(0, 0.1),
    }

def prior_2():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0.0, 0.3),
        'varsigma': RNG.uniform(0.0, 0.3),
        'gamma': RNG.uniform(0,3.0)
    }

def prior_3():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0.0, 0.3),
        'varsigma': RNG.uniform(0.0, 0.3),
        'theta': RNG.uniform(0,0.3)
    }

def prior_4a():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'tau' : RNG.uniform(0.1,1),
        'sigma_e': RNG.uniform(0,0.3),
        'sigma_k': RNG.uniform(0,0.3),
        'varsigma': RNG.uniform(0, 0.3),
        'k': RNG.uniform(0.05, 0.4),
        'theta': RNG.uniform(0,1)
    }
def prior_4b():
     # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # mu_tau_e ~ U(0.05, 0.6)
    # tau_m ~ U(0.06, 0.8)
    # sigma_e ~ U(0, 0.3)
    # varsigma ~ U(0, 0.3)
    # theta ~ U(0,1)
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma_e': RNG.uniform(0,0.3),
        'varsigma': RNG.uniform(0, 0.3),
        'theta': RNG.uniform(0,1)
    }

def prior_5():
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0,0.3),
        'varsigma': RNG.uniform(0, 0.3),
        'a_slope': RNG.uniform(0.01,0.9)
    }
def prior_6():
     # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # mu_tau_e ~ U(0.05, 0.6)
    # tau_m ~ U(0.06, 0.8)
    # sigma ~ U(0, 0.3)
    # varsigma ~ U(0, 0.3)
    # lambda ~ U(.5,4)
    return {
        'drift': RNG.uniform(-3.0, 3.0),
        'boundary': RNG.uniform(0.5, 2.0),
        'beta': RNG.uniform(0.1, 0.9),
        'mu_tau_e': RNG.uniform(0.05, 0.6),
        'tau_m': RNG.uniform(0.06, 0.8),
        'sigma': RNG.uniform(0,0.3),
        'varsigma': RNG.uniform(0, 0.3),
        'lamda': RNG.uniform(0.5,4)
    }

def prior_7():
    # Prior ranges for the simulator
    # mu_drift ~ U(0.01, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # tau ~ U(0.1, 1.0)
    # sigma ~ U(0, 2)
    # Eta ~ U(0.0, 2.0)
    return {
        'mu_drift': np.random.uniform(0.01, 3.0),
        'boundary': np.random.uniform(0.5, 2.0),
        'beta': np.random.uniform(0.1, 0.9),
        'tau': np.random.uniform(0.1, 1.0),
        'sigma': np.random.uniform(0,2),
        'eta': np.random.uniform(0.0, 2.0),
        'a_slope': np.random.uniform(0.01,0.9)
    }

def prior_8():
    # Prior ranges for the simulator
    # mu_drift ~ U(0.0, 3.0)
    # boundary ~ U(0.2, 2.0)
    # tau ~ U(0.1, 1.0)
    # sigma ~ ~ U(.5, 4) # is 1/lambda
    # gamma ~ U(-3, 3)
    # eta ~ U(0.0, 2.0)
    return {
        'mu_drift': np.random.uniform(0.0, 3.0),
        'boundary': np.random.uniform(0.2, 2.0),
        'beta': np.random.uniform(0.1, 1.0),
        'tau': np.random.uniform(0.1, 1.0),
        'sigma': np.random.uniform(0.5,4),
        'gamma':  np.random.uniform(-3,3),
        'eta': np.random.uniform(0.0, 2.0)
    }
#
def prior_9():
    # Prior ranges for the simulator
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # t_e ~ U(0.05, 0.6)
    # t_m ~ U(0.06, 0.8)
    # sigma_e ~ U(0, 0.3)
    return {
        'drift': np.random.uniform(-3.0, 3.0),
        'boundary': np.random.uniform(0.5, 2.0),
        'beta': np.random.uniform(0.1, 0.9),
        't_e': np.random.uniform(0.05, 0.6),
        't_m':  np.random.uniform(0.06, 0.8),
        'sigma_e': np.random.uniform(0, 0.3)
    }
#
def prior_10():
   # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # t_e ~ U(0.05, 0.6)
    # t_m ~ U(0.06, 0.8)
    # sigma_e ~ U(0, 0.3)
    # gamma ~ ~ U(0.1, 4)
    return {
        'drift': np.random.uniform(-3.0, 3.0),
        'boundary': np.random.uniform(0.5, 2.0),
        'beta': np.random.uniform(0.1, 0.9),
        't_e': np.random.uniform(0.05, 0.6),
        't_m':  np.random.uniform(0.06, 0.8),
        'sigma_e': np.random.uniform(0, 0.3),
        'gamma' : np.random.uniform(0.1, 4)
    }
#
def prior_11():
    # Prior ranges for the simulator
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 4.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # tau ~ U(0.1, 1.0)
    # sigma_e ~ U(0.1, 4.0)
    # gamma ~ ~ U(0.5, 4.0) 
    return {
        'drift': np.random.uniform(-3.0, 3.0),
        'boundary': np.random.uniform(0.5, 4.0),
        'beta': np.random.uniform(0.1, 0.9),
        'ndt': np.random.uniform(0.1, 1.0),
        'sigma': np.random.uniform(0.1, 4.0),
        'gamma' : np.random.uniform(0.5, 4.0)
    }
#
def prior_12():
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.5, 4.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # tau ~ U(0.1, 1.0)
    # Eta ~ U(0.01, 2.0) 
    return {
        'drift': np.random.uniform(-3.0, 3.0),
        'boundary': np.random.uniform(0.5, 4.0),
        'beta': np.random.uniform(0.1, 0.9),
        'ndt': np.random.uniform(0.1, 1.0),
        'eta': np.random.uniform(0.01, 2.0),
    }
#
def prior_13():
       # Prior ranges for the simulator
    # drift ~ U(-3.0, 3.0)
    # boundary ~ U(0.1, 2.0)
    # beta ~ U(0.1, 0.9)  # relative start point
    # t_m ~ U(0.06, 0.5)
    # sigma_e ~ U(0, 0.3)
    # lambda ~ U(.01, 2.0) # is 1/lambda
    # k ~ U(.05, 0.4) 
    return {
        'drift': np.random.uniform(-3.0, 3.0),
        'boundary': np.random.uniform(0.1, 2.0),
        'beta': np.random.uniform(0.1, 0.9),
        't_m': np.random.uniform(0.06, 0.5),
        'sigma_e': np.random.uniform(0, 3.0),
        'lam': np.random.uniform(0.01, 2.0),
        'k': np.random.uniform(0.05, 0.4)
    }
##
 





def simulator_1a(
    drift,
    boundary,
    beta,
    mu_tau_e,
    tau_m,
    sigma,
    varsigma,
    number_of_trials,
    tta_condition,
    dc=1.0,
    dt=0.005,
):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        # Lightly modulate drift/boundary so shorter TTAs produce more urgent evidence.
        tta_scale = CONDITIONS.mean()
        boundary_cond = boundary * (tta_condition / tta_scale)
        drift_cond = drift * (tta_scale / max(tta_condition, 1e-3))
        evidence = boundary_cond * beta
        while 0 < evidence < boundary_cond:
            evidence += drift_cond * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)
        z= np.random.normal(tau_e_trial, sigma)
        if evidence >= boundary_cond:
            chiocert= tau_e_trial + rt + tau_m

        else:
            chiocert= -tau_e_trial - rt - tau_m
        x = [chiocert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_1b(drift, boundary, beta, mu_tau_e, mu_tau_m, sigma, varsigma, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        ndt_trial = np.random.normal(mu_tau_e + mu_tau_m, varsigma)
        z = np.random.normal(ndt_trial - mu_tau_m, sigma)
        if evidence >= boundary:
            choicert =  ndt_trial + rt
        
        else:
            choicert = -ndt_trial - rt
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_1c(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        z = 0
        while True:
            # sual encoding time for each trial
            tau_e_trial = mu_tau_e + np.random.uniform(-.5*np.sqrt(12)*varsigma,.5*np.sqrt(12)*varsigma)
            z = np.random.normal(tau_e_trial, sigma)
            if z > 0 and z < .5:
                break

        if evidence >= boundary:
            choicert =  tau_e_trial + rt + tau_m
            
        else:
            choicert = -tau_e_trial - rt - tau_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_2(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, gamma, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

    # N200 latency
        z = np.random.normal(gamma*tau_e_trial, sigma)
        
        if evidence >= boundary:
            choicert =  tau_e_trial + rt + tau_m
            
        else:
            choicert = -tau_e_trial - rt - tau_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_3(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, theta, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

    # N200 latency, z ~ (1-theta)*normal() + theta*U(0,.3)
        z = np.random.normal(tau_e_trial, sigma)   

        if evidence >= boundary:
            ddm_choicert =  tau_e_trial + rt + tau_m
            
        else:
            ddm_choicert = -tau_e_trial - rt - tau_m
            
        # lapse distribution U(-maxrt, maxrt)
        uniform_choicert = np.random.uniform(-5, 5)
        
        # RT*ACC ~ (1-theta)*DDM + theta*U(-maxrt,maxrt)
        rng = np.random.uniform(0,1)   
        if rng <= 1-theta:
            choicert = ddm_choicert
        else:
            choicert = uniform_choicert
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_4a(drift, boundary, beta, mu_tau_e, tau_m, tau, sigma_e, sigma_k, varsigma, k, theta, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

    # N200 latency
        z1 = np.random.normal(tau_e_trial, sigma_e)
        
        z2 = np.random.normal(k, sigma_k)

        # random generation
        rng = np.random.uniform(0,1) 
        
        if rng <= 1-theta:
            z = z1
            if evidence >= boundary:
                choicert =  tau_e_trial + rt + tau_m
            else:
                choicert = -tau_e_trial - rt - tau_m
        else:
            z = z2
            if evidence >= boundary:
                choicert =  tau + rt

            else:
                choicert = -tau - rt
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_4b(drift, boundary, beta, mu_tau_e, tau_m, sigma_e, varsigma, theta, number_of_trials, tta_condition, dc=1.0, dt=0.005):

    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while 0 < evidence < boundary:
            evidence += drift * dt + np.sqrt(dt) * dc * np.random.normal()
            n_steps += 1.0
        rt = n_steps * dt
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

    # N200 latency
        z1 = np.random.normal(tau_e_trial, sigma_e)
        
        z2 = np.random.normal(mu_tau_e, np.sqrt(sigma_e**2 + varsigma**2))

        # random generation
        rng = np.random.uniform(0,1) 
        
        if rng <= 1-theta:
            z = z1
            if evidence >= boundary:
                choicert =  tau_e_trial + rt + tau_m
            else:
                choicert = -tau_e_trial - rt - tau_m
        else:
            z = z2
            if evidence >= boundary:
                choicert =  mu_tau_e + rt + tau_m

            else:
                choicert = -mu_tau_e - rt -tau_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_5(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, a_slope, number_of_trials, tta_condition, dc=1.0, dt=0.005):

    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while (evidence > a_slope*n_steps*dt and evidence < (boundary - a_slope*n_steps*dt)):

        # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        
        # visual encoding time for each trial
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

        # N200 latency
        z = np.random.normal(tau_e_trial, sigma)
        
        if evidence >= boundary - a_slope*n_steps*dt:
            choicert =  tau_e_trial + rt + tau_m      
        else:
            choicert = -tau_e_trial - rt - tau_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_6(drift, boundary, beta, mu_tau_e, tau_m, sigma, varsigma, lamda, number_of_trials, tta_condition, dc=1.0, dt=0.005):

    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        k = 3
        delt = -1

    # Simulate a single DM path
        while (evidence > (1-np.exp(-(n_steps*dt/lamda)**k))*(-.5*delt*boundary) and evidence < (boundary - (1-np.exp(-(n_steps*dt/lamda)**k))*(-.5*delt*boundary))):

            # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        
        # visual encoding time for each trial
        tau_e_trial = np.random.normal(mu_tau_e, varsigma)

        # N200 latency
        z = np.random.normal(tau_e_trial, sigma)    
        
        if evidence >= (boundary - (1-np.exp(-(n_steps*dt/lamda)**k))*(-.5*delt*boundary)):
            choicert =  tau_e_trial + rt + tau_m       
        else:
            choicert = -tau_e_trial - rt - tau_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_7(mu_drift, boundary, beta, tau, sigma, eta, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
 
    
    # trial-to-trial drift rate variability
    
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        drift_trial = mu_drift + eta * np.random.normal()
        

    # Simulate a single DM path
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt


        # CPP slope
        cpp = np.random.normal(drift_trial, sigma)


        if evidence >= boundary:
            choicert =  tau + rt
            
        else:
            choicert = -tau - rt
        x = [choicert,cpp]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_8(mu_drift, boundary, tau, sigma, gamma, eta, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    # trial-to-trial drift rate variability
   
    for _ in range(number_of_trials):
        evidence = boundary * 0.5
        n_steps = 0.0
        drift_trial = mu_drift + eta * np.random.normal()
    # Simulate a single DM path
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        
        # CPP slope
        cpp = np.random.normal(gamma*drift_trial, sigma)

        
        if evidence >= boundary:
            choicert =  tau + rt
            
        else:
            choicert = -tau - rt
        x = [choicert,cpp]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_9(drift, boundary, beta, t_e, t_m, sigma_e, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        # N200 latency
        z = np.random.normal(t_e, sigma_e)
        
        if evidence >= boundary:
            choicert =  t_e + rt + t_m
            
        else:
            choicert = -t_e - rt - t_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_10(drift, boundary, beta, t_e, t_m, sigma_e, gamma, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        # N200 latency
        z = np.random.normal(gamma*t_e, sigma_e)
        
        if evidence >= boundary:
            choicert =  t_e + rt + t_m
            
        else:
            choicert = -t_e - rt - t_m

        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_11(drift, boundary, beta, ndt, sigma, gamma, number_of_trials, tta_condition, dc=1.0, dt=0.005,max_steps=2e4):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while (evidence > 0 and evidence < boundary and n_steps < max_steps):

        # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        # decision time
        dt = n_steps * dt

        # CPP slope
        cpp = np.random.normal(gamma*drift, sigma)
        
        if evidence >= boundary:
            choicert =  dt + ndt
            
        elif evidence <= 0:
            choicert = -dt - ndt
        else:
            choicert = np.sign(evidence - boundary*.5)*(dt + ndt)  # Choose closest boundary at max_steps
        x = [choicert,cpp]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_12(drift, boundary, beta, ndt, eta, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        drift_trial = drift + eta * np.random.normal()

    # Simulate a single DM path
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        # decision time
        dt = n_steps * dt

        # CPP slope
        cpp = np.random.normal(drift, eta)
        
        if evidence >= boundary:
            choicert =  dt + ndt
            
        else:
            choicert = -dt - ndt # Choose closest boundary at max_steps
        x = [choicert,cpp]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)

def simulator_13(drift, boundary, beta, t_m, sigma_e, lam, k, number_of_trials, tta_condition, dc=1.0, dt=0.005):
    x_all = []
    x = []
    for _ in range(number_of_trials):
        n_steps = 0.
        evidence = boundary * beta
        while (evidence > 0 and evidence < boundary):

            # DDM equation
            evidence += drift*dt + np.sqrt(dt) * dc * np.random.normal()

            # Increment step
            n_steps += 1.0

        rt = n_steps * dt

        # N200 latency
        z = np.random.normal(k, sigma_e)
        
        if evidence >= boundary:
            choicert =  lam*z + rt + t_m
            
        else:
            choicert = -lam*z - rt - t_m
        x = [choicert,z]
        x_all.append(x)

    x = np.stack(x_all)
    return dict(x=x)




# Meta function for variable trials
# def meta():
#     return {"n_trials": np.random.randint(60, 301)}


# def adopt(prior):
#     def transform_fn(x):
#         # Extract parameters and data
#         params = np.array([x["parameters"][k] for k in prior().keys()], dtype=np.float32)
#         data = x["data"]["x"].astype(np.float32)  # Simulator output (n_trials, 2)
#         n_trials = x["auxiliary"]["n_trials"].astype(np.float32)  # Trial count
        
#         # Standardize data (z-score)
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0, ddof=1)
#         data_std = (data - mean) / (std + 1e-6)  # Avoid division by zero
        
#         # Feature engineering: sqrt(number_of_trials)
#         n_trials_sqrt = np.sqrt(n_trials)
        
#         # Output dictionary for neural networks
#         return {
#             "inference_variables": params,  # Parameters for inference network
#             "summary_variables": data_std,  # Standardized data for summary network
#             "n_trials_sqrt": n_trials_sqrt  # Transformed trial count
#         }
    
#     return Configurator(
#         transformation=transform_fn,
#         prior_is_batched=False  # Matches your non-batched prior
#     )

# # Create generative models
# model_1a = bf.simulation.GenerativeModel(prior=prior_1a, simulator=simulator_1a, simulator_is_batched=False, auxiliary=meta)
# model_1b = bf.simulation.GenerativeModel(prior=prior_1b, simulator=simulator_1b, simulator_is_batched=False, auxiliary=meta)
# model_2 = bf.simulation.GenerativeModel(prior=prior_2, simulator=simulator_2, simulator_is_batched=False, auxiliary=meta)
# model_3 = bf.simulation.GenerativeModel(prior=prior_3, simulator=simulator_3, simulator_is_batched=False, auxiliary=meta)
# model_4a = bf.simulation.GenerativeModel(prior=prior_4a, simulator=simulator_4a, simulator_is_batched=False, auxiliary=meta)
# model_4b = bf.simulation.GenerativeModel(prior=prior_4b, simulator=simulator_4b, simulator_is_batched=False, auxiliary=meta)
# model_5 = bf.simulation.GenerativeModel(prior=prior_5, simulator=simulator_5, simulator_is_batched=False, auxiliary=meta)
# model_6 = bf.simulation.GenerativeModel(prior=prior_6, simulator=simulator_6, simulator_is_batched=False, auxiliary=meta)
# model_7 = bf.simulation.GenerativeModel(prior=prior_7, simulator=simulator_7, simulator_is_batched=False, auxiliary=meta)
# model_8 = bf.simulation.GenerativeModel(prior=prior_8, simulator=simulator_8, simulator_is_batched=False, auxiliary=meta)
# model_9 = bf.simulation.GenerativeModel(prior=prior_9, simulator=simulator_9, simulator_is_batched=False, auxiliary=meta)
# model_10 = bf.simulation.GenerativeModel(prior=prior_10, simulator=simulator_10, simulator_is_batched=False, auxiliary=meta)
# model_11 = bf.simulation.GenerativeModel(prior=prior_11, simulator=simulator_11, simulator_is_batched=False, auxiliary=meta)
# model_12 = bf.simulation.GenerativeModel(prior=prior_12, simulator=simulator_12, simulator_is_batched=False, auxiliary=meta)
# model_13 = bf.simulation.GenerativeModel(prior=prior_13, simulator=simulator_13, simulator_is_batched=False, auxiliary=meta)

# # Dictionary of all models
# all_models = {
#     'model_1a': [model_1a, adopt(prior_1a())],
#     'model_1b': [model_1b, adopt(prior_1b())],
#     'model_2': [model_2, adopt(prior_2())],
#     'model_3': [model_3, adopt(prior_3())],
#     'model_4a': [model_4a, adopt(prior_4a())],
#     'model_4b': [model_4b, adopt(prior_4b())],
#     'model_5': [model_5, adopt(prior_5())],
#     'model_6': [model_6, adopt(prior_6())],
#     'model_7': [model_7, adopt(prior_7())],
#     'model_8': [model_8, adopt(prior_8())],
#     'model_9': [model_9, adopt(prior_9())],
#     'model_10': [model_10, adopt(prior_10())],
#     'model_11': [model_11, adopt(prior_11())],
#     'model_12': [model_12, adopt(prior_12())],
#     'model_13': [model_13, adopt(prior_13())]
# }


def meta():
    return {
        "number_of_trials": RNG.integers(60, 300),
        "tta_condition": RNG.choice(CONDITIONS),
    }

def adopt(p):
    adapter = (
        bf.Adapter()
        .broadcast("number_of_trials", to="x")  # Align trial count with data
        .as_set("x")  # Treat trials as exchangeable (permits variable length)
        .standardize("x")  # Normalize data
        .sqrt("number_of_trials")  # Feature engineering
        .broadcast("tta_condition", to="condition")  # Expose condition flag
        .convert_dtype("float64", "float32")
    # PyTorch compatibility
        .concatenate(list(p.keys()), into="inference_variables")
        .rename("x", "summary_variables")  # Rename data to "summary_variables" [[3]]
        .rename("condition", "condition_variables")
    )
    return adapter
 

model_1a = bf.simulators.make_simulator([prior_1a, simulator_1a], meta_fn=meta)
model_1b = bf.simulators.make_simulator([prior_1b, simulator_1b], meta_fn=meta)
model_1c = bf.simulators.make_simulator([prior_1c, simulator_1c], meta_fn=meta)
model_2 = bf.simulators.make_simulator([prior_2, simulator_2], meta_fn=meta)
model_3 = bf.simulators.make_simulator([prior_3, simulator_3], meta_fn=meta)
model_4a = bf.simulators.make_simulator([prior_4a, simulator_4a], meta_fn=meta)
model_4b = bf.simulators.make_simulator([prior_4b, simulator_4b], meta_fn=meta)
model_5 = bf.simulators.make_simulator([prior_5, simulator_5], meta_fn=meta)
model_6 = bf.simulators.make_simulator([prior_6, simulator_6], meta_fn=meta)
model_7 = bf.simulators.make_simulator([prior_7, simulator_7], meta_fn=meta)
model_8 = bf.simulators.make_simulator([prior_8, simulator_8], meta_fn=meta)
model_9 = bf.simulators.make_simulator([prior_9, simulator_9], meta_fn=meta)
model_10 = bf.simulators.make_simulator([prior_10, simulator_10], meta_fn=meta)
model_11 = bf.simulators.make_simulator([prior_11, simulator_11], meta_fn=meta)
model_12 = bf.simulators.make_simulator([prior_12, simulator_12], meta_fn=meta)
model_13 = bf.simulators.make_simulator([prior_13, simulator_13], meta_fn=meta)


all_models = {
    # 'model_1a': [model_1a,adopt(prior_1a())],
    # 'model_1b': [model_1b,adopt(prior_1b())],
    # 'model_1c': [model_1c,adopt(prior_1c())],
    # 'model_2': [model_2,adopt(prior_2())],
    # 'model_3': [model_3,adopt(prior_3())],
    # 'model_4a': [model_4a,adopt(prior_4a())],
    # 'model_4b': [model_4b,adopt(prior_4b())],
    # 'model_5': [model_5,adopt(prior_5())],
    # 'model_6': [model_6,adopt(prior_6())],
    # 'model_7': [model_7,adopt(prior_7())],
    'model_8': [model_8,adopt(prior_8())],
    'model_9': [model_9,adopt(prior_9())],
    'model_10': [model_10,adopt(prior_10())],
    'model_11': [model_11,adopt(prior_11())],
    'model_12': [model_12,adopt(prior_12())],
    'model_13': [model_13,adopt(prior_13())]
}

