# IMPORT LIBRARIES

import random as rn
import numpy as np
import math as mt

import matplotlib.pyplot as plt
from matplotlib import cm

import pylab as pl
import sys
from matplotlib.pyplot import cm
import glob
import pandas as pd

from astropy.table import Table
import torch
import torch.nn as nn


# Normalize Data

def normalize(data):
    norm_min_preMS = np.load('Aux/norm_min_preMS.npy')
    norm_max_preMS = np.load('Aux/norm_max_preMS.npy')
    norm_min_postMS = np.load('Aux/norm_min_postMS.npy')
    norm_max_postMS = np.load('Aux/norm_max_postMS.npy')
    x_data_preMS = torch.from_numpy((data[:,:2]-norm_min_preMS[:2])/(norm_max_preMS[:2]-norm_min_preMS[:2])).float()
    x_data_postMS = torch.from_numpy((data[:,:2]-norm_min_postMS[:2])/(norm_max_postMS[:2]-norm_min_postMS[:2])).float()

    return x_data_preMS, x_data_postMS


# Normalize Data

def unnormalize(y_pre, y_post):
    norm_min_preMS = np.load('Aux/norm_min_preMS.npy')
    norm_max_preMS = np.load('Aux/norm_max_preMS.npy')
    norm_min_postMS = np.load('Aux/norm_min_postMS.npy')
    norm_max_postMS = np.load('Aux/norm_max_postMS.npy')
    y_pre_un = y_pre*(np.array(norm_max_preMS[2:])-np.array(norm_min_preMS[2:]))+ np.array(norm_min_preMS[2:])
    y_post_un = y_post*(np.array(norm_max_postMS[2:])-np.array(norm_min_postMS[2:]))+ np.array(norm_min_postMS[2:])

    return y_pre_un, y_post_un


# Build Model

class NN(nn.Module):
    def __init__(self, D_in, D_out, num_layers, num_nodes, activation):
        super(NN, self).__init__()
        
        # Specify list of layer sizes 
        sizes = [D_in] + [num_nodes] * num_layers + [D_out]
        in_sizes, out_sizes = sizes[:-1], sizes[1:]
        
        # Construct linear layers
        self.linears = nn.ModuleList()
        for n_in, n_out in zip(in_sizes, out_sizes):
            self.linears.append(nn.Linear(n_in, n_out))
        
        # Specify activation function 
        self.activation = activation
        
    def forward(self, x):
        
        for l in self.linears[:-1]:
            x = self.activation(l(x))
        x = self.linears[-1](x)
        
        return x


# Predict

def predict(X, n_mod=20, TL=None):
    x_data_preMS, x_data_postMS = normalize(X)
    
    D_in = 2
    D_out = 2
    num_layers = 10
    num_nodes =50
    activation = nn.ReLU()    
    net = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_preMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    net_postMS = NN(D_in, D_out, num_layers, num_nodes, activation)
    num_models=n_mod
    if num_models > 20: sys.exit('Number of models should not exceed 20')
    
    pre_model = 'Models/baseline/mist_baseline_preMS{}'
    post_model = 'Models/baseline/mist_baseline_postMS{}'
    
    if TL=='DH':
        pre_model = 'Models/DH/mist_DH_preMS{}'
        post_model = 'Models/DH/mist_DH_postMS{}'
    
    net_preMS.load_state_dict(torch.load(pre_model.format(0)), strict=False)
    net_postMS.load_state_dict(torch.load(post_model.format(0)), strict=False)
    y_pred_preMS = torch.unsqueeze(net_preMS(x_data_preMS),0).detach().numpy()
    y_pred_postMS = torch.unsqueeze(net_postMS(x_data_postMS),0).detach().numpy()
    #y_pred_preMS = []
    #y_pred_postMS = []
    for i in range(1,n_mod):
        net_preMS.load_state_dict(torch.load(pre_model.format(i)), strict=False)
        net_postMS.load_state_dict(torch.load(post_model.format(i)), strict=False)
        y_pred_preMS = np.append(y_pred_preMS, torch.unsqueeze(net_preMS(x_data_preMS),0).detach().numpy(), axis=0)
        y_pred_postMS = np.append(y_pred_postMS, torch.unsqueeze(net_postMS(x_data_postMS),0).detach().numpy(), axis=0)
    
    y_pred_preMS_un, y_pred_postMS_un = unnormalize(y_pred_preMS, y_pred_postMS)

    return y_pred_preMS_un, y_pred_postMS_un


# Posterior Statistics for Each Model

def stats(y_pred):
    y_mean = np.mean(y_pred, 0)
    y_std = np.std(y_pred, 0)

    return y_mean, y_std


# Mixture of Models

def pis(y_mean_preMS, y_mean_postMS):
    boundary = pd.read_pickle('Aux/boundary')
    edge = pd.read_pickle('Aux/edge')
    
    mass_pred_pre = 10**(y_mean_preMS[:,1])
    mass_pred_post = 10**(y_mean_postMS[:,1])
    idx_bd_pre = np.argmin(abs(mass_pred_pre[:,np.newaxis]-np.array(boundary['star_mass'])),axis=1)
    idx_bd_post = np.argmin(abs(mass_pred_post[:,np.newaxis]-np.array(boundary['star_mass'])),axis=1)
    idx_ed_pre = np.argmin(abs(mass_pred_pre[:,np.newaxis]-np.array(edge['star_mass'])),axis=1)
    idx_ed_post = np.argmin(abs(mass_pred_post[:,np.newaxis]-np.array(edge['star_mass'])),axis=1)
    
    m=np.array([0.02,0.05,0.1,0.4, 0.6,0.8])
    Chi_pre= np.clip(((mass_pred_pre-0.08)/(0.42)+1).astype(int)+0.3, 0.3, 2.3)
    Chi_post = np.clip(((mass_pred_post-0.08)/(0.42)+1).astype(int)+0.3, 0.3, 2.3)
    w_pre = mass_pred_pre**(-Chi_pre) * np.array(boundary['star_age'].iloc[idx_bd_pre])
    w_post = mass_pred_post**(-Chi_post) * (np.array(edge['star_age'].iloc[idx_ed_post])-np.array(boundary['star_age'].iloc[idx_bd_post]))
    w_tot = w_pre + w_post            
    w_pre_norm = w_pre/w_tot
    w_post_norm = w_post/w_tot

    return w_pre_norm, w_post_norm


def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(sig*mt.sqrt(2*mt.pi))

    
def Gaussian_posteriors(y_mean_preMS, y_mean_postMS, y_std_preMS, y_std_postMS, pi_pre, pi_post, n_obs, num_x_points=500):
    x_values=np.linspace(0, 1, num_x_points)
    x= (x_values[:,np.newaxis]*np.ones(n_obs))[:,:,np.newaxis]*np.ones(2)*[9,6]+[4,-2]
    pre=(np.ones(num_x_points)[:,np.newaxis]*pi_pre[np.newaxis,:])[:,:,np.newaxis]*np.ones(2)
    post=(np.ones(num_x_points)[:,np.newaxis]*pi_post[np.newaxis,:])[:,:,np.newaxis]*np.ones(2)
    y_gaussian_posteriors = pre* gaussian(x, y_mean_preMS, y_std_preMS)+ post* gaussian(x, y_mean_postMS, y_std_postMS)
    return y_gaussian_posteriors


# Posterior Probability Distributions

def posteriors(y_pre , y_post, pi_pre, pi_post, n_mod =20):
    y_posteriors = (np.ones((n_mod, 1))*pi_pre[np.newaxis])[:,:,np.newaxis]*np.ones(2) * y_pre+ (np.ones((n_mod, 1))*pi_post[np.newaxis])[:,:,np.newaxis]*np.ones(2) * y_post 
    return y_posteriors


# Plot

def plot_multiple_posteriors(y_posteriors, obs_array, num_models=20, dotsize = 2):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15), sharex= True)
    for i in obs_array:
        ax[0].scatter(np.ones(num_models)*i, y_posteriors[:,i,0], s=dotsize)
        ax[1].scatter(np.ones(num_models)*i, y_posteriors[:,i,1], s=dotsize)
    ax[1].set_xlabel('Observation id',fontsize=30)
    ax[0].set_ylabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_ylabel('$\log(mass)$ [$M_{\odot}$]', fontsize=30)
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)

    return ax

def plot_posterior(y_post,obs_id, num_models=20):
    fig, ax = plt.subplots(2, 1, figsize=(20, 15), sharex= True)
    ax[0].scatter(np.arange(num_models)+1, y_post[:,obs_id,0], s=30)
    ax[1].scatter(np.arange(num_models)+1, y_post[:,obs_id,1], s=30)
    ax[0].set_xticks([i for i in range(1,21)])
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[1].set_xlabel('model number', fontsize=30)
    ax[0].set_ylabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_ylabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    
    return ax

def plot_posterior_hist(y_post, obs_id, bins=15, log_age_range=[5,12], log_mass_range=[-1.5,1.5]):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(30,10), sharex = False)
    ax[0].hist(y_post[:,obs_id,0], bins=15)
    ax[1].hist(y_post[:,obs_id,1], bins=15)   
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_xlabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    ax[0].set_ylabel('number of predictions', fontsize=30)
    
    return ax

def plot_gaussian_posteriors(y_gaussian_posteriors, obs_id, log_age_range=[4,12], log_mass_range=[-1.5,1.5]):
    x_age = np.linspace(log_age_range[0], log_age_range[1], y_gaussian_posteriors.shape[0])
    x_mass = np.linspace(log_mass_range[0], log_mass_range[1], y_gaussian_posteriors.shape[0])
    y = y_gaussian_posteriors[:,obs_id,:]
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(30,10), sharex = False)
    ax[0].plot(x_age,0.01+y[:,0])
    ax[1].plot(x_mass,0.01+y[:,1], 'm')     
    ax[0].set_xlim(log_age_range[0],log_age_range[1])
    ax[1].set_xlim(log_mass_range[0],log_mass_range[1])
    ax[0].tick_params(labelsize=25)
    ax[1].tick_params(labelsize=25)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('$\log(age \ [yrs])$', fontsize=30)
    ax[1].set_xlabel('$\log(mass \ [M_{\odot}])$', fontsize=30)
    ax[0].set_ylabel('Probability', fontsize=30)
    ax[1].text(log_mass_range[1]-(log_mass_range[1]-log_mass_range[0])/3,20, 'obs_id={}'.format(obs_id), fontsize=30)
    
    return ax
