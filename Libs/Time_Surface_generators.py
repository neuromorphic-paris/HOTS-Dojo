#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:10:06 2018

@author: marcorax

Function used to generate the Time surfaces, 
please note that these functions expect the dataset to be ordered from the 
lower timestamp to the highest
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



## Time_Surface_all: function that computes the Time_surface of an entire dataset,
#  starting from a selected timestamp.
# =============================================================================
# ydim,xdim : dimensions of the timesurface
# timestamp : original time stamp respect to which the timesurface is computed
# timecoeff : the time coeff expressing the time decay
# dataset : dataset containing the events, a list where dataset[0] contains the 
#           timestamps as microseconds, and dataset[1] contains [x,y] pixel positions 
# num_polarities : total number of labels or polarities of the time surface 
# minv : hardtreshold for the time surface, values smaller than minv will be
#        removed from the result 
# verbose : bool, if True, a graph of the surface will be plotted 
#
# tsurface : matrix of size num_polarities*xdim*ydim 
# =============================================================================
def Time_Surface_all(xdim, ydim, timestamp, timecoeff, dataset, num_polarities, minv=0.1, verbose=False):
    tmpdata = [dataset[0].copy(), dataset[1].copy(), dataset[2].copy()]
    #taking only the timestamps after the reference 
    ind_subset = tmpdata[0]>=timestamp
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    tsurface_array = np.exp(-(tmpdata[0]-timestamp)/timecoeff)
    #removing all values below minv 
    ind_subset = tsurface_array>minv
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]] 
    tsurface_array = tsurface_array[ind_subset]
    #now i need to build a matrix that will represents my surface, i will take 
    #only the highest value for each x and y as the other ar less informative
    #and we want each layer be dependant on the timecoeff of the timesurface
    #Note that exp is monotone and the timestamps are ordered, thus the last 
    #values of the dataset will be the lowest too
    tsurface = np.zeros([ydim,xdim*num_polarities])
    for i in range(len(tsurface_array)):
        if tsurface[tmpdata[1][i][1],tmpdata[1][i][0]+xdim*tmpdata[2][i]]==0:
            tsurface[tmpdata[1][i][1],tmpdata[1][i][0]+xdim*tmpdata[2][i]]=tsurface_array[i]  
    #plot graphs if verbose is set "True"
    if (verbose==True):
        plt.figure()
        sns.heatmap(tsurface)

    return tsurface

## Time_Surface_event: function that computes the Time_surface around a single event
# =============================================================================
# ldim : linear dim of the timesurface, only square surfaces are considered
# event : single event defined as [timestamp, [x, y]]. It is the reference even 
#         for the time surface
# timecoeff : the time coeff expressing the time decay
# dataset : dataset containing the events, a list where dataset[0] contains the 
#           timestamps as microseconds, and dataset[1] contains [x,y] pixel positions 
# num_polarities : total number of labels or polarities of the time surface 
# minv : hardtreshold for the time surface, values smaller than minv will be
#        removed from the result 
# verbose : bool, if True, a graph of the surface will be plotted 
#
# tsurface : matrix returned with ldim as linear dimension
# =============================================================================
def Time_Surface_event(ldim, event, timecoeff, dataset, num_polarities, minv=0.1, verbose=False):
    tmpdata = [dataset[0].copy(), dataset[1].copy(), dataset[2].copy()]
    #centering the dataset around the event
    x0 = event[1][0]
    y0 = event[1][1]
    tmpdata[1][:,0] = tmpdata[1][:,0] - x0
    tmpdata[1][:,1] = tmpdata[1][:,1] - y0
    #removing all events outside the region of interest defined as the ldim²
    #square centered on the event
    border = np.floor(ldim/2)
    ind_subset = ((tmpdata[1][:,0]>=-border) & (tmpdata[1][:,0]<=border) &
        (tmpdata[1][:,1]>=-border) & (tmpdata[1][:,1]<=border))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]   
    #taking only the timestamps after the reference 
    timestamp = event[0]
    ind_subset = tmpdata[0]>=timestamp
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    tsurface_array = np.exp(-(tmpdata[0]-timestamp)/timecoeff)
    #removing all values below minv 
    ind_subset = tsurface_array>minv
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]] 
    tsurface_array = tsurface_array[ind_subset]
    #now i need to build a matrix that will represents my surface, i will take 
    #only the highest value for each x and y as the other ar less informative
    #and we want each layer be dependant on the timecoeff of the timesurface
    #Note that exp is monotone and the timestamps are ordered, thus the last 
    #values of the dataset will be the lowest too
    tsurface = np.zeros([ldim,ldim*num_polarities])
    offs = np.int(np.floor(ldim/2))
    for i in range(len(tsurface_array)):
        if tsurface[tmpdata[1][i][1]+offs,tmpdata[1][i][0]+offs+ldim*tmpdata[2][i]]==0:
            tsurface[tmpdata[1][i][1]+offs,tmpdata[1][i][0]+offs+ldim*tmpdata[2][i]]=tsurface_array[i]  
    #plot graphs if verbose is set "True"
    if (verbose==True):
        plt.figure()
        sns.heatmap(tsurface)

    return tsurface

## generate_sparse_HOTS_net : a function for generating a randomic
# sparse HOTS net, with sets of random basis and a_j parameters
# =============================================================================
# nbasis_list : A list containing the number of basis for each layer
# ldim_list : A list containing the linear dimension of the basis for each layer
# ldim_list and nbasis_list need to have same lenght
# seed : The seed used for generating the values, if set to 0 it will be disabled 
# =============================================================================
def generate_sparse_HOTS_net(nbasis_list, ldim_list, seed=0):
    Basis_set = []
    Basis_set_temp = []
    aj_set = []
    aj_set_temp = []
    #setting the seed
    rng = np.random.RandomState()         
    if (seed!=0):
        rng.seed(seed)
    for i, nbasis in enumerate(nbasis_list):
        for j in range(nbasis):
            Basis_set_temp.append(rng.rand(ldim_list[i],ldim_list[i]))
            #aj are set to be between -1 and 1
            aj_set_temp.append((rng.rand()-0.5)*2)
        Basis_set.append(Basis_set_temp.copy())
        aj_set.append(aj_set_temp.copy())
    return Basis_set, aj_set