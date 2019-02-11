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
from bisect import bisect_left, bisect_right



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
    #taking only the timestamps before the reference 
    ind = bisect_right(tmpdata[0],timestamp)
    ind_subset = np.concatenate((np.ones(ind,bool), np.zeros(len(tmpdata[0])-(ind),bool)))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    #removing all the timestamps that will generate values below minv
    min_timestamp = timestamp + timecoeff*np.log(minv) #timestamps<min_timestamp WILL BE DISCARDED 
    ind = bisect_left(tmpdata[0],min_timestamp)
    ind_subset = np.concatenate((np.zeros(ind,bool), np.ones(len(tmpdata[0])-(ind),bool)))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)
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
# ydim,xdim : dimensions of the timesurface
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
def Time_Surface_event(xdim, ydim, event, timecoeff, dataset, num_polarities, minv=0.1, verbose=False):
    tmpdata = [dataset[0].copy(), dataset[1].copy(), dataset[2].copy()]
    #centering the dataset around the event
    x0 = event[1][0]
    y0 = event[1][1]
    tmpdata[1][:,0] = tmpdata[1][:,0] - x0
    tmpdata[1][:,1] = tmpdata[1][:,1] - y0
    #taking only the timestamps before the reference 
    timestamp = event[0]
    ind = bisect_right(tmpdata[0],timestamp)
    ind_subset = np.concatenate((np.ones(ind,bool), np.zeros(len(tmpdata[0])-(ind),bool)))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    #removing all the timestamps that will generate values below minv
    min_timestamp = timestamp + timecoeff*np.log(minv) #timestamps<min_timestamp WILL BE DISCARDED 
    ind = bisect_left(tmpdata[0],min_timestamp)
    ind_subset = np.concatenate((np.zeros(ind,bool), np.ones(len(tmpdata[0])-(ind),bool)))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]     
    #removing all events outside the region of interest defined as the xdim*ydim
    # centered on the event
    border = [np.floor(xdim/2),np.floor(ydim/2)]
    ind_subset = ((tmpdata[1][:,0]>=-border[0]) & (tmpdata[1][:,0]<=border[0]) &
        (tmpdata[1][:,1]>=-border[1]) & (tmpdata[1][:,1]<=border[1]))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]   
    tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)
    #now i need to build a matrix that will represents my surface, i will take 
    #only the highest value for each x and y as the other ar less informative
    #and we want each layer be dependant on the timecoeff of the timesurface
    #Note that exp is monotone and the timestamps are ordered, thus the last 
    #values of the dataset will be the lowest too
    tsurface = np.zeros([ydim,xdim*num_polarities])
    offs = [np.int(np.floor(xdim/2)),np.int(np.floor(ydim/2))]
    for i in range(len(tsurface_array)):
        tsurface[tmpdata[1][i][1]+offs[1],tmpdata[1][i][0]+offs[0]+xdim*tmpdata[2][i]]=tsurface_array[i]  
    #plot graphs if verbose is set "True"
    if (verbose==True):
        plt.figure()
        sns.heatmap(tsurface)

    return tsurface

