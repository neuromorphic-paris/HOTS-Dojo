#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:10:06 2018

@author: marcorax

Function used to generate the Time surfaces (modified to represent rate, event mixed 
information), please note that these functions expect the dataset to be ordered from the 
lower timestamp to the highest
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from bisect import bisect_left, bisect_right



#TODO add rate maybe
# =============================================================================
def Time_Surface_all(xdim, ydim, timestamp, timecoeff, dataset, num_polarities, minv=0.1, verbose=False):
    """
    Time_Surface_all: function that computes the Time_surface of an entire dataset,
    starting from a selected timestamp.
    Arguments : 
        ydim,xdim (int) : dimensions of the timesurface
        timestamp (int) : original time stamp respect to which the timesurface is computed
        timecoeff (float) : the time coeff expressing the time decay
        dataset (nested lists) : dataset containing the events, a list where dataset[0] contains the 
                   timestamps as microseconds, and dataset[1] contains [x,y] pixel positions 
        num_polarities (int) : total number of labels or polarities of the time surface 
        minv (float) : hardtreshold for the time surface, values smaller than minv will be
               removed from the result 
       verbose (bool) : if True, a graph of the surface will be plotted 
       
   Returns :
       tsurface (2D numpy matrix) : matrix of size num_polarities*xdim*ydim 
    """
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


# =============================================================================
def Time_Surface_event(xdim, ydim, event, timecoeff, dataset, num_polarities, minv=0.1):
    """
    Time_Surface_event: function that computes the Time_surface around a single event
    (with rate information)
    
    Arguments : 
        ydim,xdim (int) : dimensions of the timesurface
        event (nested lists) : single event defined as [timestamp, [x, y]]. It is the reference event 
                for the time surface building
        timecoeff (float) : the time coeff expressing the time decay
        dataset (nested lists) : dataset containing the events, a list where dataset[0] contains the 
                   timestamps as microseconds, and dataset[1] contains [x,y] pixel positions 
        num_polarities (int) : total number of labels or polarities of the time surface 
        minv (float) : hardtreshold for the time surface, values smaller than minv will be
               removed from the result 
    Return : 
        tsurface (1D numpy array) : array of size num_polarities*xdim*ydim 
    
    """
    # If the spiking activity has rates, tmpdata will be longer
    if len(dataset) == 4:
        tmpdata = [dataset[0].copy(), dataset[1].copy(), dataset[2].copy(), dataset[3].copy()]
    else:
        tmpdata = [dataset[0].copy(), dataset[1].copy(), dataset[2].copy()]
    # centering the dataset around the event
    x0 = event[1][0]
    y0 = event[1][1]
    tmpdata[1][:,0] = tmpdata[1][:,0] - x0
    tmpdata[1][:,1] = tmpdata[1][:,1] - y0
    # taking only the timestamps before the reference 
    timestamp = event[0]
    ind = bisect_right(tmpdata[0],timestamp)
    ind_subset = np.concatenate((np.ones(ind,bool), np.zeros(len(tmpdata[0])-(ind),bool)))
    if len(dataset) == 4:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset], tmpdata[3][ind_subset]]
    else:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]       
    # removing all the timestamps that will generate values below minv
    min_timestamp = timestamp + timecoeff*np.log(minv) #timestamps<min_timestamp WILL BE DISCARDED 
    ind = bisect_left(tmpdata[0],min_timestamp)
    ind_subset = np.concatenate((np.zeros(ind,bool), np.ones(len(tmpdata[0])-(ind),bool)))
    if len(dataset) == 4:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset], tmpdata[3][ind_subset]]
    else:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]] 
    # removing all events outside the region of interest defined as the xdim*ydim
    # centered on the event
    border = [np.floor(xdim/2),np.floor(ydim/2)]
    ind_subset = ((tmpdata[1][:,0]>=-border[0]) & (tmpdata[1][:,0]<=border[0]) &
        (tmpdata[1][:,1]>=-border[1]) & (tmpdata[1][:,1]<=border[1]))
    if len(dataset) == 4:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset], tmpdata[3][ind_subset]]
        tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)*tmpdata[3]
    else:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]     
        tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)

    # now i need to build a single array that will represents my surface, i will take 
    # only the highest value for each x and y as the other ar less informative
    # and we want each layer be dependant on the timecoeff of the timesurface
    # Note that exp is monotone and the timestamps are ordered, thus the last 
    # values of the dataset will be the lowest too
    tsurface = np.zeros(ydim*xdim*num_polarities)
    offs = [np.int(np.floor(xdim/2)),np.int(np.floor(ydim/2))]
    # The way this array is encoded is with the first xdim*ydim values representing
    # the time surface at polarity 0 and so on.
    # Eeach linear index is encoding also per the spartial position of the event
    # Being (omitting polarity) index = x_coordinate + y_coordinate*xdim
    for i in range(len(tsurface_array)):
        tsurface[(tmpdata[1][i][1]+offs[1]+(tmpdata[1][i][0]+offs[0])*xdim)+(tmpdata[2][i]*xdim*ydim)]=tsurface_array[i]  
    
    return tsurface

def Reverse_Time_Surface_event(xdim, ydim, event, tsurface, num_polarities):
    """
    Function computing the events composing a single timesurface, as the opposite 
    of the function Time_Surface_event, this function serve the pourpose to decode 
    back the surfaces built by a devoder in the events that generated them
    
    Arguments  : 
        ydim,xdim (int) : dimensions of the timesurface
        event (nested lists) : single event defined as [timestamp, [x, y]]. It is the reference event
                               used for the time surface building
        tsurface (1D numpy array) : array of size num_polarities*xdim*ydim         
        num_polarities (int) : total number of labels or polarities of the time surface 
    """
    ref_timestamp=event[0]
    x0=event[1][0]
    y0=event[1][1]
    timestamps=ref_timestamp*np.ones(len(tsurface))
    polarities=np.array([pol for pol in range(num_polarities) for ind in range(xdim*ydim)])    
    positions=[[x+x0,y+y0] for pol in range(num_polarities) for x in range(-(xdim-1)//2,(xdim+1)//2) for y in range(-(ydim-1)//2,(ydim+1)//2)]
    rates=[tsurface[pos+(pol*xdim*ydim)] for pol in range(num_polarities) for pos in range(xdim*ydim)]
    events=[timestamps, positions, polarities, rates]
    
    return events

def Reverse_Time_Surface_event_no_rate(xdim, ydim, event, tsurface, timecoeff, num_polarities):
    """
    Function computing the events composing a single timesurface, as the opposite 
    of the function Time_Surface_event, this function serve the pourpose to decode 
    back the surfaces built by a devoder in the events that generated them.
    This function doesn't expect rate
    
    Arguments  : 
        ydim,xdim (int) : dimensions of the timesurface
        event (nested lists) : single event defined as [timestamp, [x, y]]. It is the reference event
                               used for the time surface building
        tsurface (1D numpy array) : array of size num_polarities*xdim*ydim  
        timecoeff (float) : the coefficient used to build the timesurfaces for 
                            this layer
        num_polarities (int) : total number of labels or polarities of the time surface 
    """
    ref_timestamp=event[0]
    x0=event[1][0]
    y0=event[1][1]
    polarities=np.array([pol for pol in range(num_polarities) for ind in range(xdim*ydim)])    
    positions=[[x+x0,y+y0] for pol in range(num_polarities) for x in range(-(xdim-1)//2,(xdim+1)//2) for y in range(-(ydim-1)//2,(ydim+1)//2)]
    timestamps=ref_timestamp+timecoeff*np.log([tsurface[pos+(pol*xdim*ydim)] for pol in range(num_polarities) for pos in range(xdim*ydim)])
    events=[timestamps, positions, polarities]
    
    return events