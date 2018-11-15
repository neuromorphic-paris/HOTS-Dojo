#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax

This file contains the HOTS_Sparse_Net class and other complementary functions
that are used inside HOTS with have no pratical use outside the context of this class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

To better understand the math behind classification an optimization please take a look
to the original Sparse Hots paper
 
"""
import numpy as np 
from scipy import optimize
from scipy.spatial import distance 
import matplotlib.pyplot as plt
import seaborn as sns
from Libs.Time_Surface_generators import Time_Surface_event
 


# Error functions and update basis (for online algorithms)
# This functions are used for learn_offline_mean too because the mini batch are 
# compressed in a single mean surface and processed similiarly as in the online 
# algorithm (one mean surface per time)
# =============================================================================
# The notation is the same as rapresented in the paper:
# a_j : activations/coefficients aj of the net
# Phi_j: the basis of each layer
# lam_aj: Lambda coefficient for aj sparsification
# lam_phi: Lambda coefficient for Phi_j regularization
# S: Input time surface
# S_tilde: reconstructed time surface
# =============================================================================
def error_func(a_j, S, Phi_j, lam_aj):
    S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
    return np.sum((S-S_tilde)**2) + lam_aj*np.sum(np.abs(a_j))

def error_func_deriv_a_j(a_j, S, Phi_j, lam_aj):
    S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
    residue = S-S_tilde
    return [-2*np.sum(np.multiply(residue,a)) for a in zip(Phi_j)] + lam_aj*np.sign(a_j)
    
def update_basis_online(Phi_j, a_j, eta, S, lam_phi, max_steps, precision):
    n_steps = 0 
    previous_err =  error_func(a_j, S, Phi_j, 0)
    err = previous_err + precision + 1
    while (abs(err-previous_err)>precision) & (n_steps<max_steps):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
        residue = S-S_tilde
        for i in range(len(Phi_j)):
            Phi_j[i] = Phi_j[i] + residue*eta*a_j[i] -eta*lam_phi*Phi_j[i]
        previous_err = err
        err =  error_func(a_j, S, Phi_j, 0)
        n_steps += 1

# impose basis values between[1 0] rather than regularize, useful to reduce 
# parameters at expenses of precision
def update_basis_online_hard_treshold(Phi_j, a_j, eta, S, lam_phi, max_steps, precision):
    n_steps = 0 
    previous_err =  error_func(a_j, S, Phi_j, 0)
    err = previous_err + precision + 1
    while (abs(err-previous_err)>precision) & (n_steps<max_steps):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j)])
        residue = S-S_tilde
        for i in range(len(Phi_j)):
            newbase = Phi_j[i] + residue*eta*a_j[i] -eta*lam_phi*Phi_j[i]
            newbase[newbase>=1] = 1
            newbase[newbase<=0] = 0
            Phi_j[i] = newbase
        previous_err = err
        err =  error_func(a_j, S, Phi_j, 0)
        n_steps += 1
        
# Error functions (for offline algorithbatch_surfacesms)
# These functions are developed to work on multiple data per time (a whole batch)
# Notice that Error functions for the basis (here implemented as matrices) are 
# generally meant to work with scipy.optimize.minimize therefore reshaping
# has to be performed
# =============================================================================
# The notation is the same as rapresented in the paper:
# a_j : activations/coefficients aj of the net
# Phi_j: the basis of each layer
# Phi_j_dim: 
# lam_phi: Lambda coefficient for Phi_j regularization
# S_list: Input time surfaces used for learning 
# S_tilde: reconstructed time surface
# =============================================================================
def error_func_phi_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi):
    #scipy optimize minimze will input a monodimensional array
    #for the Phi_j, thus i need to reconstruct the matrixes 
    Phi_j = np.reshape(Phi_j, (Phi_j_num, Phi_j_dim_y, Phi_j_dim_x))    
    err = 0
    for k in range(len(S_list)):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j[k])])
        err += np.sum((S_list[k]-S_tilde)**2) 
    result = err + lam_phi*np.sum(Phi_j**2)
    #as scipy performs operations on monodimensional arrays i need to flatten
    #the result
    return result.flatten()

def error_func_phi_grad_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi):
    #scipy optimize minimze will input a monodimensional array
    #for the Phi_j, thus i need to reconstruct the matrixes  
    Phi_j = np.reshape(Phi_j, (Phi_j_num, Phi_j_dim_y, Phi_j_dim_x))    
    #initialize the gradient result
    grad = []
    for i in range(len(Phi_j)):
        grad.append(np.zeros(Phi_j[0].shape))
    grad = np.array(grad)
    for k in range(len(S_list)):
        S_tilde = sum([a*b for a,b in zip(Phi_j,a_j[k])])
        residue = S_list[k]-S_tilde
        grad += [a*residue for a in zip(a_j[k])]
    result = -2*grad + 2*lam_phi*Phi_j
    #as scipy performs operations on monodimensional arrays i need to flatten
    #the result
    return result.flatten()

# function that update basis with conjugate gradient method
def update_basis_offline_CG(Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi):
    # Error functions and derivatives for gradient descent
    Err = lambda Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi: error_func_phi_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi)
    dErrdPhi = lambda Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, a_j, S_list, lam_phi: error_func_phi_grad_full_batch(a_j, Phi_j, Phi_j_dim_x, Phi_j_dim_y, Phi_j_num, S_list, lam_phi)
    res = optimize.minimize(fun=Err, x0=Phi_j.flatten(), args=(Phi_j_dim_x, Phi_j_dim_y,
                             Phi_j_num, a_j, S_list, lam_phi),
                             method="CG", jac=dErrdPhi)
    #Phi_j = res.x.reshape(..) won't work because the returned Phi_j is not the 
    #reference to the original one, therefore it cannot update implicitly self.basis[0]
    return res

# Function for computing events out of net activations, the linear relation
# between the activity value a_j and the obtained timestamp is mediated through 
# a delay_coefficient (namely alpha in the paper)
# =============================================================================
# activations : An array containing all the activations of a sublayer given a
#               batch 
# events : the reference events that generated the input timesurfaces
# delay_coeff : the delay coefficient called alpha in the original paper
#    
# It returns on and off events with polarities defined as the index of the base 
# generating them, do not confuse them with the on off label. 
# The two matrices are : on_events and off_events. They are composed of 3 columns:
# The first column is the timestamp array, the second column is a [2*N] array,
# with the spatial positions of each events. Lastly the third column stores
# the previously mentioned polarities 
# =============================================================================    
def events_from_activations(activations, events, delay_coeff):
    timestamps = events[0]
    out_timestamps_on = []
    out_positions_on = []
    out_polarities_on = []
    out_timestamps_off = []
    out_positions_off = []
    out_polarities_off = []

    for i,t in enumerate(timestamps):
        for j,a_j in enumerate(activations[i]):
            tout = t + delay_coeff*(1-abs(a_j))
            if (a_j > 0):
                out_timestamps_on.append(tout)
                out_positions_on.append(events[1][i])
                out_polarities_on.append(j)
            if (a_j < 0):
                out_timestamps_off.append(tout)
                out_positions_off.append(events[1][i])
                out_polarities_off.append(j)

    # The output data need to be sorted like the original dataset in respect to
    # the timestamp, because the time surface functions are expecting it to be sorted.
    out_timestamps_on = np.array(out_timestamps_on)
    out_positions_on = np.array(out_positions_on)
    out_polarities_on = np.array(out_polarities_on)
    out_timestamps_off = np.array(out_timestamps_off)
    out_positions_off = np.array(out_positions_off)
    out_polarities_off = np.array(out_polarities_off)

    sort_ind = np.argsort(out_timestamps_on)
    out_timestamps_on = out_timestamps_on[sort_ind]
    out_positions_on = out_positions_on[sort_ind]
    out_polarities_on = out_polarities_on[sort_ind]
    on_events = [out_timestamps_on, out_positions_on, out_polarities_on]

    sort_ind = np.argsort(out_timestamps_off)
    out_timestamps_off = out_timestamps_off[sort_ind]
    out_positions_off = out_positions_off[sort_ind]
    out_polarities_off = out_polarities_off[sort_ind]
    off_events = [out_timestamps_off, out_positions_off, out_polarities_off]

    return on_events, off_events
                
    

# Class for HOTS_Sparse_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, reconstruction and classification of the input data
# =============================================================================
class HOTS_Sparse_Net:
    # Network constructor settings, the net is set in a random state,
    # with random basis and activations to define a starting point for the 
    # optimization algorithms
    # Treating ON and OFF events separately for each level generate 2 sublayers for 
    # each sublayer in the previous stage. In the first layer we have a single sublayer
    # (the layer itself), and since that the negative activations needs to be treated 
    # independently, in the next layer we have to train 2 sublayers that will deal
    # with both polarity, that will respectivelty generate 2 polarity events each.
    # This critical structure carries the result that for a layer i, the number \
    # of sublayers of this layer will be 2**i 
    # =============================================================================
    # basis_number: it is a list containing the number of basis used for each layer
    # basis_dimension: it is a list containing the linear dimension of the base set
    #                for each layer (they are expected to be squared)
    # taus: it is a list containing the time coefficient used for the time surface creations
    #       for each layer, all three lists need to share the same lenght obv.
    # delay_coeff: it's the cofficient named alfa used for generating an event in the paper
    # net_seed : seed used for net parameters generation,
    #                       if set to 0 the process will be totally random
    # =============================================================================
    def __init__(self, basis_number, basis_dimension, taus, delay_coeff, net_seed = 0):
        self.basis = []
        self.activations = []
        self.taus = taus
        self.layers = len(basis_number)
        self.basis_dimensions = basis_dimension
        self.basis_number = basis_number
        self.delay_coeff = delay_coeff
        # attribute containing all surfaces computed in each layer and sublayer
        self.surfaces = []
        # attribute containing all optimization errors computed in each layer 
        # and sublayer
        self.errors = []
        #setting the seed
        rng = np.random.RandomState()
        if (net_seed!=0):
            rng.seed(net_seed)
        # In the first layer I am going to process only 2 polarities corresponging
        # to on off events
        num_polarities = 2 
        for layer, nbasis in enumerate(basis_number):
            #basis and activations of a single sublayer
            sublayers_basis = []
            sublayers_activations = []
            for sublayer in range(2**layer):
                #basis and activations of a single layer
                basis_set = []
                activations_set = []
                for j in range(nbasis):
                    basis_set.append(rng.rand(basis_dimension[layer], basis_dimension[layer]*num_polarities))
                    #activations, or aj (as in the paper) are set randomly between -1 and 1
                    activations_set.append((rng.rand()-0.5)*2)
                sublayers_basis.append(np.array(basis_set))
                sublayers_activations.append(np.array(activations_set))
            self.basis.append(sublayers_basis)
            self.activations.append(sublayers_activations)
            num_polarities = nbasis
    
    # Method for online learning, essentially it performs a single step of gradient
    # descent for each time surface, with a settable learning_rate. 
    # The method is full online but it espects to work on an entire batch of data
    # at once to be more similiar in structure to its offline counterpart.
    # This library has the porpouse to test various sparse HOTS implementations
    # and to compare them, not to work in real time
    # =============================================================================
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # learning_rate : the learning rate for the algorithm, newer versions will
    #                   implement a dinamic approach
    # dataset: the initial dataset used for learning
    # base_norm_coeff : l2 regularization coefficient for basis 
    # =============================================================================        
    def learn_online(self, sparsity_coeff, learning_rate, dataset, base_norm_coeff): 
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][events, timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        # In the first layer I am going to process only 2 polarities corresponging
        # to on off events
        num_polarities = 2 
        for layer in range(self.layers):
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as the batches for the 
            # next one
            outbatches = []
            for sublayer in range(len(self.basis[layer])):
                sub_layer_surfaces = []
                all_surfaces = [] # all surfaces computed without distinguishing between batches
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []              
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    batch_activations = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(ldim = self.basis_dimensions[layer],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        all_surfaces.append(tsurface)
                        batch_surfaces.append(tsurface)
                        sub_layer_errors.append(self.sublayer_response_CG(layer, sublayer,
                                                                          tsurface,
                                                                          sparsity_coeff))
                        batch_activations.append(self.activations[layer][sublayer])
                        #run a single step of the gradient descent
                        update_basis_online_hard_treshold(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                            learning_rate, tsurface, base_norm_coeff, 1, 0)
                    #Let's compute again the activations after defining the set of basis
                    sub_layer_activations = []
                    for i in range(len(all_surfaces)):
                        self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff)
                        sub_layer_activations.append(self.activations[layer][sublayer].copy()) 
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(batch_activations,
                                                        [batches[sublayer][batch][0],
                                                         batches[sublayer][batch][1]],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    sub_layer_surfaces.append(batch_surfaces)
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Base optimization process per batch:"+np.str((batch/len(batches[sublayer]))*100)+"%")
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
                # Saving each sub layer events of a single polarity, they will be
                # used to optimize the consequent of suvblayers in the next layer
                outbatches.append(sub_layer_outevents_on)       
                outbatches.append(sub_layer_outevents_off)
            batches = outbatches.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)
            num_polarities = self.basis_number[layer]
    
    # Method for offline mini batch learning, in principle similiar to what implemented
    # by Olshausen and Field 1996
    # link : https://www.sciencedirect.com/science/article/pii/S0042698997001697
    # The method has been probably used for the original sparse HOTS paper too,
    # but it presents several problems like it's small sensitivity on orientation
    # given by the optimization being computed on a mean of several surfaces
    # It has been included as reference for a comparison with the other algorithms 
    # N.B. The basis are regularized with a simple l2 norm, rather than the adapting
    # the one proposed in the paper
    # =============================================================================
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # learning_rate : a list of the learning rates for basis learning for each                                
    #                   training phase implement 
    # dataset: the initial dataset used for learning
    # mini_batch_size : The size on the subset of surfaces on which perform the mean
    # phases_size : The algorith gives the possibility to divide the total process
    #               in several phases with different learning rates for basis learning
    # max_gradient_steps : The number of gradient descent steps to perform for basis learning.
    # base_norm_coeff : l2 regularization coefficient for basis
    # precision : the expected precision of the algorithm (without convergence the alhorithm
    #               will stop when reaching the max number of steps)
    # =============================================================================        
    def learn_offline_mean(self, sparsity_coeff, learning_rate, dataset, mini_batch_size, phases_size, max_gradient_steps, base_norm_coeff, precision): 
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        # In the first layer I am going to process only 2 polarities corresponging
        # to on off events
        num_polarities = 2 
        for layer in range(self.layers):
            # mean surface computed across a mini_batch
            meansurface = np.zeros([self.basis_dimensions[layer],self.basis_dimensions[layer]])
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as the batches for the 
            # next one
            outbatches = []
            for sublayer in range(len(self.basis[layer])):
                sub_layer_surfaces = []
                all_surfaces = [] # all surfaces computed without distinguishing between batches
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                sub_layer_activations = []
                phase_counter = 0
                mini_batch_counter = 0
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    # The reference events timestamps and positions used to generate
                    # the layer output events
                    in_timestamps = []
                    in_positions = []
                    # No more learning needed
                    if (phase_counter==len(phases_size)):
                        break
                    # Delete any uncomplete meansurface from previous batch
                    meansurface = np.zeros([self.basis_dimensions[layer],self.basis_dimensions[layer]*num_polarities])                
                    # counter used to store the number of surface generated, 
                    # useful to understand when it's time to train over a mean 
                    # of the extracted mini_batch
                    surfaces_subset_counter = 0
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k],
                                        batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(ldim = self.basis_dimensions[layer],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        all_surfaces.append(tsurface)
                        batch_surfaces.append(tsurface)
                        meansurface = meansurface + tsurface/mini_batch_size
                        surfaces_subset_counter += 1
                        if (surfaces_subset_counter == mini_batch_size):
                            in_timestamps.append(single_event[0])
                            in_positions.append(single_event[1])
                            sub_layer_errors.append(self.sublayer_response_CG(layer, sublayer, meansurface, sparsity_coeff))
                            # using the update basis online function, but with time surfaces obtained by performing
                            # a mean on multiple surfaces (mini_batch), thus this is considered an offline model
                            update_basis_online_hard_treshold(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                                learning_rate[phase_counter], meansurface,
                                                base_norm_coeff, max_gradient_steps, precision)
                            mini_batch_counter += 1
                            surfaces_subset_counter = 0
#                            plt.figure()
#                            sns.heatmap(meansurface)  
                            meansurface = np.zeros([self.basis_dimensions[layer],self.basis_dimensions[layer]*num_polarities])
                            sub_layer_activations.append(self.activations[layer][sublayer])
                        if (mini_batch_counter == phases_size[phase_counter]):
                            phase_counter += 1
                            mini_batch_counter = 0
                            if(phase_counter == len(phases_size)):
                                break

                    #Let's compute again the activations after defining the set of basis
                    sub_layer_activations = []
                    for i in range(len(all_surfaces)):
                        self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff)
                        sub_layer_activations.append(self.activations[layer][sublayer].copy()) 
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(sub_layer_activations,
                                                        [np.array(in_timestamps),
                                                         np.array(in_positions)],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    sub_layer_surfaces.append(batch_surfaces)
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Base optimization process per batch:"+np.str((batch/len(batches[sublayer]))*100)+"%")
                # check if during the sublayer learning it did not completed all the phases
                # due to insufficient data
                if(phase_counter != len(phases_size)):
                   print("Warning: "+"Layer_"+np.str(layer)+"-Sublayer_"+np.str(sublayer)+" learning process ended before ending phase:"+np.str(phase_counter)) 
                   print("Consider changing mini_batch_size and phases_size for improve results") 
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
                # Saving each sub layer events of a single polarity, they will be
                # used to optimize the consequent of suvblayers in the next layer
                outbatches.append(sub_layer_outevents_on)       
                outbatches.append(sub_layer_outevents_off)
            batches = outbatches.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)
            num_polarities = self.basis_number[layer]
        
    # Method for offline batch learning, derived from Olshausen and Field 1996
    # link : https://www.sciencedirect.com/science/article/pii/S0042698997001697
    # Like in the original work, first it computes the sparse response a_j, but then
    # it performs cognugate gradient descent on the all batch of images at the same time
    # to find the optimal basis. This process is then iterated until it reaches the 
    # expected precision or the max number of steps
    # Advice : It is slow as 2001: A Space Odyssey, but can be considered a good 
    # groundtruth for comparisons
    # NB: If you have a background of machine learning, please note that here 
    # the term batch is used to define a single record of events (like multiple
    # events recorded from the same letter in a multiple letter dataset)
    # Therefore this offline algorithm is performing optimization through different
    # batches at the same time. I'm sorry if you are confused by the therminolgy, 
    # looking at the code will simplify things...... ahahah just kidding it's a mess
    # =============================================================================
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # learning_rate : a list of the learning rates for basis learning for each                                
    #                   training phase implement 
    # dataset: the initial dataset used for learning
    # max_steps : The number of maximum steps performed for basis learning.
    # base_norm_coeff : l2 regularization coefficient for basis
    # precision : the expected precision of the algorithm (notice that without 
    #             a quick enough convergence the alhorithm will stop when 
    #             reaching the max number of steps)
    # =============================================================================      
    def learn_offline(self, sparsity_coeff, learning_rate, dataset, max_steps, base_norm_coeff, precision):
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        # In the first layer I am going to process only 2 polarities corresponging
        # to on off events
        num_polarities = 2 
        for layer in range(self.layers):
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as the batches for the 
            # next one
            outbatches = []
            for sublayer in range(len(self.basis[layer])):
                sub_layer_surfaces = []
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                # A single array containing all the generated surfaces 
                # for a single sublayer used for full batch learning 
                all_surfaces = []
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(ldim = self.basis_dimensions[layer],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        batch_surfaces.append(tsurface)
                        # A single array containing all the generated surfaces 
                        # for a single sublayer used for full batch learning
                        all_surfaces.append(tsurface)                    
                    sub_layer_surfaces.append(batch_surfaces)
                
                # Learning time
                n_steps = 0 
                previous_err = 0
                err = precision + 1
                while (abs(err-previous_err)>precision) & (n_steps<max_steps):
                    #Let's compute the activations
                    sub_layer_activations = []
                    for i in range(len(all_surfaces)):
                        self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff)
                        sub_layer_activations.append(self.activations[layer][sublayer].copy())    

                    #Let's compute the basis
                    opt_res = update_basis_offline_CG(self.basis[layer][sublayer], self.basis_dimensions[layer]*num_polarities,
                                            self.basis_dimensions[layer], 
                                            self.basis_number[layer], sub_layer_activations,
                                            all_surfaces, base_norm_coeff)
                    self.basis[layer][sublayer] = opt_res.x.reshape(self.basis_number[layer],
                              self.basis_dimensions[layer], self.basis_dimensions[layer]*num_polarities)
                    sub_layer_errors.append(opt_res.fun)
                    n_steps += 1
                    previous_err = err
                    err = sub_layer_errors[-1]
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Base optimization process:"+np.str((n_steps/max_steps)*100)+"%")
                if (n_steps<max_steps):
                    print("Execution stopped, requested precision reached")
                else:
                    print("Maximum number of step reached, last delta error was :" + np.str(abs(err-previous_err)))
               
                #Let's compute again the activations after defining the set of basis
                sub_layer_activations = []
                for i in range(len(all_surfaces)):
                    self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff)
                    sub_layer_activations.append(self.activations[layer][sublayer].copy()) 
                # sub_layer_activations contains all the activations to all the time surfaces
                # of different baches, in order to build the next layer batches 
                # I need to redefine the original batches ordering of the data
                current_position = 0
                for batch in range(len(batches[sublayer])):
                    current_batch_length = len(batches[sublayer][batch][0])       
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(sub_layer_activations[current_position:(current_position+current_batch_length)],
                                                        [batches[sublayer][batch][0],
                                                         batches[sublayer][batch][1]],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    current_position += current_batch_length
                        
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
                # Saving each sub layer events of a single polarity, they will be
                # used to optimize the consequent of suvblayers in the next layer
                outbatches.append(sub_layer_outevents_on)       
                outbatches.append(sub_layer_outevents_off)
            batches = outbatches.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)
            num_polarities = self.basis_number[layer]
                
                        
                    
    # Method for computing and updating the network activations for a single time surface
    # using conjugate gradient descent
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer 
    # timesurface: the imput time surface
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    #
    # It also return the Error of reconstruction
    # =============================================================================  
    def sublayer_response_CG(self, layer, sublayer, timesurface, sparsity_coeff):
        # Error functions and derivatives for gradient descent
        Err = lambda a_j, S, Phi_j, lam: error_func(a_j, S, Phi_j, lam)
        dErrdaj = lambda a_j, S, Phi_j, lam: error_func_deriv_a_j(a_j, S, Phi_j, lam)
        starting_points = np.zeros(len(self.activations[layer][sublayer]))
        res = optimize.minimize(fun=Err, x0=starting_points,
                                args=(timesurface, self.basis[layer][sublayer],
                                      sparsity_coeff), method="CG", jac=dErrdaj)
        self.activations[layer][sublayer]=res.x
        return res.fun
    
    # Method for computing the network response for a whole dataset
    # using conjugate gradient descent
    # =============================================================================
    # dataset: the input dataset the network will response to
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    #
    # It returns the whole history of the network activity
    # =============================================================================  
    def full_net_dataset_response_CG(self, dataset, sparsity_coeff):
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][events, timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        full_net_activations = []
        # In the first layer I am going to process only 2 polarities corresponging
        # to on off events
        num_polarities = 2 
        for layer in range(self.layers):
            outbatches = []
            layer_activations = []
            for sublayer in range(len(self.basis[layer])):
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                sub_layer_activations = []
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    batch_activations = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(ldim = self.basis_dimensions[layer],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        batch_surfaces.append(tsurface)
                        sub_layer_errors.append(self.sublayer_response_CG(layer, sublayer,
                                                                          tsurface,
                                                                          sparsity_coeff))
                        batch_activations.append(self.activations[layer][sublayer])
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(batch_activations,
                                                        [batches[sublayer][batch][0],
                                                         batches[sublayer][batch][1]],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    sub_layer_activations.append(batch_activations)
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Batch processing:"+np.str((batch/len(batches[sublayer]))*100)+"%")
                layer_activations.append(sub_layer_activations)
                outbatches.append(sub_layer_outevents_on)       
                outbatches.append(sub_layer_outevents_off)
            full_net_activations.append(layer_activations)
            batches = outbatches.copy()
            num_polarities = self.basis_number[layer]

        return full_net_activations
    
    
    
    # Method for plotting reconstruction heatmap and reconstruction error
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer     
    # timesurface: the imput time surface
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # It also returns the Error of reconstruction
    # =============================================================================  
    def sublayer_reconstruct(self, layer, sublayer, timesurface, sparsity_coeff):
        error = self.sublayer_response_CG(layer, sublayer, timesurface, sparsity_coeff)
        S_tilde = sum([a*b for a,b in zip(self.basis[layer][sublayer],
                                          self.activations[layer][sublayer])]) 
        plt.figure()
        sns.heatmap(S_tilde)
        print(error)
        return error
    
    # Method for computing the reconstruction error of a sublayer set of basis
    # for an entire batch of time surfaces
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer         
    # timesurfaces: the batch of time surfaces that the layer will try to reconstruct
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    #               coefficient used to impose sparseness on the net activations aj
    #
    # It returns the mean error of reconstruction
    # =============================================================================  
    def batch_sublayer_reconstruct_error(self, layer, sublayer, timesurfaces, sparsity_coeff):
        error = 0
        total_surf = 0
        for i in range(len(timesurfaces)):
            for k in range(len(timesurfaces[i])):
                error += self.sublayer_response_CG(layer, sublayer, timesurfaces[i][k], sparsity_coeff)
                total_surf += 1
        return error/total_surf

    
    # Method for training an histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # dataset: the input dataset the network will response to
    # labels: the labels used for the classification task (positive integers from 0 to N)
    # number_of_labels: the maximum number of labels of the dataset
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    #               coefficient used to impose sparseness on the net activations aj
    # =============================================================================      
    def histogram_classification_train(self, dataset, labels, number_of_labels, sparsity_coeff):
        net_activity = self.full_net_dataset_response_CG(dataset, sparsity_coeff)
        last_layer_activity = net_activity[-1]        
        histograms = []
        normalized_histograms = []
        n_basis = self.basis_number[-1]
        # this array it's used to count the number of batches with a certain label,
        # the values will be then used to compute normalized histograms
        for label in range(number_of_labels):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for sublayer in range(len(last_layer_activity)):
            num_of_batch_per_labels = np.zeros(number_of_labels)
            for batch in range(len(dataset)):
                current_label = labels[batch]
                batch_histogram = sum(last_layer_activity[sublayer][batch])
                normalized_bach_histogram = batch_histogram/len(last_layer_activity[sublayer][batch])
                histograms[current_label][n_basis*sublayer:n_basis*(sublayer+1)] += batch_histogram
                normalized_histograms[current_label][n_basis*sublayer:n_basis*(sublayer+1)] += normalized_bach_histogram
                num_of_batch_per_labels[current_label] += 1
        for label in range(number_of_labels):
            normalized_histograms[label] = normalized_histograms[label]/num_of_batch_per_labels[label]
            histograms[label] = histograms[label]/num_of_batch_per_labels[label]
        self.histograms = histograms
        self.normalized_histograms = normalized_histograms
        print("Training ended, you can now look at the histograms with in the attribute .histograms and .normalized_histograms")

    # Method for testing the histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # dataset: the input dataset the network will response to
    # labels: the labels used for the classification task (positive integers from 0 to N)
    # number_of_labels: the maximum number of labels of the dataset
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    #               coefficient used to impose sparseness on the net activations aj
    # It returns the computed distances per batch (Euclidean, normalized Euclidean
    # and Bhattacharyyan) and the predicted_labels per batch and per metric(distance)
    # =============================================================================      
    def histogram_classification_test(self, dataset, labels, number_of_labels, sparsity_coeff):
        net_activity = self.full_net_dataset_response_CG(dataset, sparsity_coeff)
        last_layer_activity = net_activity[-1]
        histograms = []
        normalized_histograms = []
        n_basis = self.basis_number[-1]
        for batch in range(len(dataset)):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for sublayer in range(len(last_layer_activity)):
            for batch in range(len(dataset)):
                batch_histogram = sum(last_layer_activity[sublayer][batch])
                normalized_bach_histogram = batch_histogram/len(last_layer_activity[sublayer][batch])
                histograms[batch][n_basis*sublayer:n_basis*(sublayer+1)] = batch_histogram
                normalized_histograms[batch][n_basis*sublayer:n_basis*(sublayer+1)] = normalized_bach_histogram
        # compute the distances per each histogram from the models
        distances = []
        predicted_labels = []
        for batch in range(len(dataset)):
            single_batch_distances = []
            for label in range(number_of_labels):
                single_label_distances = []  
                single_label_distances.append(distance.euclidean(histograms[batch],self.histograms[label]))
                single_label_distances.append(distance.euclidean(normalized_histograms[batch],self.normalized_histograms[label]))
                Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(normalized_histograms[batch], self.normalized_histograms[label])]) 
                single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
                single_batch_distances.append(single_label_distances)
            single_batch_distances = np.array(single_batch_distances)
            single_batch_predicted_labels = np.argmin(single_batch_distances, 0)
            distances.append(single_batch_distances)
            predicted_labels.append(single_batch_predicted_labels)
        
        return distances, predicted_labels

    
    # Method for plotting the basis set of a single sublayer
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer
    # =============================================================================
    def plot_basis(self, layer, sublayer):
        for i in range(self.basis_number[layer]):
            plt.figure()
            sns.heatmap(self.basis[layer][sublayer][i])