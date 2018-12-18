#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax

This file contains the HOTS_Sparse_Net class 

To better understand the math behind classification an optimization please take a look
to the original Sparse Hots paper and the new one by myself

The links:
 
"""
import numpy as np 
from scipy import optimize
from scipy.spatial import distance 
import matplotlib.pyplot as plt
import seaborn as sns
from Libs.Time_Surface_generators import Time_Surface_event
from Libs.HOTS_Sparse_Libs import error_func, error_func_deriv_a_j,\
error_func_phi_full_batch, error_func_phi_grad_full_batch, update_basis_online,\
update_basis_online_hard_treshold, update_basis_offline_CG, events_from_activations,\
create_figures, update_figures, exp_decay

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
    # basis_dimension: it is a list containing the linear dimensions of the base set
    #                for each layer 
    # taus: it is a list containing the time coefficient used for the time surface creations
    #       for each layer, all three lists need to share the same lenght obv.
    # first_layer_polarities : The number of polarities that the first layer will
    #                          have to handle, for ATIS cameras, it is usually,
    #                          2 or 1 depending wether if you consider the additional 
    #                                   
    # delay_coeff: it's the cofficient named alfa used for generating an event in the paper
    # net_seed : seed used for net parameters generation,
    #                       if set to 0 the process will be totally random
    # =============================================================================
    def __init__(self, basis_number, basis_dimensions, taus, first_layer_polarities, delay_coeff, net_seed = 0):
        self.basis = []
        self.activations = []
        self.taus = taus
        self.layers = len(basis_number)
        self.basis_dimensions = basis_dimensions
        self.basis_number = basis_number
        self.first_layer_polarities = first_layer_polarities
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
        num_polarities = 1 
        for layer, nbasis in enumerate(basis_number):
            #basis and activations of a single sublayer
            sublayers_basis = []
            sublayers_activations = []
            for sublayer in range(2**layer):
                #basis and activations of a single layer
                basis_set = []
                activations_set = []
                for j in range(nbasis):
                    basis_set.append(rng.rand(basis_dimensions[layer][1], basis_dimensions[layer][0]*num_polarities))
                    basis_set[j][basis_dimensions[layer][1]//2, [basis_dimensions[layer][0]//2 + basis_dimensions[layer][0]*a  for a in range(num_polarities)]] = 1
                    #activations, or aj (as in the paper) are set randomly between -1 and 1
                    activations_set.append((rng.rand()-0.5)*2)
                sublayers_basis.append(np.array(basis_set))
                sublayers_activations.append(np.array(activations_set))
            self.basis.append(sublayers_basis)
            self.activations.append(sublayers_activations)
            num_polarities = nbasis
    

    
    # Method for online learning, essentially it performs a single step of gradient
    # descent for each time surface, with a settable learning_rate. 
    # The method is full online but it espects to work on an entire dataset
    # at once to be more similiar in structure to its offline counterpart.
    # This library has the porpouse to test various sparse HOTS implementations
    # and to compare them, not to work in real time! (If it's worthy a real time
    # will hit the shelfs soon)
    # =============================================================================
    # dataset: the initial dataset used for learning
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    # base_norm : Select whether use L2 norm or threshold constraints on basis
    #             learning with the keywords "L2" "Thresh" 
    # -----------------------------------------------------------------------------
    # The following series of parameters can evolve during learning, thus 
    # they are expected to be defined as lists with [begin_value, end_value, time_coeff]
    # Where begin_value is the value used for learning the first time surface,
    # end_value is the value at steady state and time_coeff defines how quickly,
    # the monotonic function describing the evolution of the parameter reaches it
    #
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)
    # learning_rate : It's the learning rate, what do you expect me to say ?                  
    # ----------------------------------------------------------------------------- 
    # base_norm_coeff : l2 regularization coefficient for basis (Not used if
    #                   base_norm="Thresh")
    # verbose : If verbose is on, it will display graphs to assest the ongoing,
    #           learning algorithm
    # =============================================================================        
    def learn_online(self, dataset, method="Exp distance", base_norm="Thresh", 
                     noise_ratio=[0,0,1], sparsity_coeff=[0,0,1], sensitivity=[0,0,1],
                     learning_rate=[0,0,1], base_norm_coeff=0, verbose=False): 
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][events, timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        
        # Setting the number of polarities for the first layer 
        num_polarities = self.first_layer_polarities
        
        self.evolution_pre_allocation(len(dataset))
        for layer in range(self.layers):
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as the batches for the 
            # next one
            outbatches = []
            
            for sublayer in range(len(self.basis[layer])):
              
                # Create network evolution parameters, at the moment I'm using 
                # an exponential decay kernel as a function, it can be changed in the code pretty
                # easly
                
                # end_time:  how many time surfaces will be generated in the learing 
                # process for this sublayer
                end_time = 0
                for i in range(len(batches[sublayer])):
                    end_time += len(batches[sublayer][i][0])
                time = np.arange(end_time)
                
                noise_evolution = exp_decay(noise_ratio[0], noise_ratio[1], noise_ratio[2], time)
                learning_evolution = exp_decay(learning_rate[0], learning_rate[1], learning_rate[2], time)
                sparsity_evolution = exp_decay(sparsity_coeff[0], sparsity_coeff[1], sparsity_coeff[2], time)                
                sensitivity_evolution = exp_decay(sensitivity[0], sensitivity[1], sensitivity[2], time)
                
                # Plot evolution parameters and basis if required 
                if verbose is True:
                    figures, axes = create_figures(surfaces=self.basis[layer][sublayer], num_of_plots=self.basis_number[layer])
                    tsurface_fig, tsurface_ax = create_figures(surfaces=self.basis[layer][sublayer][0], num_of_plots=1, fig_names="Input surface")
                    net_evolution_fig = plt.figure('Network parameters evolution, Layer: '+str(layer)+' Sublayer: '+str(sublayer))
                    net_evolution_ax = []
                    net_evolution_ax.append(net_evolution_fig.add_subplot(311))
                    net_evolution_ax.append(net_evolution_fig.add_subplot(312))
                    net_evolution_ax.append(net_evolution_fig.add_subplot(313))
                    net_evolution_ax[0].plot(noise_evolution)
                    net_evolution_ax[0].set_title('Activity Noise ratio')
                    net_evolution_ax[1].plot(learning_evolution)
                    net_evolution_ax[1].set_title('Learning step size')
                    net_evolution_ax[2].plot(sparsity_evolution)
                    net_evolution_ax[2].set_title('Sparsity coefficient')
                    plt.show()
                    plt.draw()

                    
                # Counter of surfaces that will be used to update evolution parameters
                time = 0
                sub_layer_surfaces = []
                sub_layer_reference_events = []
                sub_layer_errors = []
           
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    batch_reference_events = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(xdim = self.basis_dimensions[layer][0],
                                                    ydim = self.basis_dimensions[layer][1],
                                                    event=single_event,
                                                    timecoeff=self.taus[layer],
                                                    dataset=batches[sublayer][batch],
                                                    num_polarities=num_polarities,
                                                    verbose=False)
                        if method=="CG" :
                            sub_layer_errors.append(self.sublayer_response_CG(layer, sublayer,
                                                                                      tsurface,
                                                                                      sparsity_evolution[time], 
                                                                                      noise_evolution[time]))
                        if method=="Exp distance" :
                            sub_layer_errors.append(self.sublayer_response_exp_distance(layer, sublayer,
                                                                                        tsurface,
                                                                                        sparsity_evolution[time], 
                                                                                        noise_evolution[time], 
                                                                                        sensitivity_evolution[time]))
                        batch_surfaces.append(tsurface)
                        batch_reference_events.append(single_event)
                        # Run a single step of the gradient descent
                        if base_norm=="L2":
                            update_basis_online(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                                    learning_evolution[time], tsurface, base_norm_coeff, 1, 0)
                        if base_norm=="Thresh":
                            update_basis_online_hard_treshold(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                                              learning_evolution[time], tsurface, 1, 0)
                        
                        
                        # Plot updated basis at each cycle if required
                        if verbose is True:
                            update_figures(figures=figures, axes=axes, surfaces=self.basis[layer][sublayer])
                            update_figures(figures=tsurface_fig, axes=tsurface_ax, surfaces=tsurface)
                            plt.draw()
                            plt.pause(0.0001)
                        time += 1
                    if time!=0:                        
                        self.evolution_save_point(layer, sublayer, batch, noise_evolution[time-1], sparsity_evolution[time-1], sensitivity_evolution[time-1], learning_evolution[time-1])
                    sub_layer_surfaces.append(batch_surfaces)
                    sub_layer_reference_events.append(batch_reference_events)
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Base optimization process per batch:"+np.str(((batch+1)/len(batches[sublayer]))*100)+"%")

                # Check if we are at the last layer or we need to recompute the activations
                # to train the next layer
                if layer+1 != self.layers:
                    # Let's compute again the activations after defining the set of basis!
                    
                    # I use the evolution parameters at the steady state
                    time = time - 1
                    # Where I store all the events computed in a sublayer
                    # on and off activations originate two batches out of a single one
                    sub_layer_outevents_on = []
                    sub_layer_outevents_off = []
                    for batch in range(len(batches[sublayer])):
                        batch_activations = []
                        for k in range(len(batches[sublayer][batch][0])):
                            if method=="CG" :
                                self.sublayer_response_CG(layer, sublayer,
                                                          sub_layer_surfaces[batch][k],
                                                          sparsity_evolution[time], 
                                                          noise_evolution[time])
                            if method=="Exp distance" :
                                self.sublayer_response_exp_distance(layer, sublayer,
                                                                    sub_layer_surfaces[batch][k],
                                                                    sparsity_evolution[time], 
                                                                    noise_evolution[time], 
                                                                    sensitivity_evolution[time])
                            batch_activations.append(self.activations[layer][sublayer].copy())

                        # Obtaining the events resulting from the activations in a single batch
                        outevents = events_from_activations(batch_activations,
                                                            sub_layer_reference_events[batch],
                                                             self.delay_coeff)
                        sub_layer_outevents_on.append(outevents[0])
                        sub_layer_outevents_off.append(outevents[1])
                        
                    
                    # They will be used to optimize the consequent of sublayers
                    # in the next layer
                    outbatches.append(sub_layer_outevents_on)       
                    outbatches.append(sub_layer_outevents_off)       
                    
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
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
    # dataset: the initial dataset used for learning
    #                     
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # learning_rate : a list of the learning rates for basis learning for each                                
    #                   training phase implement 
    # max_steps : The number of maximum steps performed for basis learning.
    # base_norm_coeff : l2 regularization coefficient for basis
    # precision : the expected precision of the algorithm (not
    #             a quick enough convergence the alhorithm will stop when 
    #             reaching the max number of steps)
    # verbose : If verbose is on, it will display graphs to assest the ongoing,
    #           learning algorithm
    # =============================================================================      
    def learn_offline(self, dataset, sparsity_coeff, learning_rate,
                      max_steps, base_norm_coeff, precision, verbose=False):
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        # Setting the number of polarities for the first layer 
        num_polarities = self.first_layer_polarities
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
                all_reference_events = []
                # Plot basis if required 
                if verbose is True:
                    figures, axes = create_figures(surfaces=self.basis[layer][sublayer],
                                                   num_of_plots=self.basis_number[layer])
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(xdim = self.basis_dimensions[layer][0],
                                                      ydim = self.basis_dimensions[layer][1],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        batch_surfaces.append(tsurface)
                        # A single array containing all the generated surfaces 
                        # for a single sublayer used for full batch learning
                        all_surfaces.append(tsurface) 
                        all_reference_events.append(single_event)
                    sub_layer_surfaces.append(batch_surfaces)
                
                # Learning time
                n_steps = 0 
                previous_err = 0
                err = precision + 1
                while (abs(err-previous_err)>precision) & (n_steps<max_steps):
                    #Let's compute the activations
                    sub_layer_activations = []
                    for i in range(len(all_surfaces)):
                        self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff, 0)
                        sub_layer_activations.append(self.activations[layer][sublayer].copy())    
                    #Let's compute the basis
                    opt_res = update_basis_offline_CG(self.basis[layer][sublayer],
                                                      self.basis_dimensions[layer][0]*num_polarities,
                                                      self.basis_dimensions[layer][1], 
                                                      self.basis_number[layer], sub_layer_activations,
                                                      all_surfaces, base_norm_coeff)
                    
                    self.basis[layer][sublayer] = opt_res.x.reshape(self.basis_number[layer],
                              self.basis_dimensions[layer][1], self.basis_dimensions[layer][0]*num_polarities)
                    sub_layer_errors.append(opt_res.fun)
                    # Plot updated basis at each cycle if required
                    if verbose is True:
                        update_figures(figures=figures, axes=axes, surfaces=self.basis[layer][sublayer])
                        plt.draw()
                        plt.pause(0.0001)
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
                    self.sublayer_response_CG(layer, sublayer, all_surfaces[i], sparsity_coeff, 0)
                    sub_layer_activations.append(self.activations[layer][sublayer].copy()) 
                # sub_layer_activations contains all the activations to all the time surfaces
                # of different baches, in order to build the next layer batches 
                # I need to redefine the original batches ordering of the data
                current_position = 0
                for batch in range(len(batches[sublayer])):
                    current_batch_length = len(batches[sublayer][batch][0])       
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(sub_layer_activations[current_position:(current_position+current_batch_length)],
                                                        all_reference_events[current_position:(current_position+current_batch_length)],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    current_position += current_batch_length
                        
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
                # Saving each sub layer events deviding between negative and positive
                # activations, they will be used to optimize the consequent of 
                # sublayers in the next layer
                outbatches.append(sub_layer_outevents_on)       
                outbatches.append(sub_layer_outevents_off)
            batches = outbatches.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)
            num_polarities = self.basis_number[layer]

    # Method that preallocates lists that will store evolution save points
    # =============================================================================
    # dataset_length : the lenght of the entire dataset in number of batches
    # =============================================================================  
    def evolution_pre_allocation(self, dataset_length):
        #it will store the intial basis too, thus +1 on the dataset length 
        self.basis_history=[[[np.zeros(self.basis[layer][sublayer].shape) for batch in range(dataset_length+1)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        for layer in range(self.layers):
            for sublayer in range(2**layer):
                self.basis_history[layer][sublayer][0]=self.basis[layer][sublayer]
        self.noise_history=[[[0 for batch in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        self.sparsity_history=self.noise_history.copy()
        self.sensitivity_history=self.noise_history.copy()
        self.learning_history=self.noise_history.copy()
        self.basis_distance=[[[np.zeros(len(self.basis[layer][sublayer])) for batch in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        
                
    # Method useful to save network status in relation of its previous history
    # in this way is possible to test evolution parameters and test wether an 
    # online method is overfitting in a particular batch.
    # In the future the outlook of the online method will be to implement overparameters
    # to control the network in a fine and less supervised manner.
    # Thus this method won't be needed anymore
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer 
    # batch : the index of the batch in the dataset
    # noise: the noise coefficient at the saving time.
    # sparsity: the sparsity coefficient at the saving time.
    # sensitivity: the sensitivity coefficient at the saving time.
    # learning: the learning rate at the saving time.
    #
    # It saves the set of basis, and the status of the evolution parameters.
    # =============================================================================  
    def evolution_save_point(self, layer, sublayer, batch, noise, sparsity, sensitivity, learning):
        self.basis_history[layer][sublayer][batch+1]=self.basis[layer][sublayer]
        self.noise_history[layer][sublayer][batch]=noise
        self.sparsity_history[layer][sublayer][batch]=sparsity
        self.sensitivity_history[layer][sublayer][batch]=sensitivity
        self.learning_history[layer][sublayer][batch]=learning
        distance = np.zeros(len(self.basis[layer][sublayer]))
        for i,feature in enumerate(self.basis[layer][sublayer]):
            distance[i] = np.sum(self.basis_history[layer][sublayer][batch][i,:,:]-feature)**2
        self.basis_distance[layer][sublayer][batch]=distance
                
        
        
    # Method for computing and updating the network activations for a single time surface
    # using conjugate gradient descent
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer 
    # timesurface: the imput time surface
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # coefficient used to impose sparseness on the net activations aj
    # noise_coeff: coefficient use to add noise to the net activities
    #
    # It also return the Error of reconstruction
    # =============================================================================  
    def sublayer_response_CG(self, layer, sublayer, timesurface, sparsity_coeff, noise_coeff):
        # Error functions and derivatives for gradient descent
        Err = lambda a_j, S, Phi_j, lam: error_func(a_j, S, Phi_j, lam)
        dErrdaj = lambda a_j, S, Phi_j, lam: error_func_deriv_a_j(a_j, S, Phi_j, lam)
        starting_points = np.zeros(len(self.activations[layer][sublayer]))
        res = optimize.minimize(fun=Err, x0=starting_points,
                                args=(timesurface, self.basis[layer][sublayer],
                                      sparsity_coeff), method="CG", jac=dErrdaj)
        activations=res.x
        # Here I mix noise with the activations
        activations = activations*(1-noise_coeff) + (noise_coeff)*np.exp(-np.random.rand(len(activations)))
        self.activations[layer][sublayer]=activations
        return res.fun
    
    
    
    # Method for computing and updating the network activations for a single time surface
    # using exponential distance between the surface and the basis
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer 
    # timesurface: the imput time surface
    # sparsity_coeff: It is a value moltiplicating the activity of the best 
    #                 fitting base. The result will be subtracted to the other
    #                 activities de facto implementing a simple lateral inhibition
    # noise_coeff: coefficient use to add noise to the net activities
    # sensitivity: it shapes the sensitivty for each feature in computing the 
    #              activities of each base
    # It also return the Error of reconstruction of the whole timesurface
    # =============================================================================  
    def sublayer_response_exp_distance(self, layer, sublayer, timesurface, sparsity_coeff, noise_coeff, sensitivity):
        euclidean_distances = []
        for i, base in enumerate(self.basis[layer][sublayer]):
            euclidean_distances.append(np.sum(np.abs(base-timesurface)**2))
        activations = np.exp(-sensitivity*np.array(euclidean_distances))
        # Here I mix noise with the activations
        activations = activations*(1-noise_coeff) + (noise_coeff)*np.exp(-np.random.rand(len(activations)))
        # Here I implement the lateral inhibition
        winner_ind=np.argmax(activations)
        losers_ind= np.arange(len(activations))!=winner_ind
        activations[losers_ind] = activations[losers_ind]-sparsity_coeff*activations[winner_ind]
        # I kill the basis too inhibited (with negative values) and update the activities
        activations[activations<=0] = 0
        self.activations[layer][sublayer] = activations
        # Here i compute the error as the euclidean distance between a reconstruced
        # Time surface and the original one
        S_tilde = sum([a*b for a,b in zip(self.basis[layer][sublayer], activations)])
        residue = timesurface-S_tilde
        error =  np.sum(residue**2)
        return error
    
    

    # Method for computing the network response for a whole dataset
    # =============================================================================
    # dataset: the dataset that you want to compute divided in batches,
    #          that you want to classificate
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    #
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)   
    #            
    # It returns the whole history of the network activity
    # =============================================================================  
    def full_net_dataset_response(self, dataset, method="Exp distance", noise_ratio=0, sparsity_coeff=0,
                     sensitivity=0):
        # The list of all data batches devided for sublayer 
        # batches[sublayer][actual_batch][events, timestamp if 0 or xy coordinates if 1]
        batches = []
        # In the first layer the data batches are the input dataset given by the user
        batches.append(dataset)
        full_net_activations = []
        #
        num_polarities = self.first_layer_polarities
        for layer in range(self.layers):
            outbatches = []
            layer_activations = []
            for sublayer in range(len(self.basis[layer])):
                # Counter of surfaces that will be used to update evolution parameters
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                sub_layer_activations = []
                for batch in range(len(batches[sublayer])):
                    batch_surfaces = []
                    batch_reference_events = []
                    batch_activations = []
                    for k in range(len(batches[sublayer][batch][0])):
                        single_event = [batches[sublayer][batch][0][k], batches[sublayer][batch][1][k]]
                        tsurface = Time_Surface_event(xdim = self.basis_dimensions[layer][0],
                                                      ydim = self.basis_dimensions[layer][1],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=batches[sublayer][batch],
                                                      num_polarities=num_polarities,
                                                      verbose=False)
                        batch_surfaces.append(tsurface)
                        batch_reference_events.append(single_event)
                        if method == "CG":
                            sub_layer_errors.append(self.sublayer_response_CG(layer, sublayer,
                                                                              tsurface,
                                                                              sparsity_coeff,
                                                                              noise_ratio))
                        if method == "Exp distance": 
                            sub_layer_errors.append(self.sublayer_response_exp_distance(layer, sublayer,
                                                                          tsurface,
                                                                          sparsity_coeff, 
                                                                          noise_ratio, 
                                                                          sensitivity))
                        
                        
                        batch_activations.append(self.activations[layer][sublayer])
                    # Obtaining the events resulting from the activations in a single batch
                    outevents = events_from_activations(batch_activations,
                                                        batch_reference_events,
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    sub_layer_activations.append(batch_activations)
                    print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Batch processing:"+np.str(((batch+1)/len(batches[sublayer]))*100)+"%")
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
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    #
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)   
    #    
    # It also returns the Error of reconstruction
    # =============================================================================  
    def sublayer_reconstruct(self, layer, sublayer, timesurface, method,
                             noise_ratio, sparsity_coeff, sensitivity):
        if method == "CG":
            error = self.sublayer_response_CG(layer, sublayer,
                                              timesurface,
                                              sparsity_coeff,
                                              noise_ratio)
        if method == "Exp distance": 
            error = self.sublayer_response_exp_distance(layer, sublayer,
                                                        timesurface,
                                                        sparsity_coeff, 
                                                        noise_ratio, 
                                                        sensitivity)
        print("Layer activity : ")
        print(self.activations[layer][sublayer])   
        S_tilde = sum([a*b for a,b in zip(self.basis[layer][sublayer],
                                          self.activations[layer][sublayer])]) 
        plt.figure("Reconstructed Figure")
        sns.heatmap(S_tilde)
        print("Reconstruction error : ")
        print(error)
        return error
    
    # Method for computing the reconstruction error of a sublayer set of basis
    # for an entire batch of time surfaces
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer         
    # timesurfaces: the batch of time surfaces that the layer will try to reconstruct
    # sparsity_coeff: reported as lambda in the Sparse Hots paper, is the norm
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    #
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)   
    #    
    # It returns the mean error of reconstruction
    # =============================================================================  
    def batch_sublayer_reconstruct_error(self, layer, sublayer, timesurfaces,
                                         method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        error = 0
        total_surf = 0
        for i in range(len(timesurfaces)):
            for k in range(len(timesurfaces[i])):
                if method == "CG":
                    error = self.sublayer_response_CG(layer, sublayer,
                                                      timesurfaces[i][k],
                                                      sparsity_coeff,
                                                      noise_ratio)
                if method == "Exp distance": 
                    error = self.sublayer_response_exp_distance(layer, sublayer,
                                                                timesurfaces[i][k],
                                                                sparsity_coeff, 
                                                                noise_ratio, 
                                                                sensitivity)
                total_surf += 1
        return error/total_surf

    
    # Method for training an histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # dataset: the input dataset the network will respond to
    # labels: the labels used for the classification task (positive integers from 0 to N)
    # number_of_labels: the maximum number of labels of the dataset
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)   
    #  
    # =============================================================================      
    def histogram_classification_train(self, dataset, labels, number_of_labels,
                                       method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        
        
        last_layer_activity = net_activity[-1]        
        histograms = []
        normalized_histograms = []
        n_basis = self.basis_number[-1]
        # Normalization factor for building normalized histograms
        input_spikes_per_label = np.zeros(number_of_labels)
        batch_per_label  = np.zeros(number_of_labels)
        # this array it's used to count the number of batches with a certain label,
        # the values will be then used to compute mean histograms
        for label in range(number_of_labels):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for batch in range(len(dataset)):
            current_label = labels[batch]
            input_spikes_per_label[current_label] += len(dataset[batch][0])
            batch_per_label[current_label] += 1            
            for sublayer in range(len(last_layer_activity)):
                batch_histogram = sum(last_layer_activity[sublayer][batch])
                histograms[current_label][n_basis*sublayer:n_basis*(sublayer+1)] += batch_histogram               
        for label in range(number_of_labels):
            normalized_histograms[label] = histograms[label]/input_spikes_per_label[label]
            histograms[label] = histograms[label]/batch_per_label[label]
        self.histograms = histograms
        self.normalized_histograms = normalized_histograms
        print("Training ended, you can now look at the histograms with in the attribute .histograms and .normalized_histograms")

    # Method for testing the histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # dataset: the input dataset the network will respond to
    # labels: the labels used for the classification task (positive integers from 0 to N)
    # number_of_labels: the maximum number of labels of the dataset
    # method : string with the method you want to use :
    #          "CG" for Coniugate Gradient Descent
    #          "Exp distance" for the model with lateral inhibition and explicit
    #           exponential distance computing
    # noise_ratio: The model responses can be mixed with noise, it can improve  
    #               learning speed 
    # sparsity_coeff: The model presents lateral inhibtion between the basis,
    #                 it selects a winner depending on the responses, and feed 
    #                 back it's activity timed minus the sparsity_coeff to 
    #                 inhibit the neighbours. a value near one will result
    #                 in a Winner-Takes-All mechanism like on the original 
    #                 HOTS (For CG it's implemented as a weighted L1 coinstraint
    #                 on the activations in the Error function)
    # sensitivity : Each activation is computed as a guassian distance between 
    #               a base and the upcoming timesurface, this coefficient it's
    #               the gaussian 'variance' and it modulates cell selectivity
    #               to their encoded feature (Feature present only on Exp distance)   
    # 
    # It returns the computed distances per batch (Euclidean, normalized Euclidean
    # and Bhattacharyyan) and the predicted_labels per batch and per metric(distance)
    # =============================================================================      
    def histogram_classification_test(self, dataset, labels, number_of_labels, 
                                      method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        last_layer_activity = net_activity[-1]
        histograms = []
        normalized_histograms = []
        n_basis = self.basis_number[-1]
        # Normalization factor for building normalized histograms
        input_spikes_per_batch = np.zeros(len(dataset))
        for batch in range(len(dataset)):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for batch in range(len(dataset)):
            input_spikes_per_batch[batch] += len(dataset[batch][0])
            for sublayer in range(len(last_layer_activity)):
                batch_histogram = sum(last_layer_activity[sublayer][batch])
                histograms[batch][n_basis*sublayer:n_basis*(sublayer+1)] += batch_histogram               
        for batch in range(len(dataset)):
            normalized_histograms[batch] = histograms[batch]/input_spikes_per_batch[batch]
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
        
        return distances, predicted_labels, histograms, normalized_histograms

    
    # Method for plotting the basis set of a single sublayer
    # =============================================================================
    # layer : the index of the selected layer 
    # sublayer : the index of the selected sublayer
    # =============================================================================
    def plot_basis(self, layer, sublayer):
        for i in range(self.basis_number[layer]):
            plt.figure("Base N: "+str(i))
            sns.heatmap(self.basis[layer][sublayer][i])