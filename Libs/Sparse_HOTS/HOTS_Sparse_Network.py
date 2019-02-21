"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax

This file contains the HOTS_Sparse_Net class 

This Network offers the possibility to implement many iteration of original 
HOTS as proposed in the original paper here:
    https://ieeexplore.ieee.org/document/7508476
or in the Sparse HOTS paper here:
    https://arxiv.org/abs/1804.09236
as a new blend of them.

These techniques can then be used as a reference for other models as Variational
HOTS to experiment and benchmark.
  
 
"""
# General purpouse libraries
import numpy as np 
import keras
from scipy import optimize
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# Homemade Fresh Libraries like Gandma taught
from Libs.Sparse_HOTS.Time_Surface_generators import Time_Surface_event,Time_Surface_all
from Libs.Sparse_HOTS.HOTS_Sparse_Libs import error_func, error_func_deriv_a_j,\
update_basis_online, update_basis_online_hard_treshold, update_basis_offline_CG,\
events_from_activations, surface_live_plot, exp_decay, create_mlp

# Class for HOTS_Sparse_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, reconstruction and classification of the input data
# =============================================================================
class HOTS_Sparse_Net:


    # =============================================================================
    def __init__(self, features_number, surfaces_dimensions, taus, first_layer_polarities,
                 delay_coeff, net_seed = 0, verbose=False):
        """
        Network constructor settings, the net is set in a random state,
        with random basis and activations to define a starting point for the 
        optimization algorithms
        Treating ON and OFF events separately for each level generate 2 sublayers for 
        each sublayer in the previous stage. In the first layer we have a single sublayer
        (the layer itself), and since that the negative activations needs to be treated 
        independently, in the next layer we have to train 2 sublayers that will deal
        with both polarity, that will respectivelty generate 2 polarity events each.
        This critical structure carries the result that for a layer i, the number \
        of sublayers of this layer will be 2**i 
        
        Arguments:
            
                features_number (list of int) : it is a list containing the number of features extracted by each layer
                surfaces_dimensions (list of list of int) : A list containing per each layer the 
                                                           dimension of time surfaces in this format:
                                                           [xdim,ydim]
                taus (list of int) : It's a list containing the time coefficient used for the time
                                     surface creations for each layer
                first_layer_polarities (int) : The number of polarities that the first layer will
                                               have to handle, for ATIS cameras, it is usually,
                                               2 or 1 depending wether if you consider the additional                                                
                delay_coeff (list of int) : it's the cofficient named alfa used for generating an event in the paper
                net_seed (int) : seed used for net parameters generation,
                                 if set to 0 the process will be totally random
                verbose (bool) : If verbose is on, it will display graphs to assest the ongoing,
                                 learning algorithm (Per recording)
        
        """
        self.basis = []
        self.activations = []
        self.taus = taus
        self.layers = len(features_number)
        self.surfaces_dimensions = surfaces_dimensions
        self.features_number = features_number
        self.delay_coeff = delay_coeff
        self.verbose = verbose
        self.polarities = []
        self.polarities.append(first_layer_polarities)
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
        for layer, nfeatures in enumerate(features_number):
            #basis and activations of a single sublayer
            sublayers_basis = []
            sublayers_activations = []
            self.polarities.append(nfeatures)
            for sublayer in range(2**layer):
                #basis and activations of a single layer
                basis_set = []
                activations_set = []
                for j in range(nfeatures):
                    basis_set.append(rng.rand(surfaces_dimensions[layer][1], surfaces_dimensions[layer][0]*num_polarities))
                    basis_set[j][surfaces_dimensions[layer][1]//2, [surfaces_dimensions[layer][0]//2 + surfaces_dimensions[layer][0]*a  for a in range(num_polarities)]] = 1
                    #activations, or aj (as in the paper) are set randomly between -1 and 1
                    activations_set.append((rng.rand()-0.5)*2)
                sublayers_basis.append(np.array(basis_set))
                sublayers_activations.append(np.array(activations_set))
            self.basis.append(sublayers_basis)
            self.activations.append(sublayers_activations)
            num_polarities = nfeatures
    

    

    # =============================================================================        
    def learn_online(self, dataset, method="Exp distance", base_norm="Thresh", 
                     noise_ratio=[0,0,1], sparsity_coeff=[0,0,1], sensitivity=[0,0,1],
                     learning_rate=[0,0,1], base_norm_coeff=0): 
        """
        Method for online learning, essentially it performs a single step of gradient
        descent for each time surface, with a settable learning_rate. 
        The method is full online but it espects to work on an entire dataset
        at once to be more similiar in structure to its offline counterpart.
        This library has the porpouse to test various sparse HOTS implementations
        and to compare them, not to work in real time! (If it's worthy a real time
                                                        will hit the shelfs soon)
        Arguments:
            
            dataset (nested lists) : the initial dataset used for learning
            method (string) : string with the method you want to use :
                              "CG" for Coniugate Gradient Descent
                              "Exp distance" for the model with lateral inhibition and explicit
                              exponential distance computing
                              base_norm : Select whether use L2 norm or threshold constraints on basis
                              learning with the keywords "L2" "Thresh" 
        
        -----------------------------------------------------------------------------
        
        The following series of parameters can evolve during learning, thus 
        they are expected to be defined as lists with [begin_value, end_value, time_coeff]
        With float values
        Where begin_value is the value used for learning the first time surface,
        end_value is the value at steady state and time_coeff defines how quickly,
        the monotonic function describing the evolution of the parameter reaches it
        
        noise_ratio: The model responses can be mixed with noise, it can improve  
                           learning speed 
        sparsity_coeff: The model presents lateral inhibtion between the basis,
                         it selects a winner depending on the responses, and feed 
                         back it's activity timed minus the sparsity_coeff to 
                         inhibit the neighbours. a value near one will result
                         in a Winner-Takes-All mechanism like on the original 
                         HOTS (For CG it's implemented as a weighted L1 coinstraint
                         on the activations in the Error function)
        sensitivity : Each activation is computed as a guassian distance between 
                      a base and the upcoming timesurface, this coefficient it's
                      the gaussian 'variance' and it modulates cell selectivity
                      to their encoded feature (Feature present only on Exp distance)
        
        -----------------------------------------------------------------------------               
                       
        learning_rate (float) : It's the learning rate, what do you expect me to say ?                  
         
        base_norm_coeff (float) : l2 regularization coefficient for basis (Not used if
                                                                           base_norm="Thresh")
        """
        # The list of all data processed per layer devided for sublayer 
        # input_data[sublayer][actual_recording][events, timestamp if 0 or xy coordinates if 1]
        input_data = []
        # In the first layer the input_data is the input dataset given by the user
        input_data.append(dataset)

        # Preallocating evolution methods, for plotting the network evolution.
        self.evolution_pre_allocation(len(dataset))
        
        for layer in range(self.layers):
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as input for the 
            # next one
            outrecordings = []
            
            for sublayer in range(len(self.basis[layer])):
              
                # Create network evolution parameters, at the moment I'm using 
                # an exponential decay kernel as a function, it can be changed in the code pretty
                # easly
                
                # end_time:  how many time surfaces will be generated in the learing 
                # process for this sublayer
                end_time = 0
                for i in range(len(input_data[sublayer])):
                    end_time += len(input_data[sublayer][i][0])
                time = np.arange(end_time)
                
                noise_evolution = exp_decay(noise_ratio[0], noise_ratio[1], noise_ratio[2], time)
                learning_evolution = exp_decay(learning_rate[0], learning_rate[1], learning_rate[2], time)
                sparsity_evolution = exp_decay(sparsity_coeff[0], sparsity_coeff[1], sparsity_coeff[2], time)                
                sensitivity_evolution = exp_decay(sensitivity[0], sensitivity[1], sensitivity[2], time)
                
                # Generate and allocate plots for online plotting
                if self.verbose is True:
                    # Building a list to contain the basis at each run and the original timesurface (the whole picture)
                    # at the position 0
                    timesurfaces_list = [Time_Surface_all(xdim=35, ydim=35, timestamp=dataset[0][0][-1], timecoeff=self.taus[0], dataset=dataset[0], num_polarities=1, minv=0.1, verbose=False)]
                    plot_names = ["Original Time Surface"]
                    for i in range(len(self.basis[layer][sublayer])):
                        timesurfaces_list.append(self.basis[layer][sublayer][i])
                        plot_names.append("Base N: "+str(i))
                    live_figures, live_axes = surface_live_plot(surfaces=timesurfaces_list, fig_names=plot_names)


                    
                # Counter of surfaces that will be used to update evolution parameters
                time = 0
                sub_layer_surfaces = []
                sub_layer_reference_events = []
                sub_layer_errors = []
           
                for recording in range(len(input_data[sublayer])):
                    recording_surfaces = []
                    recording_reference_events = []
                    for k in range(len(input_data[sublayer][recording][0])):
                        single_event = [input_data[sublayer][recording][0][k], input_data[sublayer][recording][1][k]]
                        tsurface = Time_Surface_event(xdim = self.surfaces_dimensions[layer][0],
                                                    ydim = self.surfaces_dimensions[layer][1],
                                                    event=single_event,
                                                    timecoeff=self.taus[layer],
                                                    dataset=input_data[sublayer][recording],
                                                    num_polarities=self.polarities[layer],
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
                        if method=="Dot product":
                            sub_layer_errors.append(self.sublayer_response_dot_product(layer, sublayer,
                                                                                        tsurface,
                                                                                        sparsity_evolution[time], 
                                                                                        noise_evolution[time], 
                                                                                        sensitivity_evolution[time]))
                        recording_surfaces.append(tsurface)
                        recording_reference_events.append(single_event)
                        # Run a single step of the gradient descent
                        if base_norm=="L2":
                            update_basis_online(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                                    learning_evolution[time], tsurface, base_norm_coeff, 1, 0)
                        if base_norm=="Thresh":
                            update_basis_online_hard_treshold(self.basis[layer][sublayer], self.activations[layer][sublayer],
                                                              learning_evolution[time], tsurface, 1, 0)
                        
                        time += 1
                    if time!=0:                        
                        self.evolution_save_point(layer, sublayer, recording, noise_evolution[time-1], sparsity_evolution[time-1], sensitivity_evolution[time-1], learning_evolution[time-1])
                    # Update plots
                    if self.verbose is True:    
                        timesurfaces_list[0]=Time_Surface_all(xdim=35, ydim=35, timestamp=dataset[recording][0][-1], timecoeff=self.taus[0], dataset=dataset[recording], num_polarities=1, minv=0.1, verbose=False)
                        for i in range(len(self.basis[layer][sublayer])):
                            timesurfaces_list[i+1]=self.basis[layer][sublayer][i]
                        live_figures, live_axes = surface_live_plot(surfaces=timesurfaces_list, figures=live_figures, axes=live_axes)
                        print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  Base optimization process per recording:"+np.str(((recording+1)/len(input_data[sublayer]))*100)+"%")
                    sub_layer_surfaces.append(recording_surfaces)
                    sub_layer_reference_events.append(recording_reference_events)
                # Close plots before displaying next set of basis
                if self.verbose is True:
                    for i in range(len(live_figures)):
                        plt.close(fig=live_figures[i])
                    live_axes.clear()
                    live_figures.clear()
                        
                    
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
                    for recording in range(len(input_data[sublayer])):
                        recording_activations = []
                        for k in range(len(input_data[sublayer][recording][0])):
                            if method=="CG" :
                                self.sublayer_response_CG(layer, sublayer,
                                                          sub_layer_surfaces[recording][k],
                                                          sparsity_evolution[time], 
                                                          noise_evolution[time])
                            if method=="Exp distance" :
                                self.sublayer_response_exp_distance(layer, sublayer,
                                                                    sub_layer_surfaces[recording][k],
                                                                    sparsity_evolution[time], 
                                                                    noise_evolution[time], 
                                                                    sensitivity_evolution[time])
                            if method=="Dot product":
                                sub_layer_errors.append(self.sublayer_response_dot_product(layer, sublayer,
                                                                                           sub_layer_surfaces[recording][k],
                                                                                           sparsity_evolution[time], 
                                                                                           noise_evolution[time], 
                                                                                           sensitivity_evolution[time]))
                            recording_activations.append(self.activations[layer][sublayer].copy())

                        # Obtaining the events resulting from the activations in a single recording
                        outevents = events_from_activations(recording_activations,
                                                            sub_layer_reference_events[recording],
                                                             self.delay_coeff)
                        
                        sub_layer_outevents_on.append(outevents[0])
                        sub_layer_outevents_off.append(outevents[1])
                        
                    
                    # They will be used to optimize the consequent of sublayers
                    # in the next layer
                    outrecordings.append(sub_layer_outevents_on)       
                    outrecordings.append(sub_layer_outevents_off)       
                    
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
            input_data = outrecordings.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)

        

    # =============================================================================      
    def learn_offline(self, dataset, sparsity_coeff, learning_rate,
                      max_steps, base_norm_coeff, precision):
        """
        Method for offline learning, derived from Olshausen and Field 1996
        link : https://www.sciencedirect.com/science/article/pii/S0042698997001697
        Like in the original work, first it computes the sparse response a_j, but then
        it performs cognugate gradient descent on the all recordings at the same time
        to find the optimal basis. This process is then iterated until it reaches the 
        expected precision or the max number of steps
        Advice : It is slow as 2001: A Space Odyssey, but can be considered a good 
        groundtruth for comparisons
        Arguments:
            
            dataset (nested lists) : the initial dataset used for learning
                            
            sparsity_coeff (float) : reported as lambda in the Sparse Hots paper, is the norm
                            coefficient used to impose sparseness on the net activations aj
            learning_rate (float) : a list of the learning rates for basis learning for each                                
                                    training phase implement 
            max_steps (int) : The number of maximum steps performed for basis learning.
            base_norm_coeff (float) : l2 regularization coefficient for basis
            precision (float) : the expected precision of the algorithm (not
                                a quick enough convergence the alhorithm will stop when 
                                reaching the max number of steps)
        """
        # The list of all data processed per layer devided for sublayer 
        # input_data[sublayer][actual_recording][events, timestamp if 0 or xy coordinates if 1]
        input_data = []
        # In the first layer the input_data is the input dataset given by the user
        input_data.append(dataset)
        
        for layer in range(self.layers):
            # now let's generate the surfaces and train the layers
            single_layer_surfaces = []
            single_layer_errors = []
            # The result of a single layer learning that will serve as input for the 
            # next one
            outrecordings = []
            for sublayer in range(len(self.basis[layer])):
                sub_layer_surfaces = []
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                # A single array containing all the generated surfaces 
                # for a single sublayer used for full offline learning 
                all_surfaces = []
                all_reference_events = []
                # Generate and allocate plots for online plotting
                if self.verbose is True:
                    plot_names = []
                    # Building a list to contain the basis at each run
                    for i in range(len(self.basis[layer][sublayer])):
                        plot_names.append("Base N: "+str(i))
                    live_figures, live_axes = surface_live_plot(surfaces=self.basis[layer][sublayer], fig_names=plot_names)
                for recording in range(len(input_data[sublayer])):
                    recording_surfaces = []
                    for k in range(len(input_data[sublayer][recording][0])):
                        single_event = [input_data[sublayer][recording][0][k], input_data[sublayer][recording][1][k]]
                        tsurface = Time_Surface_event(xdim = self.surfaces_dimensions[layer][0],
                                                      ydim = self.surfaces_dimensions[layer][1],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=input_data[sublayer][recording],
                                                      num_polarities=self.polarities[layer],
                                                      verbose=False)
                        recording_surfaces.append(tsurface)
                        # A single array containing all the generated surfaces 
                        # for a single sublayer used for full recording learning
                        all_surfaces.append(tsurface) 
                        all_reference_events.append(single_event)
                    sub_layer_surfaces.append(recording_surfaces)
                
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
                                                      self.surfaces_dimensions[layer][0]*self.polarities[layer],
                                                      self.surfaces_dimensions[layer][1], 
                                                      self.features_number[layer], sub_layer_activations,
                                                      all_surfaces, base_norm_coeff)
                    
                    self.basis[layer][sublayer] = opt_res.x.reshape(self.features_number[layer],
                              self.surfaces_dimensions[layer][1], self.surfaces_dimensions[layer][0]*self.polarities[layer])
                    sub_layer_errors.append(opt_res.fun)
                    n_steps += 1
                    previous_err = err
                    err = sub_layer_errors[-1]
                    # Update plots
                    if self.verbose is True:    
                        live_figures, live_axes = surface_live_plot(surfaces=self.basis[layer][sublayer], figures=live_figures, axes=live_axes)
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
                # of different recordings, in order to build the next layer input data 
                # I need to redefine the original ordering of the data
                current_position = 0
                for recording in range(len(input_data[sublayer])):
                    current_recording_length = len(input_data[sublayer][recording][0])       
                    # Obtaining the events resulting from the activations in a single recording
                    outevents = events_from_activations(sub_layer_activations[current_position:(current_position+current_recording_length)],
                                                        all_reference_events[current_position:(current_position+current_recording_length)],
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    current_position += current_recording_length
                        
                single_layer_surfaces.append(sub_layer_surfaces)
                single_layer_errors.append(sub_layer_errors)
                # Saving each sub layer events deviding between negative and positive
                # activations, they will be used to optimize the consequent of 
                # sublayers in the next layer
                outrecordings.append(sub_layer_outevents_on)       
                outrecordings.append(sub_layer_outevents_off)
                # Close plots before displaying next set of basis
                if self.verbose is True:
                    for i in range(len(live_figures)):
                        plt.close(fig=live_figures[i])
                    live_axes.clear()
                    live_figures.clear()
            input_data = outrecordings.copy()
            self.surfaces.append(single_layer_surfaces)
            self.errors.append(single_layer_errors)


    # =============================================================================  
    def evolution_pre_allocation(self, dataset_length):
        """
        Method that preallocates lists that will store evolution save points
        Arguments:
            dataset_length (int) : the lenght of the entire dataset in number of recordings
        """
        #it will store the intial basis too, thus +1 on the dataset length 
        self.basis_history=[[[np.zeros(self.basis[layer][sublayer].shape) for recording in range(dataset_length+1)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        for layer in range(self.layers):
            for sublayer in range(2**layer):
                self.basis_history[layer][sublayer][0]=self.basis[layer][sublayer]
        self.noise_history=[[[0 for recording in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        self.sparsity_history=[[[0 for recording in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        self.sensitivity_history=[[[0 for recording in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        self.learning_history=[[[0 for recording in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        self.basis_distance=[[[np.zeros(len(self.basis[layer][sublayer])) for recording in range(dataset_length)] for sublayer in range(2**layer)] for layer in range(self.layers)]
        
                

    # =============================================================================  
    def evolution_save_point(self, layer, sublayer, recording, noise, sparsity, sensitivity, learning):
        """
        Method useful to save network status in relation of its previous history
        in this way is possible to test evolution parameters and test wether an 
        online method is overfitting in a particular recording.
        In the future the outlook of the online method will be to implement overparameters
        to control the network in a fine and less supervised manner.
        Arguments:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer 
            recording (int) : the index of the recording in the dataset
            noise (float) : the noise coefficient at the saving time.
            sparsity (float) : the sparsity coefficient at the saving time.
            sensitivity (float) : the sensitivity coefficient at the saving time.
            learning (float) : the learning rate at the saving time.
               
        # It saves the set of basis, and the status of the evolution parameters.
        """
        self.basis_history[layer][sublayer][recording+1]=self.basis[layer][sublayer].copy()
        self.noise_history[layer][sublayer][recording]=noise
        self.sparsity_history[layer][sublayer][recording]=sparsity
        self.sensitivity_history[layer][sublayer][recording]=sensitivity
        self.learning_history[layer][sublayer][recording]=learning
        distance = np.zeros(len(self.basis[layer][sublayer]))
        for i,feature in enumerate(self.basis[layer][sublayer]):
            distance[i] = np.sum(self.basis_history[layer][sublayer][recording][i,:,:]-feature)**2
        self.basis_distance[layer][sublayer][recording]=distance
                


    # =============================================================================  
    def sublayer_response_CG(self, layer, sublayer, timesurface, sparsity_coeff, noise_coeff):
        """
        Method for computing and updating the network activations for a single time surface
        using conjugate gradient descent
        Arguments: 
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer 
            timesurface (numpy matrix) : the input time surface
            sparsity_coeff (float) : reported as lambda in the Sparse Hots paper, is the norm
                                coefficient used to impose sparseness on the net activations aj
            noise_coeff (float) : coefficient use to add noise to the net activities
        Returns:
            res.fun (float) : the error of reconstruction
        """
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
    
    
    
    # =============================================================================  
    def sublayer_response_exp_distance(self, layer, sublayer, timesurface, sparsity_coeff, noise_coeff, sensitivity):
        """
        Method for computing and updating the network activations for a single time surface
        using exponential distance between the surface and the set of features
        Arguments: 
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer 
            timesurface (numpy matrix) : the input time surface
            sparsity_coeff (float) : It is a value moltiplicating the activity of the best 
                                     fitting feature. The result will be subtracted to the other
                                     activities de facto implementing a simple lateral inhibition
            noise_coeff (float) : coefficient use to add noise to the net activities
            sensitivity (float) : it shapes the sensitivty for each feature in computing the 
                                  activities of each feature
        Returns:
            error (float) : The error of the loss function
        """
        euclidean_distances = []
        for i, feat in enumerate(self.basis[layer][sublayer]):
            euclidean_distances.append(np.sum(np.abs(feat-timesurface)**2))
        activations = np.exp(-sensitivity*np.array(euclidean_distances))
        # Here I mix noise with the activations
        activations = activations*(1-noise_coeff) + (noise_coeff)*np.exp(-np.random.rand(len(activations)))
        # Here I implement the lateral inhibition
        winner_ind=np.argmax(activations)
        losers_ind=np.arange(len(activations))!=winner_ind
        activations[losers_ind] = activations[losers_ind]-sparsity_coeff*activations[winner_ind]
        # I kill the basis too inhibited (with negative values) and update the activities
        activations[activations<=0] = 0
        self.activations[layer][sublayer] = activations
#        print(activations)
        # Here i compute the error as the euclidean distance between a reconstruced
        # Time surface and the original one
        S_tilde = sum([a*b for a,b in zip(self.basis[layer][sublayer], activations)])
        residue = timesurface-S_tilde
        error =  np.sum(residue**2)
        return error
    

    # =============================================================================  
    def sublayer_response_dot_product(self, layer, sublayer, timesurface, sparsity_coeff, noise_coeff, sensitivity):
        """
        Method for computing and updating the network activations for a single time surface
        using exponential distance between the surface and the set of features
        Attributes:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer 
            timesurface (numpy matrix) : the input time surface
            sparsity_coeff (float) : It is a value moltiplicating the activity of the best 
                                     fitting feature. The result will be subtracted to the other
                                     activities de facto implementing a simple lateral inhibition
            noise_coeff (float) : coefficient use to add noise to the net activities
            sensitivity (float) : it shapes the sensitivty for each feature in computing the 
                                  activities of each feature
        Returns:
            error (float) : The error of the loss function
        """
        projections = []
        norm_timesurface = np.sqrt(np.sum(timesurface**2))
        for i, feat in enumerate(self.basis[layer][sublayer]):
            norm_feat = np.sqrt(np.sum(feat**2))
            projections.append(np.sum(np.multiply(feat,timesurface))/(norm_feat*norm_timesurface))
        activations = np.array(projections)
        # Here I mix noise with the activations
        activations = activations*(1-noise_coeff) + (noise_coeff)*np.exp(-np.random.rand(len(activations)))
        # Here I implement the lateral inhibition
        winner_ind=np.argmax(activations)
        losers_ind= np.arange(len(activations))!=winner_ind
        activations[losers_ind] = activations[losers_ind]-sparsity_coeff*activations[winner_ind]
        # I kill the basis too inhibited (with negative values) and update the activities
        activations[activations<=0] = 0
        self.activations[layer][sublayer] = activations
#        print(activations)
        # Here i compute the error as the euclidean distance between a reconstruced
        # Time surface and the original one
        S_tilde = sum([a*b for a,b in zip(self.basis[layer][sublayer], activations)])
        residue = timesurface-S_tilde
        error =  np.sum(residue**2)
        return error
    
    # =============================================================================  
    def full_net_dataset_response(self, dataset, method="Exp distance", noise_ratio=0, sparsity_coeff=0,
                     sensitivity=0):
        """
        Method for computing the network response for a whole dataset
        Arguments:
            
            dataset (nested lists) : the initial dataset used for learning
            method (string) : string with the method you want to use :
                              "CG" for Coniugate Gradient Descent
                              "Exp distance" for the model with lateral inhibition and explicit
                              exponential distance computing
                              base_norm : Select whether use L2 norm or threshold constraints on basis
                              learning with the keywords "L2" "Thresh" 
        
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                           learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                             it selects a winner depending on the responses, and feed 
                             back it's activity timed minus the sparsity_coeff to 
                             inhibit the neighbours. a value near one will result
                             in a Winner-Takes-All mechanism like on the original 
                             HOTS (For CG it's implemented as a weighted L1 coinstraint
                             on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance)
                                  
        Returns:
            full_net_activations (nested lists) : the whole history of the network activity 
        """
        # The list of all data processed per layer devided for sublayer 
        # input_data[sublayer][actual_recording][events, timestamp if 0 or xy coordinates if 1]
        input_data = []
        # In the first layer the input_data is the input dataset given by the user
        input_data.append(dataset)
        
        full_net_activations = []
        
        for layer in range(self.layers):
            outrecordings = []
            layer_activations = []
            for sublayer in range(len(self.basis[layer])):
                # Counter of surfaces that will be used to update evolution parameters
                sub_layer_errors = []
                sub_layer_outevents_on = []
                sub_layer_outevents_off = []
                sub_layer_activations = []
                for recording in range(len(input_data[sublayer])):
                    recording_surfaces = []
                    recording_reference_events = []
                    recording_activations = []
                    for k in range(len(input_data[sublayer][recording][0])):
                        single_event = [input_data[sublayer][recording][0][k], input_data[sublayer][recording][1][k]]
                        tsurface = Time_Surface_event(xdim = self.surfaces_dimensions[layer][0],
                                                      ydim = self.surfaces_dimensions[layer][1],
                                                      event=single_event,
                                                      timecoeff=self.taus[layer],
                                                      dataset=input_data[sublayer][recording],
                                                      num_polarities=self.polarities[layer],
                                                      verbose=False)
                        recording_surfaces.append(tsurface)
                        recording_reference_events.append(single_event)
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
                        if method=="Dot product":
                            sub_layer_errors.append(self.sublayer_response_dot_product(layer, sublayer,
                                                                                       tsurface,
                                                                                       sparsity_coeff, 
                                                                                       noise_ratio, 
                                                                                       sensitivity))
                        
                        
                        recording_activations.append(self.activations[layer][sublayer])
                    # Obtaining the events resulting from the activations in a single recording
                    outevents = events_from_activations(recording_activations,
                                                        recording_reference_events,
                                                         self.delay_coeff)
                    sub_layer_outevents_on.append(outevents[0])
                    sub_layer_outevents_off.append(outevents[1])
                    sub_layer_activations.append(recording_activations)
                    if self.verbose is True:
                        print("Layer:"+np.str(layer)+"  Sublayer:"+np.str(sublayer)+"  recording processing:"+np.str(((recording+1)/len(input_data[sublayer]))*100)+"%")
                layer_activations.append(sub_layer_activations)
                outrecordings.append(sub_layer_outevents_on)       
                outrecordings.append(sub_layer_outevents_off)
            full_net_activations.append(layer_activations)
            input_data = outrecordings.copy()

        return full_net_activations
    
    

    # =============================================================================  
    def sublayer_reconstruct(self, layer, sublayer, timesurface, method,
                             noise_ratio, sparsity_coeff, sensitivity):
        """
        Method for plotting reconstruction heatmap and reconstruction error
        Arguments:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer     
            timesurface (numpy matrix) : the imput time surface
            method (string) : string with the method you want to use :
                         "CG" for Coniugate Gradient Descent
                         "Exp distance" for the model with lateral inhibition and explicit
                          exponential distance computing
            
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                                  learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                                it selects a winner depending on the responses, and feed 
                                back it's activity timed minus the sparsity_coeff to 
                                inhibit the neighbours. a value near one will result
                                in a Winner-Takes-All mechanism like on the original 
                                HOTS (For CG it's implemented as a weighted L1 coinstraint
                                on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance) 
        Returns:
            error (float) : The error of reconstruction
        """
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
        if method=="Dot product":
            error = self.sublayer_response_dot_product(layer, sublayer,
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
    

    # =============================================================================  
    def batch_sublayer_reconstruct_error(self, layer, sublayer, timesurfaces,
                                         method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        """
        Method for computing the reconstruction error of a sublayer set of basis
        for an entire recording of time surfaces
        
                Method for plotting reconstruction heatmap and reconstruction error
        Arguments:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer     
            timesurfaces (list of numpy matrices) : the imput time surface
            method (string) : string with the method you want to use :
                         "CG" for Coniugate Gradient Descent
                         "Exp distance" for the model with lateral inhibition and explicit
                          exponential distance computing
            
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                                  learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                                it selects a winner depending on the responses, and feed 
                                back it's activity timed minus the sparsity_coeff to 
                                inhibit the neighbours. a value near one will result
                                in a Winner-Takes-All mechanism like on the original 
                                HOTS (For CG it's implemented as a weighted L1 coinstraint
                                on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance) 
       Returns:
            error (float) : The mean error of reconstruction
        
        """
        error = 0
        total_surf = 0
        for i in range(len(timesurfaces)):
            for k in range(len(timesurfaces[i])):
                if method == "CG":
                    error += self.sublayer_response_CG(layer, sublayer,
                                                      timesurfaces[i][k],
                                                      sparsity_coeff,
                                                      noise_ratio)
                if method == "Exp distance": 
                    error += self.sublayer_response_exp_distance(layer, sublayer,
                                                                timesurfaces[i][k],
                                                                sparsity_coeff, 
                                                                noise_ratio, 
                                                                sensitivity)
                if method=="Dot product":
                    error += self.sublayer_response_dot_product(layer, sublayer,
                                                               timesurfaces[i][k],
                                                               sparsity_coeff, 
                                                               noise_ratio, 
                                                               sensitivity)
                total_surf += 1
        return error/total_surf

    # Method for training a mlp classification model NB can work only on a single sublayer (Thus with rectified outputs)
    # =============================================================================      
    def mlp_classification_train(self, dataset, labels, number_of_labels, learning_rate, method,
                                 noise_ratio, sparsity_coeff, sensitivity):
        """
        Method to train a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            dataset (nested lists) : the dataset used for learn classification.
            labels (list of int) : List of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
            learning_rate (float) : The method is Adam
            
            method (string) : string with the method you want to use to compute activations:
                         "CG" for Coniugate Gradient Descent
                         "Exp distance" for the model with lateral inhibition and explicit
                          exponential distance computing
            
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                                  learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                                it selects a winner depending on the responses, and feed 
                                back it's activity timed minus the sparsity_coeff to 
                                inhibit the neighbours. a value near one will result
                                in a Winner-Takes-All mechanism like on the original 
                                HOTS (For CG it's implemented as a weighted L1 coinstraint
                                on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance) 


        """
        
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        last_layer_activity = net_activity[-1][0]            

        
        n_events_per_recording = [len(last_layer_activity[recording]) for recording in range(len(labels))]
        concatenated_last_layer_activity = np.concatenate(last_layer_activity)
        processed_labels = np.concatenate([labels[recording]*np.ones(n_events_per_recording[recording]) for recording in range(len(labels))])
        processed_labels = keras.utils.to_categorical(processed_labels, num_classes = number_of_labels)
        input_size = self.features_number[-1]
        self.mlp = create_mlp(input_size=input_size, hidden_size=20, output_size=number_of_labels, 
                              learning_rate=learning_rate)
        self.mlp.summary()
        self.mlp.fit(concatenated_last_layer_activity, processed_labels,
          epochs=50,
          batch_size=125)
        return processed_labels
        if self.verbose is True:
            print("Training ended, you can now access the trained network with the method .mlp")

    # Method for testing the mlp classification model (Thus with rectified outputs)
    # =============================================================================      
    def mlp_classification_test(self, dataset, labels, number_of_labels, method,
                                 noise_ratio, sparsity_coeff, sensitivity):
        """
        Method to test a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            dataset (nested lists) : the dataset used for learn classification.
            labels (list of int) : List of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
           
            method (string) : string with the method you want to use to compute activations:
                         "CG" for Coniugate Gradient Descent
                         "Exp distance" for the model with lateral inhibition and explicit
                          exponential distance computing
            
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                                  learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                                it selects a winner depending on the responses, and feed 
                                back it's activity timed minus the sparsity_coeff to 
                                inhibit the neighbours. a value near one will result
                                in a Winner-Takes-All mechanism like on the original 
                                HOTS (For CG it's implemented as a weighted L1 coinstraint
                                on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance) 


        """
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        last_layer_activity = net_activity[-1][0]                
        concatenated_last_layer_activity = np.concatenate(last_layer_activity)
        predicted_labels_ev=self.mlp.predict(concatenated_last_layer_activity,batch_size=125)
        counter=0
        predicted_labels=[]
        for recording in range(len(last_layer_activity)):
            activity_sum = sum(predicted_labels_ev[counter:counter+len(last_layer_activity[recording])])
            predicted_labels.append(np.argmax(activity_sum))
            counter += len(last_layer_activity[recording])
        prediction_rate=0
        for i,true_label in enumerate(labels):
            prediction_rate += (predicted_labels[i] == true_label)/len(labels)
        return prediction_rate, predicted_labels, predicted_labels_ev
    
    

    # =============================================================================
    def plot_basis(self, layer, sublayer):
        """
        Method for plotting the basis set of a single sublayer
        Arguments:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer
        """
        for i in range(self.features_number[layer]):
            plt.figure("Base N: "+str(i))
            sns.heatmap(self.basis[layer][sublayer][i])
            

    # =============================================================================  
    def plot_evolution(self, layer, sublayer, figures=[], axes=[]):
        """
        Method for printing the evolution of the network after online learning   
        Arguments:
            layer (int) : the index of the selected layer 
            sublayer (int) : the index of the selected sublayer 
            figures (list of matplotlib figures) : is figure is empty, the function will generate a new set of them,
                                                   otherwise the new axes will be printed on the old ones to update them
            axes (list of matplotlib axes) : the axes of the previous figures, if not provided the function will generate
                                             a new set of them
        Returns:
            figures (list of matplotlib figures) : If needed they can be updated
            axes (list of matplotlib axes) : If needed they can be updated
        """
        if figures==[]:
            figures.append(plt.figure("Basis evolution per recording (euclidean distance between sets of basis)"))
            distance_legend = ["Base N: "+str(i) for i in range(len(self.basis[layer][sublayer]))]
            axes.append(figures[0].add_subplot('111'))
            axes[0].plot(self.basis_distance[layer][sublayer])
            axes[0].legend(tuple(distance_legend))
            axes[0].set_xlabel("Recording Number")
            
            evolution_parameters=np.array([self.noise_history[layer][sublayer], 
                                           self.learning_history[layer][sublayer],
                                           self.sparsity_history[layer][sublayer],
                                           self.sensitivity_history[layer][sublayer]]).transpose()
            evolution_legend = ("Noise","Learning step","Sparsity","Sensitivity")
            figures.append(plt.figure("Network parameter evolution per recording"))
            axes.append(figures[1].add_subplot('111'))
            axes[1].plot(evolution_parameters)
            axes[1].legend(tuple(evolution_legend))
            axes[1].set_xlabel("Recording Number")
        else:
            
            axes[0].clear()
            distance_legend = ["Base N: "+str(i) for i in range(len(self.basis[layer][sublayer]))]                  
            axes[0].plot(self.basis_distance[layer][sublayer])
            axes[0].legend(tuple(distance_legend))
            axes[0].set_xlabel("Recording Number")
            
            axes[1].clear()
            evolution_parameters=np.array([self.noise_history[layer][sublayer], 
                                           self.learning_history[layer][sublayer],
                                           self.sparsity_history[layer][sublayer],
                                           self.sensitivity_history[layer][sublayer]]).transpose()
            evolution_legend = ("Noise","Learning step","Sparsity","Sensitivity")
            axes[1].plot(evolution_parameters)
            axes[1].legend(tuple(evolution_legend))
            axes[1].set_xlabel("Recording Number")
        return figures, axes



    # =============================================================================      
    def histogram_classification_train(self, dataset, labels, number_of_labels,
                                       method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        """
        Method for training an histogram classification model as proposed in 
        HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
        Arguments:
            dataset (nested lists) : the initial dataset used for learning
            labels (list of int) : the labels used for the classification task (from 0 to number_of_labels-1)
            number_of_labels (int) : the maximum number of labels of the dataset
            method (string) : string with the method you want to use :
                              "CG" for Coniugate Gradient Descent
                              "Exp distance" for the model with lateral inhibition and explicit
                              exponential distance computing
                              base_norm : Select whether use L2 norm or threshold constraints on basis
                              learning with the keywords "L2" "Thresh" 
        
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                           learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                             it selects a winner depending on the responses, and feed 
                             back it's activity timed minus the sparsity_coeff to 
                             inhibit the neighbours. a value near one will result
                             in a Winner-Takes-All mechanism like on the original 
                             HOTS (For CG it's implemented as a weighted L1 coinstraint
                             on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance)
        """
       
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        
        
        last_layer_activity = net_activity[-1]        
        histograms = []
        normalized_histograms = []
        n_basis = self.features_number[-1]
        # Normalization factor for building normalized histograms
        input_spikes_per_label = np.zeros(number_of_labels)
        recording_per_label  = np.zeros(number_of_labels)
        # this array it's used to count the number of recordings with a certain label,
        # the values will be then used to compute mean histograms
        for label in range(number_of_labels):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for recording in range(len(dataset)):
            current_label = labels[recording]
            input_spikes_per_label[current_label] += len(dataset[recording][0])
            recording_per_label[current_label] += 1            
            for sublayer in range(len(last_layer_activity)):
                recording_histogram = sum(last_layer_activity[sublayer][recording])
                histograms[current_label][n_basis*sublayer:n_basis*(sublayer+1)] += recording_histogram               
        for label in range(number_of_labels):
            normalized_histograms[label] = histograms[label]/input_spikes_per_label[label]
            histograms[label] = histograms[label]/recording_per_label[label]
        self.histograms = histograms
        self.normalized_histograms = normalized_histograms
        if self.verbose is True:
            print("Training ended, you can now look at the histograms with in the "+
                  "attribute .histograms and .normalized_histograms, or using "+
                  "the .plot_histograms method")


    # =============================================================================      
    def histogram_classification_test(self, dataset, labels, number_of_labels, 
                                      method, noise_ratio, sparsity_coeff,
                                         sensitivity):
        """
        Method for training an histogram classification model as proposed in 
        HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
        Arguments:
            dataset (nested lists) : the initial dataset used for learning
            labels (list of int) : the labels used for the classification task (from 0 to number_of_labels-1)
            number_of_labels (int) : the maximum number of labels of the dataset
            method (string) : string with the method you want to use :
                              "CG" for Coniugate Gradient Descent
                              "Exp distance" for the model with lateral inhibition and explicit
                              exponential distance computing
                              base_norm : Select whether use L2 norm or threshold constraints on basis
                              learning with the keywords "L2" "Thresh" 
        
            noise_ratio (float) : The model responses can be mixed with noise, it can improve  
                           learning speed 
            sparsity_coeff (float) : The model presents lateral inhibtion between the basis,
                             it selects a winner depending on the responses, and feed 
                             back it's activity timed minus the sparsity_coeff to 
                             inhibit the neighbours. a value near one will result
                             in a Winner-Takes-All mechanism like on the original 
                             HOTS (For CG it's implemented as a weighted L1 coinstraint
                             on the activations in the Error function)
            sensitivity (float) : Each activation is computed as a guassian distance between 
                                  a base and the upcoming timesurface, this coefficient it's
                                  the gaussian 'variance' and it modulates cell selectivity
                                  to their encoded feature (Feature present only on Exp distance)
        Returns:
            prediction_rates (list of float) : The prediction rates in decimal for 
                                               euclidean distance, normalised 
                                               euclidean disrance and bhattarchaya
                                               distance
            distances (nested lists of floats): The set of distances per each class 
                                                of the test dataset and the computed signatures
                                                
            predicted_labels (list of int) : The list of predicted labels 
        """
        net_activity = self.full_net_dataset_response(dataset, method, 
                                                      noise_ratio, 
                                                      sparsity_coeff,
                                                      sensitivity)
        last_layer_activity = net_activity[-1]
        histograms = []
        normalized_histograms = []
        n_basis = self.features_number[-1]
        # Normalization factor for building normalized histograms
        input_spikes_per_recording = np.zeros(len(dataset))
        for recording in range(len(dataset)):
            histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
            normalized_histograms.append(np.zeros(n_basis*(2**(self.layers-1))))
        for recording in range(len(dataset)):
            input_spikes_per_recording[recording] += len(dataset[recording][0])
            for sublayer in range(len(last_layer_activity)):
                recording_histogram = sum(last_layer_activity[sublayer][recording])
                histograms[recording][n_basis*sublayer:n_basis*(sublayer+1)] += recording_histogram               
        for recording in range(len(dataset)):
            normalized_histograms[recording] = histograms[recording]/input_spikes_per_recording[recording]
        # compute the distances per each histogram from the models
        distances = []
        predicted_labels = []
        for recording in range(len(dataset)):
            single_recording_distances = []
            for label in range(number_of_labels):
                single_label_distances = []  
                single_label_distances.append(distance.euclidean(histograms[recording],self.histograms[label]))
                single_label_distances.append(distance.euclidean(normalized_histograms[recording],self.normalized_histograms[label]))
                Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(normalized_histograms[recording], self.normalized_histograms[label])]) 
                single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
                single_recording_distances.append(single_label_distances)
            single_recording_distances = np.array(single_recording_distances)
            single_recording_predicted_labels = np.argmin(single_recording_distances, 0)
            distances.append(single_recording_distances)
            predicted_labels.append(single_recording_predicted_labels)
        self.test_histograms = histograms
        self.test_normalized_histograms = normalized_histograms    
        # Computing the results
        eucl = 0
        norm_eucl = 0
        bhatta = 0
        for recording,true_label in enumerate(labels):
            eucl += (predicted_labels[recording][0] == true_label)/len(labels)
            norm_eucl += (predicted_labels[recording][1] == true_label)/len(labels)
            bhatta += (predicted_labels[recording][2] == true_label)/len(labels)
        prediction_rates = [eucl, norm_eucl, bhatta]
        if self.verbose is True:
            print("Testing ended, you can also look at the test histograms with in"+
                  " the attribute .test_histograms and .test_normalized_histograms, "+
                  "or using the .plot_histograms method")
        return prediction_rates, distances, predicted_labels

    # =============================================================================
    def plot_histograms(self, label_names, labels=[]):
        """
        Method for plotting the histograms of the network, either result of train 
        or testing
        Arguments:
            label_names (tuple of strings) : the names of each label that will displayed
                                             in the legend
            labels (list of int) : the labels of the test dataset used to generate
                                   the histograms, if empty, the function will plot 
                                   instead the class histograms computed using 
                                   .histogram_classification_train
        """
        if labels == []:
            hist = np.transpose(self.histograms)
            norm_hist = np.transpose(self.normalized_histograms)
            eucl_fig, eucl_ax = plt.subplots()
            eucl_ax.set_title("Train histogram based on euclidean distance")
            eucl_ax.plot(hist)
            eucl_ax.legend(label_names)

            norm_fig, norm_ax = plt.subplots()
            norm_ax.set_title("Train histogram based on normalized euclidean distance")
            norm_ax.plot(norm_hist)
            norm_ax.legend(label_names)
        else:
            eucl_fig, eucl_ax = plt.subplots()
            eucl_ax.set_title("Test histogram based on euclidean distance")
            
            norm_fig, norm_ax = plt.subplots()
            norm_ax.set_title("Test histogram based on normalized euclidean distance")
            custom_lines = [Line2D([0], [0], color="C"+str(label), lw=1) for label in range(len(label_names))]
            for recording in range(len(labels)):
                eucl_ax.plot(self.test_histograms[recording].transpose(),"C"+str(labels[recording]))
                norm_ax.plot(self.test_normalized_histograms[recording].transpose(),"C"+str(labels[recording]))
            
            eucl_ax.legend(custom_lines,label_names)
            norm_ax.legend(custom_lines,label_names)