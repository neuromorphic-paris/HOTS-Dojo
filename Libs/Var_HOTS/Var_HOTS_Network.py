#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax

This file contains the Var_HOTS_Net class 
 
"""
# General purpouse libraries
import numpy as np 
import keras
from scipy.spatial import distance
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Homemade Fresh Libraries like Gandma taught
from Libs.Var_HOTS.Var_HOTS_Libs import create_vae, create_mlp, events_from_activations
from Libs.Var_HOTS.Time_Surface_generators import Time_Surface_event


# Class for Var_HOTS_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, reconstruction and classification of the input data
# =============================================================================
class Var_HOTS_Net:
    """
    Variational Autoencoding HOTS, a version of the network aimed at dense feature 
    encoding and image reconstruction
    
    Variational HOTS constructor Arguments:
    
        latent_variables (list of int) : A list containing the latent variables per each 
                                        layer 
        surfaces_dimensions (list of list of int) : A list containing per each layer the 
                                               dimension of time surfaces in this format:
                                               [xdim,ydim]
        taus (list of int) : It's a list containing the time coefficient used for the time
                             surface creations for each layer       
        learning_rate (list of float) : List containing the gradient descent for Adam 
                                        iterative optimizer
        first_layer_polarities (int) : How many event polarities the first layer will 
                                       have to manage 
        threads (int) : The network can compute timesurfaces in a parallel way,
                        this parameter set the number of multiple threads allowed to run
        verbose (bool) : If verbose is on, it will display graphs to assest the ongoing,
                         learning algorithm (Per batch)
                                       
    """
    # Network constructor
    # =============================================================================
    def __init__(self, latent_variables, surfaces_dimensions, taus, first_layer_polarities, threads=1, verbose=False):

        self.last_layer_activations = []
        self.taus = taus
        self.layers = len(latent_variables)
        self.latent_variables = latent_variables
        self.surfaces_dimensions = surfaces_dimensions
        self.polarities = []
        self.polarities.append(first_layer_polarities)
        self.vaes = []
        self.threads = threads
        for layer in range(self.layers-1):
            self.polarities.append(self.latent_variables[layer])
        self.verbose=verbose
        # attribute containing all surfaces computed in each layer and sublayer
        self.surfaces = []
        # attribute containing all optimization errors computed in each layer 
        # and sublayer
        self.errors = []
  
    

    # Method for learning
    # =============================================================================        
    def learn(self, dataset, learning_rate, coding_costraint = 0.08): 
        """
        The method is full online but it espects to work on an entire dataset
        at once to be more similiar in structure to its offline counterpart.
        This library has the porpouse to test various sparse HOTS implementations
        and to compare them, not to work in real time! (If it's worthy a real time
                                                        will hit the shelfs soon)
        Arguments:
            dataset (nested lists) : the initial dataset used for learning
            learning_rate (float) : It's the learning rate of ADAM online optimization                 
            coding_costraint (float) : a Lagrange multiplier to constraint the autoencoders
                                       to move low active time surface rapresentation to smaller 
                                       absolute values of the latent variables (wich is fundamental 
                                       for data encoding with timesurfaces)
        
        """
        # The list of all data batches to currently process
        # input_data[recording][events, timestamp if 0 or xy coordinates if 1]
        # In the first layer the input data is directly dataset given by the user
        input_data=dataset
    
        for layer in range(self.layers):
            layer_activations = []
            new_data = []
            all_surfaces = []
            # Create the varational autoencoder for this layer
            #intermediate_dim = self.surfaces_dimensions[layer][0]*self.surfaces_dimensions[layer][0]
            intermediate_dim = 40
            self.vaes.append(create_vae(self.surfaces_dimensions[layer][0]*self.surfaces_dimensions[layer][1]*self.polarities[layer],
                                        self.latent_variables[layer], intermediate_dim, learning_rate[layer], coding_costraint))
            # The code is going to run on gpus, to improve performances rather than 
            # a pure online algorithm I am going to minibatch 
            batch_size = 500
            for recording in range(len(input_data)):
                n_batch = len(input_data[recording][0]) // batch_size
                
                
                # Cut the excess data in the first layer : 
                if layer == 0 :
                    input_data[recording][0]=input_data[recording][0][:n_batch*batch_size]
                    input_data[recording][1]=input_data[recording][1][:n_batch*batch_size]
                    input_data[recording][2]=input_data[recording][2][:n_batch*batch_size]
                    event = [[input_data[recording][0][event_ind],
                              input_data[recording][1][event_ind],
                              input_data[recording][2][event_ind]]for event_ind in range(n_batch*batch_size)] 
                else :
                    event = [[input_data[recording][0][event_ind],
                              input_data[recording][1][event_ind],
                              input_data[recording][2][event_ind],
                              input_data[recording][3][event_ind]]for event_ind in range(n_batch*batch_size)] 
                # The multiple event polarities are all synchonized in the layers after the first.
                # As a single time surface is build on all polarities, there is no need to build a time 
                # surface per each event with a different polarity and equal time stamp, thus only 
                # a fraction of the events are extracted here
                if layer != 0 :
                    recording_surfaces = Parallel(n_jobs=self.threads)(delayed(Time_Surface_event)(self.surfaces_dimensions[layer][0],
                                        self.surfaces_dimensions[layer][1], event[event_ind].copy(),
                                        self.taus[layer], input_data[recording].copy(), self.polarities[layer], minv=0.1) for event_ind in range(0,n_batch*batch_size,self.polarities[layer]))
                else:
                    recording_surfaces = Parallel(n_jobs=self.threads)(delayed(Time_Surface_event)(self.surfaces_dimensions[layer][0],
                                        self.surfaces_dimensions[layer][1], event[event_ind].copy(),
                                        self.taus[layer], input_data[recording].copy(), self.polarities[layer], minv=0.1) for event_ind in range(n_batch*batch_size))
                all_surfaces = all_surfaces + recording_surfaces
            all_surfaces=np.array(all_surfaces)
            self.vaes[layer][0].fit(all_surfaces, shuffle=False,
                     epochs=20, batch_size=batch_size,
                     validation_data=(all_surfaces, None))
            current_pos = 0
            for recording in range(len(input_data)):                
                # Get network activations at steady state (after learning)
                if layer != 0 :
                    recording_results, _, _ = self.vaes[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(input_data[recording][0])//self.polarities[layer]]), batch_size=batch_size)
                    current_pos += len(input_data[recording][0])//self.polarities[layer]
                else:
                    recording_results, _, _ = self.vaes[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(input_data[recording][0])]), batch_size=batch_size)
                    current_pos += len(input_data[recording][0])
                layer_activations.append(recording_results)
                
                # Generate new events only if I am not at the last layer
                if layer != (self.layers-1):
                    if layer != 0:
                        new_data.append(events_from_activations(recording_results, [input_data[recording][0][range(0,len(input_data[recording][0]),self.polarities[layer])],
                                                                                    input_data[recording][1][range(0,len(input_data[recording][0]),self.polarities[layer])]]))
                    else:
                        new_data.append(events_from_activations(recording_results, input_data[recording]))
                        
            input_data=new_data
        self.last_layer_activations = layer_activations
    
    # Method to compute a full network renspose, similiar to learn in the code 
    # structure
    # =============================================================================  
    def full_net_dataset_response(self, dataset, layer_stop = -1, save=False):
        """
        This method compute a full network response, but it saves only the activations
        of the last layer to save memory, if you are interested in a layer different
        from the last one, you can select it with layer_stop
        
        Arguments:
            dataset (nested lists) : the initial dataset that the network will have to respond to 
            layer_stop (int) : It's the learning rate, what do you expect me to say ?                  
            verbose (bool) : If verbose is on, it will display graphs to assest the ongoing,
                                                           learning algorithm (Per batch)
        
        """
        
        # The list of all data batches to currently process
        # input_data[recording][events, timestamp if 0 or xy coordinates if 1]
        # In the first layer the input data is directly dataset given by the user
        input_data=dataset
        
        # If required by the user, the computation will stop ad a predefined layer
        if layer_stop==-1:
            layer_stop=self.layers-1
        
        for layer in range(layer_stop+1):
            layer_activations = []
            new_data = []
            # The code is going to run on gpus, to improve performances rather than 
            # a pure online algorithm I am going to minibatch 
            batch_size = 125
            for recording in range(len(input_data)):
                recording_surfaces = []
                n_batch = len(input_data[recording][0]) // batch_size
                # Cut the excess data in the first layer : 
                if layer == 0 :
                    input_data[recording][0]=input_data[recording][0][:n_batch*batch_size]
                    input_data[recording][1]=input_data[recording][1][:n_batch*batch_size]
                    input_data[recording][2]=input_data[recording][2][:n_batch*batch_size]
                    event = [[input_data[recording][0][event_ind],
                              input_data[recording][1][event_ind],
                              input_data[recording][2][event_ind]]for event_ind in range(n_batch*batch_size)] 
                else :
                    event = [[input_data[recording][0][event_ind],
                              input_data[recording][1][event_ind],
                              input_data[recording][2][event_ind],
                              input_data[recording][3][event_ind]]for event_ind in range(n_batch*batch_size)] 

                # The multiple event polarities are all synchonized in the layers after the first.
                # As a single time surface is build on all polarities, there is no need to build a time 
                # surface per each event with a different polarity and equal time stamp, thus only 
                # a fraction of the events are extracted here
                if layer != 0 :
                    recording_surfaces = Parallel(n_jobs=self.threads)(delayed(Time_Surface_event)(self.surfaces_dimensions[layer][0],
                                        self.surfaces_dimensions[layer][1], event[event_ind].copy(),
                                        self.taus[layer], input_data[recording].copy(), self.polarities[layer], minv=0.1) for event_ind in range(0,n_batch*batch_size,self.polarities[layer]))
                else:
                    recording_surfaces = Parallel(n_jobs=self.threads)(delayed(Time_Surface_event)(self.surfaces_dimensions[layer][0],
                                        self.surfaces_dimensions[layer][1], event[event_ind].copy(),
                                        self.taus[layer], input_data[recording].copy(), self.polarities[layer], minv=0.1) for event_ind in range(n_batch*batch_size))
                recording_results, _, _ = self.vaes[layer][1].predict(np.array(recording_surfaces), batch_size=batch_size)
                layer_activations.append(recording_results)
                # Generate new events only if I am not at the last layer
                if layer != (self.layers-1):
                    if layer != 0:
                        new_data.append(events_from_activations(recording_results, [input_data[recording][0][range(0,len(input_data[recording][0]),self.polarities[layer])],
                                                                                    input_data[recording][1][range(0,len(input_data[recording][0]),self.polarities[layer])]]))
                    else:
                        new_data.append(events_from_activations(recording_results, input_data[recording]))
            input_data=new_data
        
        if save is True:
            self.last_layer_activations = layer_activations
        
        return layer_activations
        
        

    
    # Method for training a mlp classification model 
    # =============================================================================      
    def mlp_classification_train(self, labels, number_of_labels, learning_rate, dataset=[]):
        """
        Method to train a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            labels (list of int) : List of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
            learning_rate (float) : The method is Adam 
            dataset (nested lists) : the dataset used for learn classification, if not declared the method
                         will use the last response of the network. To avoid surprises 
                         check that the labels inserted here and the dataset used for 
                         either .learn .full_net_dataset_response a previous .mlp_classification_train
                         .mlp_classification_test is matching (Which is extremely faster
                         than computing the dataset again, but be aware of that)

        """
        
        if dataset:
            self.full_net_dataset_response(dataset=dataset, save=True)            
        last_layer_activity = self.last_layer_activations      
        processed_labels = np.concatenate([labels[recording]*np.ones(len(last_layer_activity[recording])) for recording in range(len(labels))])
        processed_labels = keras.utils.to_categorical(processed_labels, num_classes = number_of_labels)
        last_layer_activity = np.concatenate(last_layer_activity)
        n_latent_var = self.latent_variables[-1]
        self.mlp = create_mlp(input_size=n_latent_var,hidden_size=50, output_size=number_of_labels, 
                              learning_rate=learning_rate)
        self.mlp.summary()
        self.mlp.fit(np.array(last_layer_activity), np.array(processed_labels),
          epochs=50,
          batch_size=125)
            
        if self.verbose is True:
            print("Training ended, you can now access the trained network with the method .mlp")

    # Method for testing the mlp classification model
    # =============================================================================
    def mlp_classification_test(self, labels, number_of_labels, dataset=[]):
        """
        Method to test a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            labels (list of int) : List of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
            dataset (nested lists) : the dataset used for testing classification, if not declared the method
                                     will use the last response of the network. To avoid surprises 
                                     check that the labels inserted here and the dataset used for 
                                     either .learn .full_net_dataset_response a previous .mlp_classification_train
                                     .mlp_classification_test is matching (Which is extremely faster
                                     than computing the dataset again, but be aware of that)

        """
        if dataset:
            self.full_net_dataset_response(dataset=dataset, save=True)     
        last_layer_activity = self.last_layer_activations    
        last_layer_activity = np.concatenate(last_layer_activity)
        predicted_labels_ev=self.mlp.predict(np.array(last_layer_activity),batch_size=125)
        counter=0
        predicted_labels=[]
        for recording in range(len(self.last_layer_activations)):
            activity_sum = sum(predicted_labels_ev[counter:counter+len(self.last_layer_activations[recording])])
            predicted_labels.append(np.argmax(activity_sum))
            counter += len(self.last_layer_activations[recording])
        prediction_rate=0
        for i,true_label in enumerate(labels):
            prediction_rate += (predicted_labels[i] == true_label)/len(labels)
        return prediction_rate, predicted_labels, predicted_labels_ev


    # =============================================================================      
    def plot_vae_decode_2D(self, layer, variables_ind, variable_fix=0):
        """
        Plots reconstructed timesurfaces as function of 2-dim latent vector
        of a selected layer of the network
        Arguments:
            layer (int) : layer used to locate the latent variable that will be 
                         used to decode the timesurfaces
            variables_ind (list of 2 int) : as this method plot a 2D rapresentation
                                            it expects two latent variables to work
                                            with, thus here you can select the index
                                            of the latent variables that will 
                                            be displayed
            variable_fix (float) : the value at which all other latent variables
                                   will be fixed, 0 on default
        """
        
        num_polarities = self.polarities[layer]
        size_x = self.surfaces_dimensions[layer][0]*num_polarities
        size_y = self.surfaces_dimensions[layer][1]
        decoder = self.vaes[layer][2]
        
        # display a 30x30 2D manifold of timesurfaces
        n = 30
        figure = np.zeros((size_y * n, size_x*n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
    
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.ones(self.latent_variables[layer])*variable_fix
                z_sample[variables_ind[0]]=xi
                z_sample[variables_ind[1]]=yi
                x_decoded = decoder.predict(np.array([z_sample]))
                tsurface = x_decoded[0].reshape(size_y, size_x)
                figure[i * size_y: (i + 1) * size_y,
                       j * size_x: (j + 1) * size_x] = tsurface
    
        plt.figure(figsize=(10, 10))
        plt.title("2D Latent decoding grid")
        start_range_x = size_x // 2
        end_range_x = n * size_x + start_range_x + 1
        pixel_range_x = np.arange(start_range_x, end_range_x, size_x)
        start_range_y = size_y // 2
        end_range_y = n * size_y + start_range_y + 1
        pixel_range_y = np.arange(start_range_y, end_range_y, size_y)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range_x, sample_range_x)
        plt.yticks(pixel_range_y, sample_range_y)
        plt.xlabel("z["+str(variables_ind[0])+"]")
        plt.ylabel("z["+str(variables_ind[1])+"]")
        plt.imshow(figure)
        plt.show()
    
    # =============================================================================          
    def plot_vae_space_2D(self, layer, variables_ind, label_names, labels, dataset=[]):
        """
        Plots latent rapresentation of a dataset for a given layer, or the latest
        network activations if no dataset is given,
        the data is also colored given the labels 
        Arguments:
            layer (int) : layer used to locate the latent variable that will be 
                           plotted, ignored if no dataset is given 
            variables_ind (list of 2 int) : as this method plot a 2D rapresentation
                                            it expects two latent variables to work
                                            with, thus here you can select the index
                                            of the latent variables that will 
                                            be displayed
            labels_names (list of strings) : The name of each label, used to plot 
                                             the legend
            labels (list of int) : List of integers (labels) of the dataset
            dataset (nested lists) : the dataset that will generate the responses
                                     through the .full_net_dataset_response method
                                     
                                     To avoid surprises 
                                     check that the labels inserted here and the dataset used for 
                                     either .learn .full_net_dataset_response a previous .mlp_classification_train
                                     .mlp_classification_test is matching (Which is extremely faster
                                     than computing the dataset again, but be aware of that)
        """
        if dataset:
            selected_layer_activity = self.full_net_dataset_response(dataset=dataset,layer_stop=layer)     
        else:
            selected_layer_activity = self.last_layer_activations   
        
        # Plot variable Space
        # display a 2D plot of the time surfaces in the latent space
        
        # Create a label array to specify the labes for each timesurface
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title("2D Latent dataset rapresentation")
        custom_lines = [Line2D([0], [0], color="C"+str(label), lw=1) for label in range(len(label_names))]
        for recording in range(len(labels)):
            ax.scatter(selected_layer_activity[recording][:, variables_ind[0]], selected_layer_activity[recording][:, variables_ind[1]], c="C"+str(labels[recording]))
        ax.legend(custom_lines,label_names)
        plt.xlabel("z["+str(variables_ind[0])+"]")
        plt.ylabel("z["+str(variables_ind[1])+"]")
        plt.show()
    
    # UNDER WORK #
    # =============================================================================      
    def reconstruct(self, dataset, recording, beg_ind, end_ind, xdim, ydim):
        data = [dataset[recording][0][beg_ind:end_ind],dataset[recording][1][beg_ind:end_ind],dataset[recording][2][beg_ind:end_ind]]      
        input_surfaces=Parallel(n_jobs=self.threads)(delayed(Time_Surface_event)(self.surfaces_dimensions[0][0],
                     self.surfaces_dimensions[0][1], [data[0][event_ind],data[1][event_ind],data[2][event_ind]],
                     self.taus[0], data, self.polarities[0], minv=0.1) for event_ind in range(end_ind-beg_ind))   
        original_image = np.zeros([ydim+self.surfaces_dimensions[0][1],xdim+self.surfaces_dimensions[0][0]])
        mean_norm = np.zeros([ydim+self.surfaces_dimensions[0][1],xdim+self.surfaces_dimensions[0][0]])
        xoff = self.surfaces_dimensions[0][0]//2
        yoff = self.surfaces_dimensions[0][1]//2  
        for i in range(len(data[0])):
            x0 = data[1][i,0]
            y0 = data[1][i,1]
            original_image[(y0):(y0+2*yoff+1),(x0):(x0+2*xoff+1)] += input_surfaces[i].reshape(self.surfaces_dimensions[0][1],self.surfaces_dimensions[0][0])
            mean_norm[(y0):(y0+2*yoff+1),(x0):(x0+2*xoff+1)]  += input_surfaces[i].reshape(self.surfaces_dimensions[0][1],self.surfaces_dimensions[0][0]).astype(bool)
        plt.imshow(original_image)
        
        
             ## ELEPHANT GRAVEYARD, WHERE ALL THE UNUSED METHODS GO TO SLEEP, ##
              ##  UNTIL A LAZY DEVELOPER WILL DECIDE WHAT TO DO WITH THEM    ##
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================  
    
    
    
    # Method for training an histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # =============================================================================      
    def histogram_classification_train(self, labels, number_of_labels, dataset=[]):
        if dataset:
            self.full_net_dataset_response(dataset=dataset, save=True)            
        last_layer_activity = self.last_layer_activations      
        histograms = []
        normalized_histograms = []
        n_latent_var = self.latent_variables[-1]
        # Normalization factor for building histograms
        recording_per_label  = np.zeros(number_of_labels)
        # this array it's used to count the number of recordings with a certain label,
        # the values will be then used to compute mean histograms
        for label in range(number_of_labels):
            histograms.append(np.zeros(n_latent_var))
            normalized_histograms.append(np.zeros(n_latent_var))
        for recording in range(len(labels)):
            current_label = labels[recording]
            recording_per_label[current_label] += 1            
            recording_histogram = sum(last_layer_activity[recording])
            histograms[current_label] += recording_histogram               
        for label in range(number_of_labels):
            normalized_histograms[label] = histograms[label]/sum(abs(histograms[label]))
            histograms[label] = histograms[label]/recording_per_label[label]
        self.histograms = histograms
        self.normalized_histograms = normalized_histograms
        if self.verbose is True:
            print("Training ended, you can now look at the histograms with in the "+
                  "attribute .histograms and .normalized_histograms, or using "+
                  "the .plot_histograms method")

    # Method for testing the histogram classification model as proposed in 
    # HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476
    # =============================================================================
    # =============================================================================      
    def histogram_classification_test(self, labels, number_of_labels, dataset=[]):
        if dataset:
            self.full_net_dataset_response(dataset=dataset, save=True)     
        last_layer_activity = self.last_layer_activations    
        histograms = []
        normalized_histograms = []
        n_latent_var = self.latent_variables[-1]
        for recording in range(len(labels)):
            histograms.append(np.zeros(n_latent_var))
            normalized_histograms.append(np.zeros(n_latent_var))
            batch_histogram = sum(last_layer_activity[recording])
            histograms[recording] += batch_histogram               
            normalized_histograms[recording] = histograms[recording]/sum(abs(histograms[recording]))
        # compute the distances per each histogram from the models
        distances = []
        predicted_labels = []
        for recording in range(len(labels)):
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
        for i,true_label in enumerate(labels):
            eucl += (predicted_labels[i][0] == true_label)/len(labels)
            norm_eucl += (predicted_labels[i][1] == true_label)/len(labels)
            bhatta += (predicted_labels[i][2] == true_label)/len(labels)
        prediction_rates = [eucl, norm_eucl, bhatta]
        if self.verbose is True:
            print("Testing ended, you can also look at the test histograms with in"+
                  " the attribute .test_histograms and .test_normalized_histograms, "+
                  "or using the .plot_histograms method")
        return prediction_rates, distances, predicted_labels
        
    # Method for plotting the histograms of the network, either result of train 
    # or testing
    # =============================================================================
    # label_names : tuple containing the names of each label that will displayed
    #               in the legend
    # labels : list containing the labels of the test dataset used to generate
    #          the histograms, if empty, the function will plot the class histograms
    #          computed using .histogram_classification_train
    # =============================================================================
    def plot_histograms(self, label_names, labels=[]):
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