"""
@author: marcorax

This script can be used to test and save different parameters for
Variational HOTS networks
 
"""

# General porpouse libraries
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle
import datetime


# Data loading variables
from Libs.Cards_loader import Cards_loader

# Libraries for Variational HOTS 
from Libs.Var_HOTS.Var_HOTS_Network import Var_HOTS_Net

# To avoid MKL inefficient multithreading
os.environ['MKL_NUM_THREADS'] = '1'

# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 12 # seed used for dataset shuffling, if set to 0 the process will be totally random

#%% Cards dataset
# The number of cards recordings is not the same for every symbol, thus in order
# to have the same probability if extracting a certain card suit I will have to 
# take max 17 recordings of each symbol as i don't have more clubs recording than that
# Thus learning_set_length+testing_set_length needs to be < number_of_batch_per_label
# i.e. 17

learning_set_length = 12 
testing_set_length = 5
data_folder = "Datasets/Cards/usable/pips/"
parameter_folder = "Parameters/Cards/"
# I am merging polarity information because is not informative for the pips dataset
dataset_polarities = 1 

dataset_learning, labels_learning, dataset_testing, labels_testing = Cards_loader(data_folder, learning_set_length,
                                                                                  testing_set_length, shuffle_seed)
legend = ("clubs","diamonds","heart", "spades") # Legend containing the labes used for
                                                # plots
                                                
#%% Network Creation

# Network settings
# =============================================================================
# latent_variables (list of int) : A list containing the latent variables per each 
#                                  layer 
# surface_dimensions (list of list of int) : A list containing per each layer the 
#                                            dimension of time surfaces in this format:
#                                            [xdim,ydim]
# taus (list of int) : It's a list containing the time coefficient used for the time
#                      surface creations for each layer       
# learning_rate (list of float) : List containing the gradient descent for Adam 
#                                 iterative optimizer
# first_layer_polarities (int) : How many event polarities the first layer will 
#                                have to manage 
# threads (int) : the max number of parallel threads used to timesurface creation
# =============================================================================

latent_variables = [6,8]
surfaces_dimensions = [[11,11],[15,15]]
taus = [1000,5000]
learning_rate = [0.001,0.001]
coding_costraint = 1

first_layer_polarities = dataset_polarities


# Generate the network
Net = Var_HOTS_Net(latent_variables, surfaces_dimensions, taus, first_layer_polarities,
                   threads=8, verbose=True)

#%% Learning phase

delay = [700,3000]

start_time = time.time()

Net.learn(dataset=dataset_learning, dataset_test=dataset_testing, learning_rate=learning_rate, event_delay=delay, coding_costraint=coding_costraint)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Print features

# 2D plots that can be used to analyze the network after learning
# =============================================================================
# layer (int) : Layer selected for printing features and latent space
# variables_ind (list if 2 ind) : A list contining the couple of latent variables
#                                 that will be used to generate the plots
# variable_fix (int) : Fixed value for all the non selected state variables used
#                      used for decoding
# =============================================================================

layer = 1
variables_ind = [0,1] 
variable_fix = 0

Net.plot_vae_decode_2D(0, variables_ind, variable_fix)   
Net.plot_vae_decode_2D(layer, variables_ind, variable_fix)
Net.plot_vae_space_2D(0, variables_ind, legend, labels_learning, dataset_learning)
Net.plot_vae_space_2D(layer, variables_ind, legend, labels_learning, dataset_learning)

plt.pause(0.1)

#%% Mlp classifier training

#number_of_labels=len(legend)
#mlp_learningx_rate = 0.001
#Net.mlp_classification_train(labels_learning,   
#                                   number_of_labels, mlp_learning_rate)

#%% Mlp classifier testing

#prediction_rate, predicted_labels, predicted_labels_ev = Net.mlp_classification_test(labels_testing, number_of_labels, dataset_testing)
#print('Prediction rate is '+str(prediction_rate*100)+'%')

#%% Save network parameters
#now=datetime.datetime.now()
#file_name = "2_L_Variational_HOTS_Parameters_"+str(now).replace(" ","_")+".pkl"
#    
#with open(parameter_folder+file_name, 'wb') as f:
#    pickle.dump([latent_variables, surfaces_dimensions, taus, learning_rate, first_layer_polarities,
#                 coding_costraint, mlp_learning_rate], f) 

#%% Prediction
#from Libs.Var_HOTS.Time_Surface_generators import Time_Surface_all
[predicted_surfaces, predicted_data, real_surfaces, real_data, 
 new_data, wewewewe, we]=Net.reconstruct(dataset_learning, 0, 2000, 14000, 35, 35)
import numpy as np
test=np.abs(np.sqrt(np.sum(we**2,1)))
#Time_Surface_all(35, 35, 1000, 1000, dataset_learning[6], 1, minv=0.1, verbose=True)

#%% Histogram classifier training, not used anymore

#number_of_labels=len(legend)
#Net.histogram_classification_train(labels_learning,   
#                                   number_of_labels)
## Plotting results
#Net.plot_histograms(legend)
#plt.show()       
#plt.pause(0.01)

#%% Histogram classifier testing, not used anymore

#test_results, distances, predicted_labels = Net.histogram_classification_test(labels_testing,
#                                                                              number_of_labels,
#                                                                              dataset_testing)
#
## Plotting results
#print("Euclidean distance recognition rate :             "+str(test_results[0]))
#print("Normalsed euclidean distance recognition rate :   "+str(test_results[1]))
#print("Bhattachaya distance recognition rate :           "+str(test_results[2]))
#
#Net.plot_histograms(legend, labels=labels_testing)
#plt.show()
#plt.pause(0.01)  

