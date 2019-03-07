"""
@author: marcorax

This script can be used to test and save different parameters for
the different iterations of sparse HOTS networks
 
"""
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import os
import pickle
from Libs.Sparse_HOTS.HOTS_Sparse_Network import HOTS_Sparse_Net
from Libs.Cards_loader import Cards_loader

# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'


# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 0 # seed used for dataset shuffling if set to 0 the process will be totally random

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
       
dataset_learning, labels_learning, dataset_testing, labels_testing = Cards_loader(data_folder, learning_set_length,
                                                                                  testing_set_length, shuffle_seed)
legend = ("clubs","diamonds","heart", "spades") # Legend containing the labes used for
                                                # plots
                                                
#%% Network Creation

# Network settings
# =============================================================================
#   features_number (list of int) : it is a list containing the number of features extracted by each layer
#   surfaces_dimensions (list of list of int) : A list containing per each layer the 
#                                          dimension of time surfaces in this format:
#                                          [xdim,ydim]
#   taus (list of int) : It's a list containing the time coefficient used for the time
#                    surface creations for each layer
#   first_layer_polarities (int) : The number of polarities that the first layer will
#                              have to handle, for ATIS cameras, it is usually,
#                              2 or 1 depending wether if you consider the additional                                                
#   delay_coeff (list of int) : it's the cofficient named alfa used for generating an event in the paper
#   net_seed (int) : seed used for net parameters generation,
#                    if set to 0 the process will be totally random
#   verbose (bool) : If verbose is on, it will display graphs to assest the ongoing,
#                    learning algorithm (Per recording)
# 
# =============================================================================

features_number = [12]
surfaces_dimensions = [[11,11]]
taus = [1000]

# I won't use polarity information because is not informative for the given task
first_layer_polarities = 1

net_seed = 2

delay_coeff = 0

# Generate the network
Net = HOTS_Sparse_Net(features_number, surfaces_dimensions, taus, first_layer_polarities,
                      delay_coeff, net_seed, verbose=True)

#%% Learning-online-Exp distance and Thresh

learning_method = "learn_online"
activation_method = "Exp distance"
base_norm="Thresh"
start_time = time.time()

sparsity_coeff = [1, 0.6, 900]
learning_rate = [0.075, 0.0012, 900]
noise_ratio = [1, 0, 900]
sensitivity = [0.05, 0.12, 1200]

Net.learn_online(dataset=dataset_learning,
                  method=activation_method, base_norm=base_norm,
                  noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                  sensitivity=sensitivity,
                  learning_rate=learning_rate)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Learning-online-Dot product distance and Thresh (with full spartsity)
# This method is similiar to the one implemented in the first 2D HOTS
# Sparsity needs to be set to always 1 

learning_method = "learn_online"
activation_method = "Dot product"
base_norm="Thresh"
start_time = time.time()

sparsity_coeff = [1, 1, 10000]
learning_rate = [0.075, 0.0012, 900]
noise_ratio = [1, 0, 900]
sensitivity = [1, 1, 1] # This parameter is a name place holder

Net.learn_online(dataset=dataset_learning,
                  method=activation_method, base_norm=base_norm,
                  noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                  sensitivity=sensitivity,
                  learning_rate=learning_rate)

elapsed_time = time.time()-start_time
#print("Learning elapsed time : "+str(elapsed_time))

#%% Learning offline full batch

learning_method = "learn_offline"
activation_method = "CG"
base_norm="Thresh"
start_time = time.time()

sparsity_coeff = 0.8
learning_rate = 0.2        
max_steps = 3
base_norm_coeff = 0.0005
precision = 0.01

  
Net.learn_offline(dataset_learning, sparsity_coeff, learning_rate, max_steps, base_norm_coeff, precision)
    
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))



#%% Mlp classifier training
if learning_method=="learn_online": # The parameters are evolving
    # Taking the steady state values to perform classification
    sparsity_coeff_mlp = sparsity_coeff[1]
    noise_ratio_mlp = 0
    sensitivity_mlp = sensitivity[1]

if learning_method=="learn_offline": # The parameters are fixed
    # Taking the steady state values to perform classification
    sparsity_coeff_mlp = sparsity_coeff
    noise_ratio_mlp = 0
    sensitivity_mlp = 0
    
number_of_labels=len(legend)
mlp_learning_rate = 0.01
Net.mlp_classification_train(dataset_learning, labels_learning, number_of_labels, mlp_learning_rate, activation_method,
                                noise_ratio_mlp, sparsity_coeff_mlp, sensitivity_mlp)

#%% Mlp classifier testing

prediction_rate, predicted_labels, predicted_labels_ev = Net.mlp_classification_test(dataset_testing, labels_testing, number_of_labels, activation_method,
                                 noise_ratio_mlp, sparsity_coeff_mlp, sensitivity_mlp)
print('Prediction rate is '+str(prediction_rate*100)+'%')    

#%% Save network parameters
now=datetime.datetime.now()

if learning_method=="learn_online":
    if activation_method == "Exp distance":
        file_name = "1_L_My_HOTS_Parameters_"+str(now).replace(" ","_")+".pkl"
    if activation_method == "Dot product":
        file_name = "1_L_Original_HOTS_Parameters_"+str(now).replace(" ","_")+".pkl"
    additional_save = []    

if learning_method=="learn_offline":
    file_name = "1_L_Compressive_HOTS_Parameters_"+str(now).replace(" ","_")+".pkl"
    # Additional parameter to save
    additional_save = [max_steps,base_norm_coeff,precision]    
    sensitivity = 0
    noise_ratio = 0
    
with open(parameter_folder+file_name, 'wb') as f:
    pickle.dump([features_number, surfaces_dimensions, taus, delay_coeff, learning_method,
                 activation_method ,base_norm, sparsity_coeff,
                 learning_rate, noise_ratio, sensitivity, mlp_learning_rate, additional_save], f)
    
#%% Plot Network Evolution

layer = 0
sublayer = 0    
Net.plot_evolution(layer,sublayer)
plt.show()       

#%% Plot Basis 

layer = 0
sublayer = 0
Net.plot_basis(layer, sublayer)
plt.show()       
        
#%% Reconstruction/Generality Test _single surface_  (for Sparse Hots)
card_n = -1
surface_n = -3
layer = 0
sublayer = 0

plt.figure("Original Surface")
sns.heatmap(Net.surfaces[layer][sublayer][card_n][surface_n])           
Net.sublayer_reconstruct(layer, sublayer, Net.surfaces[layer][sublayer][card_n][surface_n],
                          "Exp distance", noise_ratio, sparsity_coeff, sensitivity)
Net.activations[layer][sublayer]
plt.show()       

#%% Complete batch reconstruction error for a single sublayer (For Sparse Hots)

layer = 0
sublayer = 0
sparsity_coeff = 0
timesurfaces = Net.surfaces[layer][sublayer]

Cards_err = Net.batch_sublayer_reconstruct_error(layer, sublayer, timesurfaces,
                                                 "Exp distance", noise_ratio,
                                                 sparsity_coeff, sensitivity)

#%% Classification train with histograms

if learning_method=="learn_online": # The parameters are evolving
    # Taking the steady state values to perform classification
    sparsity_coeff_hist = sparsity_coeff[1]
    noise_ratio_hist = 0
    sensitivity_hist = sensitivity[1]

if learning_method=="learn_offline": # The parameters are fixed
    # Taking the steady state values to perform classification
    sparsity_coeff_hist = sparsity_coeff
    noise_ratio_hist = 0
    sensitivity_hist = 0
    
number_of_labels=len(legend)
Net.histogram_classification_train(dataset_learning, labels_learning,   
                                   number_of_labels, activation_method, noise_ratio_hist,
                                   sparsity_coeff_hist, sensitivity_hist)
# Plotting results
Net.plot_histograms(legend)
plt.show()       

#%% Classification test with histograms
test_results, distances, predicted_labels = Net.histogram_classification_test(dataset_testing, labels_testing,
                                                                              number_of_labels, activation_method,
                                                                              noise_ratio_hist, sparsity_coeff_hist,
                                                                              sensitivity_hist)

# Plotting results
print("Euclidean distance recognition rate :             "+str(test_results[0]))
print("Normalsed euclidean distance recognition rate :   "+str(test_results[1]))
print("Bhattachaya distance recognition rate :           "+str(test_results[2]))

Net.plot_histograms(legend, labels=labels_testing)
plt.show()   