import numpy as np 
from scipy import optimize 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from Libs.readwriteatis_kaerdat import readATIS_td
from Libs.Time_Surface_generators import Time_Surface_all, Time_Surface_event
from Libs.HOTS_Sparse_Network import HOTS_Sparse_Net, events_from_activations
from scipy.spatial import distance 


# Plotting settings
plt.style.use("dark_background")

# Network settings
# =============================================================================
# nbasis is a list containing the number of basis used for each layer
# ldim: is a list containing the linear dimension of every base for each layer
# taus: is a list containing the time coefficient used for the time surface creations
#       for each layer, all three lists need to share the same lenght obv.
# shuffle_seed, net_seed : seed used for dataset shuffling and net generation,
#                       if set to 0 the process will be totally random
# 
# =============================================================================

basis_number = [3]
basis_dimension = [5] 
taus = [5000]

shuffle_seed = 7
net_seed = 25

delay_coeff = 10000
       
# Preparing the card dataset 
card_sets = ["cl_","di_","he_", "sp_"]
card_set_starting_number = [60, 17, 77, 75]
number_of_labels = 4
number_of_batch_per_label = 17
dataset = []
labels = []

# The number of cards recordings is not the same for every symbol, thus in order
# to have the same probability if extracting a certain card suit I will have to 
# take max 17 recordings of each symbol as i don't have more clubs recording than that
# Thus learning_set_length+testing_set_length needs to be < number_of_batch_per_label
# i.e. 17

learning_set_length = 7 
testing_set_length = 3
pips_folder_position = "Datasets/pokerDVS/usable/pips/"
# extracting data
for label in range(number_of_labels):
    for batch in range(number_of_batch_per_label):
         file_name = (pips_folder_position+card_sets[label]+np.str(card_set_starting_number[label]+batch)+"_td.dat")
         data = readATIS_td(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = False, events_restriction = [0, np.inf])
         # I won't use polarity information because is not informative for the given task
         dataset.append([data[0].copy(), data[1].copy()])
         labels.append(label)



# learning dataset
dataset_learning = []
labels_learning = []
for label in range(number_of_labels):
    for batch in range(learning_set_length):
         dataset_learning.append(dataset[batch + label * number_of_batch_per_label])
         labels_learning.append(labels[batch + label * number_of_batch_per_label])

# testing dataset
dataset_testing = []
labels_testing = []
for label in range(number_of_labels):
    for batch in range(testing_set_length):
        testing_batch = batch + learning_set_length
        dataset_testing.append(dataset[testing_batch + label * number_of_batch_per_label])
        labels_testing.append(labels[testing_batch + label * number_of_batch_per_label])         
         


#Preparing to shuffle
rng = np.random.RandomState()
if(shuffle_seed!=0):         
    rng.seed(shuffle_seed)

#shuffle the dataset and the labels with the same order
combined_data = list(zip(dataset_learning, labels_learning))
rng.shuffle(combined_data)
dataset_learning[:], labels_learning[:] = zip(*combined_data)

#shuffle the dataset and the labels with the same order
combined_data = list(zip(dataset_testing, labels_testing))
rng.shuffle(combined_data)
dataset_testing[:], labels_testing[:] = zip(*combined_data)

# Print an element to check if it's all right
tsurface=Time_Surface_all(xdim=35, ydim=35, timestamp=0, timecoeff=taus[0], dataset=dataset_learning[2], verbose=True)

# Generate the network
Net = HOTS_Sparse_Net(basis_number, basis_dimension, taus, delay_coeff, net_seed)


#%% Learning-online

start_time = time.time()

sparsity_coeff = 5
learning_rate = 0.2
base_norm_coeff = 0.0005

Net.learn_online(sparsity_coeff, learning_rate, dataset_learning, base_norm_coeff)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Learning-offline-mean

start_time = time.time()

sparsity_coeff = 0.8
learning_rate = [5, 2.5, 1.0]
mini_batch_size = 5
phases_size = [2400, 2400, 2400]
max_gradient_steps = 10
base_norm_coeff = 0.0005
precision = 0.01

Net.learn_offline_mean(sparsity_coeff, learning_rate,
                       dataset_learning, mini_batch_size,
                       phases_size, max_gradient_steps,
                       base_norm_coeff, precision)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Learning offline full batch

#start_time = time.time()
#
#sparsity_coeff = 0.8
#learning_rate = 0.2        
#max_steps = 10
#base_norm_coeff = 0.0005
#precision = 0.01
#  
#Net.learn_offline(sparsity_coeff, learning_rate, dataset_learning, max_steps, base_norm_coeff, precision)
#    
#elapsed_time = time.time()-start_time
#print("Learning elapsed time : "+str(elapsed_time))           
    
#%% Plot Basis 

layer = 0
sublayer = 0
Net.plot_basis(layer, sublayer)
       
        
#%% Reconstruction/Generality Test _single surface_

card_n = 1
surface_n = -300
layer = 0
sublayer = 0

plt.figure()
sns.heatmap(Net.surfaces[layer][sublayer][card_n][surface_n])           
Net.sublayer_reconstruct(layer, sublayer, Net.surfaces[layer][sublayer][card_n][surface_n], sparsity_coeff)
Net.activations[layer][sublayer]
#%% Complete batch reconstruction error for a single sublayer

layer = 0
sublayer = 0
sparsity_coeff = 0
timesurfaces = Net.surfaces[layer][sublayer]

Cards_err = Net.batch_sublayer_reconstruct_error(layer, sublayer, timesurfaces, sparsity_coeff)

#%% Classification train

Net.histogram_classification_train(dataset_learning, labels_learning, number_of_labels, sparsity_coeff)



#%%

net_activity = Net.full_net_dataset_response_CG(dataset_testing, sparsity_coeff)
last_layer_activity = net_activity[-1]
#%%
histograms = []
normalized_histograms = []
n_basis = Net.basis_number[-1]
for batch in range(len(dataset_testing)):
    histograms.append(np.zeros(n_basis*(2**(Net.layers-1))))
    normalized_histograms.append(np.zeros(n_basis*(2**(Net.layers-1))))
for sublayer in range(len(last_layer_activity)):
    for batch in range(len(dataset_testing)):
        batch_histogram = sum(last_layer_activity[sublayer][batch])
        normalized_bach_histogram = batch_histogram/len(last_layer_activity[sublayer][batch])
        histograms[batch][n_basis*sublayer:n_basis*(sublayer+1)] = batch_histogram
        normalized_histograms[batch][n_basis*sublayer:n_basis*(sublayer+1)] = normalized_bach_histogram
# compute the distances per each histogram from the models
distances = []
predicted_labels = []
for batch in range(len(dataset_testing)):
    single_batch_distances = []
    for label in range(number_of_labels):
        single_label_distances = []  
        single_label_distances.append(distance.euclidean(histograms[batch],Net.histograms[label]))
        single_label_distances.append(distance.euclidean(normalized_histograms[batch],Net.normalized_histograms[label]))
        Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(normalized_histograms[batch], Net.normalized_histograms[label])]) 
        single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
        single_batch_distances.append(single_label_distances)
    single_batch_distances = np.array(single_batch_distances)
    single_batch_predicted_labels = np.argmin(single_batch_distances, 0)
    distances.append(single_batch_distances)
    predicted_labels.append(single_batch_predicted_labels)


hist = np.transpose(Net.histograms)
norm_hist = np.transpose(Net.normalized_histograms)
test_hist = np.transpose(histograms)
test_norm_hist = np.transpose(normalized_histograms)
#%% Classification test 

test_results = Net.histogram_classification_test(dataset_testing, labels_testing, number_of_labels, sparsity_coeff)


