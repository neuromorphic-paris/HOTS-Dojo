import time
import os
import pickle
from joblib import Parallel, delayed
from Libs.Cards_loader import Cards_loader
from Libs.Sparse_HOTS.Benchmark_Libs import param_load, bench, compute_m_v

# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'

# Simultaneus threads you want to utilise on your machine 
threads = 10

# Number of runs 
runs = 10

# I won't use polarity information because is not informative for the given task
first_layer_polarities = 1

#%% Cards dataset
# The number of cards recordings is not the same for every symbol, thus in order
# to have the same probability if extracting a certain card suit I will have to 
# take max 17 recordings of each symbol as i don't have more clubs recording than that
# Thus learning_set_length+testing_set_length needs to be < number_of_batch_per_label
# i.e. 17

shuffle_seed = 0 # (the seed won't be used if 0)

learning_set_length = 12 
testing_set_length = 5
data_folder = "Datasets/Cards/usable/pips/"
parameter_folder = "Parameters/Cards/"
result_folder = "Results/Cards/"
labels = ("clubs","diamonds","heart", "spades") 

dataset = Parallel(n_jobs=threads)(delayed(Cards_loader)(data_folder, learning_set_length,
                                  testing_set_length, shuffle_seed) for run in range(runs))

#%% Load Networks parameter saved from the Playground
file_names = ["1_L_Compressive_HOTS_Parameters_2019-03-07_17:28:50.024671.pkl","1_L_My_HOTS_Parameters_2019-03-07_14:22:44.799756.pkl","1_L_Original_HOTS_Parameters_2019-03-07_14:46:49.705022.pkl"]
number_of_nets = len(file_names)
nets_parameters = Parallel(n_jobs=threads)(delayed(param_load)(parameter_folder+file_names[net]) for net in range(number_of_nets))   

#%% Execute benchmark
classification_type = False # if true enable histogram classification, if false use mlps
start_time = time.time()
bench_results = bench(dataset, nets_parameters, number_of_nets, len(labels), first_layer_polarities, classification_type, threads, runs)
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

#%% Compute mean and variance of the scores of each nework
 
mean,var = compute_m_v(bench_results)

#%% Save Results

for file_index in range(len(file_names)):
    res_file_name=file_names[file_index].replace('Parameters','Results')
    with open(result_folder+res_file_name, 'wb') as f:
        pickle.dump([mean[file_index],var[file_index],bench_results[file_index]], f)
