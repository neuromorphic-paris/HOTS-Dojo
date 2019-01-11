import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import time
import datetime
import os
import pickle
from joblib import Parallel, delayed
from Libs.Cards_loader import Cards_loader
from Libs.Benchmark_Libs import param_load, bench, compute_m_v

# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'

# Simultaneus threads you want to utilise on your machine 
threads = 50

# Plotting settings
sns.set()


# Number of runs 
runs = 50

# I won't use polarity information because is not informative for the given task
first_layer_polarities = 1

#%% Cards dataset
# The number of cards recordings is not the same for every symbol, thus in order
# to have the same probability if extracting a certain card suit I will have to 
# take max 17 recordings of each symbol as i don't have more clubs recording than that
# Thus learning_set_length+testing_set_length needs to be < number_of_batch_per_label
# i.e. 17

shuffle_seed = 0 # (the seed won't be used if 0)

learning_set_length = 7 
testing_set_length = 10
data_folder = "Datasets/Cards/usable/pips/"
parameter_folder = "Parameters/Cards/"
result_folder = "Results/Cards/"
labels = ("clubs","diamonds","heart", "spades") 

dataset = Parallel(n_jobs=threads)(delayed(Cards_loader)(data_folder, learning_set_length,
                                  testing_set_length, shuffle_seed) for run in range(runs))

#%% Load Networks parameter saved from the Playground
file_names = ["My_HOTS_Dot.pkl","Original_HOTS.pkl","Compressive_HOTS.pkl"] 
number_of_nets = len(file_names)-1
nets_parameters = Parallel(n_jobs=threads)(delayed(param_load)(parameter_folder+file_names[net]) for net in range(number_of_nets))   

#%% Execute benchmark
start_time = time.time()
bench_results =  Parallel(n_jobs=threads)(delayed(bench)(dataset[run],nets_parameters,number_of_nets,len(labels),first_layer_polarities) for run in range(runs))   
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))
#%% Compute mean and variance of the scores of each nework
mean,var = compute_m_v(bench_results)
#%% Plots
x = range(3)
distances = ('Euclidean','Normalised Euclidean','Bhattacharya')            
for net in range(number_of_nets):
    fig, ax = plt.subplots()
    plt.bar(x,mean[net]*100,yerr=var[net]*100)
    plt.xticks(x,distances)
    plt.ylim(0,100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title(file_names[net][:-4])
    plt.show()
#%% Save Results
now=datetime.datetime.now()
res_file_name=str(now)+'.pkl'
with open(result_folder+res_file_name, 'wb') as f:
    pickle.dump([file_names,number_of_nets,mean,var,bench_results], f)
#%% Load Results
res_file_name='2019-01-11 13:16:09.768874.pkl'
result_folder = "Results/Cards/"
with open(result_folder+res_file_name, 'rb') as f:
       [file_names,number_of_nets,mean,var,bench_results] = pickle.load(f)