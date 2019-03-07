import pickle
import numpy as np
from Libs.Var_HOTS.Var_HOTS_Network import Var_HOTS_Net

def param_load(file_name):
    with open(file_name, 'rb') as f:
       [latent_variables, surfaces_dimensions, taus, learning_rate, first_layer_polarities,
                 coding_costraint, mlp_learning_rate] = pickle.load(f)
    return [latent_variables, surfaces_dimensions, taus, learning_rate, first_layer_polarities,
                 coding_costraint, mlp_learning_rate]


def bench(dataset,nets_parameters,number_of_nets,number_of_labels, first_layer_polarities,
          classification_type, threads, runs):
    bench_results = []
    for net in range(number_of_nets):
        single_net_results = []
        [latent_variables, surfaces_dimensions, taus, learning_rate, first_layer_polarities,
         coding_costraint, mlp_learning_rate] = nets_parameters[net]
        for run in range(runs) :             
            dataset_learning, labels_learning, dataset_testing, labels_testing = dataset[run]
            
            # Generate the network
            Net = Var_HOTS_Net(latent_variables, surfaces_dimensions, taus, first_layer_polarities,
                   threads=threads)
        
            Net.learn(dataset=dataset_learning, learning_rate=learning_rate, coding_costraint=coding_costraint)
            
            Net.mlp_classification_train(labels_learning,   
                                   number_of_labels, mlp_learning_rate)
            
            prediction_rate, predicted_labels, predicted_labels_ev = Net.mlp_classification_test(labels_testing, number_of_labels, dataset_testing)
            
            single_net_results.append([prediction_rate])
        bench_results.append(single_net_results)
        
    return bench_results

def compute_m_v(bench_results):
    runs = len(bench_results[0])
    nets = len(bench_results)
    metrics = len(bench_results[0][0])
    mean=np.zeros([nets, metrics]) 
    var=np.zeros([nets, metrics])       
    for run in range(runs):
        mean+=[bench_results[net][run] for net in range(nets)]
    mean = mean/runs
    for run in range(runs):
        var += ([bench_results[net][run] for net in range(nets)]-mean)**2
    return mean,var
