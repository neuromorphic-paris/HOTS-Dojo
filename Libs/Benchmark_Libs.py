import pickle
import numpy as np
from Libs.HOTS_Sparse_Network import HOTS_Sparse_Net

def param_load(file_name):
    with open(file_name, 'rb') as f:
       [basis_number, basis_dimension, taus, delay_coeff, learning_method,
        activation_method ,base_norm, sparsity_coeff,
        learning_rate, noise_ratio, sensitivity, additional_save] = pickle.load(f)
    return basis_number, basis_dimension, taus, delay_coeff, learning_method, activation_method ,base_norm, sparsity_coeff, learning_rate, noise_ratio, sensitivity, additional_save


def bench(dataset,nets_parameters,number_of_nets,number_of_labels, first_layer_polarities):
    single_run_results = []
    net_seed = 0 #Network creation is complitely randomic
    dataset_learning, labels_learning, dataset_testing, labels_testing = dataset
    for net in range(number_of_nets):
        [basis_number, basis_dimension, taus, delay_coeff, learning_method,
        activation_method ,base_norm, sparsity_coeff,
        learning_rate, noise_ratio, sensitivity, additional_save] = nets_parameters[net]
        # Generate the network
        Net = HOTS_Sparse_Net(basis_number, basis_dimension, taus, first_layer_polarities, delay_coeff, net_seed)
        # check if the net is online or offline
        if learning_method=="learn_online":
            Net.learn_online(dataset=dataset_learning,
                  method=activation_method, base_norm=base_norm,
                  noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                  sensitivity=sensitivity,
                  learning_rate=learning_rate)
            Net.histogram_classification_train(dataset_learning, labels_learning,   
                                   number_of_labels, activation_method, 0,
                                   sparsity_coeff[1], sensitivity[1])
            test_results, distances, predicted_labels = Net.histogram_classification_test(dataset_testing, labels_testing,
                                                                              number_of_labels, activation_method,0,
                                                                              sparsity_coeff[1], sensitivity[1])
        if learning_method=="learn_offline":
            Net.learn_offline(dataset_learning, sparsity_coeff, learning_rate,
                              additional_save[0], additional_save[1], additional_save[2])
            Net.histogram_classification_train(dataset_learning, labels_learning,   
                                   number_of_labels, activation_method, 0,
                                   sparsity_coeff, 0)
            test_results, distances, predicted_labels = Net.histogram_classification_test(dataset_testing, labels_testing,
                                                                              number_of_labels, activation_method,0,
                                                                              sparsity_coeff, 0)
        if learning_method!="learn_online" and learning_method!="learn_offline": 
            print(learning_method+" : is not a valid learnig method")
            return
        single_run_results.append(test_results)
        
    return single_run_results

def compute_m_v(bench_results):
    runs = len(bench_results)
    mean=np.zeros([len(bench_results[0]),3]) 
    var=np.zeros([len(bench_results[0]),3])     
    for run in range(runs):
        mean+=bench_results[run]
    mean = mean/runs
    for run in range(runs):
        var += (bench_results[run]-mean)**2
    return mean,var