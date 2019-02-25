import pickle
import numpy as np
from joblib import Parallel, delayed
from Libs.Sparse_HOTS.HOTS_Sparse_Network import HOTS_Sparse_Net

def param_load(file_name):
    with open(file_name, 'rb') as f:
       [features_number, surfaces_dimensions, taus, delay_coeff, learning_method,
                 activation_method ,base_norm, sparsity_coeff,
                 learning_rate, noise_ratio, sensitivity, mlp_learning_rate, additional_save] = pickle.load(f)
    return [features_number, surfaces_dimensions, taus, delay_coeff, learning_method,
                 activation_method ,base_norm, sparsity_coeff, learning_rate,
                 noise_ratio, sensitivity, mlp_learning_rate, additional_save]

def wrapper_learn_online(HOTS_Sparse_Net, dataset, method="Exp distance", base_norm="Thresh", 
                     noise_ratio=[0,0,1], sparsity_coeff=[0,0,1], sensitivity=[0,0,1],
                     learning_rate=[0,0,1], base_norm_coeff=0):
    HOTS_Sparse_Net.learn_online(dataset, method, base_norm, noise_ratio, sparsity_coeff,
                                 sensitivity, learning_rate, base_norm_coeff)
    return HOTS_Sparse_Net
def wrapper_learn_offline(HOTS_Sparse_Net, dataset, sparsity_coeff, learning_rate,
                      max_steps, base_norm_coeff, precision):
    HOTS_Sparse_Net.learn_offline(dataset, sparsity_coeff, learning_rate,
                      max_steps, base_norm_coeff, precision)
    return HOTS_Sparse_Net

def bench(dataset,nets_parameters,number_of_nets,number_of_labels, first_layer_polarities,
          classification_type, threads, runs):
    bench_results = []
    net_seed = 0 #Network creation is complitely randomic
    for net in range(number_of_nets):
        single_net_results = []
        [features_number, surfaces_dimensions, taus, delay_coeff, learning_method,
         activation_method ,base_norm, sparsity_coeff, learning_rate,
         noise_ratio, sensitivity, mlp_learning_rate, additional_save] = nets_parameters[net]
        # Generate the network
        Nets = Parallel(n_jobs=threads)(delayed(HOTS_Sparse_Net)(features_number, surfaces_dimensions, taus, first_layer_polarities,
                      delay_coeff, net_seed) for run in range(runs))
        
        # check if the net is online or offline
        if learning_method=="learn_online":
            Nets = Parallel(n_jobs=threads)(delayed(wrapper_learn_online)(Nets[run],dataset=dataset[run][0],
                      method=activation_method, base_norm=base_norm,
                      noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                      sensitivity=sensitivity,
                      learning_rate=learning_rate)for run in range(runs))
            if classification_type is True:
                for run in range(runs):
                    Nets[run].histogram_classification_train(dataset[run][0], dataset[run][1],   
                                           number_of_labels, activation_method, 0,
                                           sparsity_coeff[1], sensitivity[1])
                    prediction_rates, distances, predicted_labels = Nets[run].histogram_classification_test(dataset[run][2], dataset[run][3],
                                                                                      number_of_labels, activation_method,0,
                                                                                      sparsity_coeff[1], sensitivity[1])
                    single_net_results.append(prediction_rates)
            else:
                for run in range(runs):
                    Nets[run].mlp_classification_train(dataset[run][0], dataset[run][1], number_of_labels, mlp_learning_rate, activation_method,
                                                 0, sparsity_coeff[1], sensitivity[1])
                    prediction_rate, predicted_labels, predicted_labels_ev = Nets[run].mlp_classification_test(dataset[run][2], dataset[run][3], number_of_labels, activation_method,
                                                                                                      0, sparsity_coeff[1], sensitivity[1])
                    single_net_results.append([prediction_rate])

        if learning_method=="learn_offline":
            Nets = Parallel(n_jobs=threads)(delayed(wrapper_learn_offline)(Nets[run], dataset[run][0], sparsity_coeff, learning_rate,
                              additional_save[0], additional_save[1], additional_save[2]) for run in range(runs))
            if classification_type is True:
                for run in range(runs):
                    Nets[run].histogram_classification_train(dataset[run][0], dataset[run][1],   
                                           number_of_labels, activation_method, 0,
                                           sparsity_coeff, 0)
                    prediction_rates, distances, predicted_labels = Nets[run].histogram_classification_test(dataset[run][2], dataset[run][3],
                                                                                      number_of_labels, activation_method,0,
                                                                                      sparsity_coeff, 0)
                    single_net_results.append(prediction_rates)
            else:
                for run in range(runs):
                    Nets[run].mlp_classification_train(dataset[run][0], dataset[run][1], number_of_labels, mlp_learning_rate, activation_method,
                                    0, sparsity_coeff, 0)
                    prediction_rate, predicted_labels, predicted_labels_ev = Nets[run].mlp_classification_test(dataset[run][2], dataset[run][3], number_of_labels, activation_method,
                                     0, sparsity_coeff, 0)
                    single_net_results.append([prediction_rate])
                
        if learning_method!="learn_online" and learning_method!="learn_offline": 
            print(learning_method+" : is not a valid learnig method")
            return
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
