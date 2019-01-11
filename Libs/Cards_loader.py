# -*- coding: utf-8 -*-
"""
Created on Wed Jan 9 11:40:52 2019

@author: marcorax

Function loading the Cards dataset
"""
import numpy as np
# for reading the aedat files
from Libs.readwriteatis_kaerdat import readATIS_td

def Cards_loader(data_folder, learning_set_length, testing_set_length, shuffle_seed):
    # Dataset settings
    card_sets = ["cl_","di_","he_", "sp_"]
    card_set_starting_number = [60, 17, 77, 75]
    number_of_labels = 4
    number_of_batch_per_label = 17
    dataset = []
    labels = []
    
    
    # extracting data
    for label in range(number_of_labels):
        for batch in range(number_of_batch_per_label):
             file_name = (data_folder+card_sets[label]+np.str(card_set_starting_number[label]+batch)+"_td.dat")
             data = readATIS_td(file_name, orig_at_zero = True, drop_negative_dt = True, verbose = False, events_restriction = [0, np.inf])
             # I won't use polarity information because is not informative for the given task
             dataset.append([data[0].copy(), data[1].copy(), (data[2].copy()**0)-1])
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

    return dataset_learning, labels_learning, dataset_testing, labels_testing