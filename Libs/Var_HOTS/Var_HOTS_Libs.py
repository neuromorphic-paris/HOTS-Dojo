"""
@author: marcorax

This file contains the Var_HOTS_Net class complementary functions that are used
inside the network that have no pratical use outside the context of the class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

To better understand the math behind classification an optimization please take a look
to the upcoming Var Hots paper
 
"""

from keras.layers import Lambda, Input, Dense, BatchNormalization
from keras.models import Model, Sequential
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers, regularizers
from itertools import compress

import numpy as np 
import matplotlib.pyplot as plt

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
# =============================================================================  
def sampling(args):
    """
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    Arguments :
        args (tensor): mean and log of variance of Q(z|X)
    Returns :
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# =============================================================================
def events_from_activations(activations, events):
    """
    Function used to compute events out of simple activations (raw values)
    for a single recording
    Arguments :
        activations (list of list of floats): a list containing all latent values
                                              for a time surface
    Returns :
        events (nested list) : a list containing all events generated for the current
                               recording
    """
    out_timestamps = []
    out_positions = []
    out_polarities = []
    out_rates = []
    for i in range(len(events[0])):
        t = events[0][i]
        for j,a_j in enumerate(activations[i]):
            out_timestamps.append(t)
            out_positions.append(events[1][i])
            out_polarities.append(j)
            out_rates.append(a_j)



    out_timestamps = np.array(out_timestamps)
    out_positions = np.array(out_positions)
    out_polarities = np.array(out_polarities)
    out_rates = np.array(out_rates)
    
    # Might still be needed if i add the delay, thus is only commented out for now 

#    # The output data need to be sorted like the original dataset in respect to
#    # the timestamp, because the time surface functions are expecting it to be sorted.
#    sort_ind = np.argsort(out_timestamps)
#    out_timestamps = out_timestamps[sort_ind]
#    out_positions = out_positions[sort_ind]
#    out_polarities = out_polarities[sort_ind]
#    out_rates = out_rates[sort_ind]


    return [out_timestamps, out_positions, out_polarities, out_rates]

def event_cutter(event_list, num_polarities, timecoeff, minv, xdim, ydim, tsurface_dim):
    """
    In the reconstruction problem we have to move from timesurfaces to events.
    The concept of time surface is reduntant (as the same event might appear
    in different time surfaces at different moments), thus the events generated 
    by the stacked decoder tend to increase exponentially even with few layer
    this function helps too cut away redundant information, by comparing timestamps
    and rate. It then cuts the original event_list
    Arguments :
        event_list (nested lists) : A list containing a list of all timestamps, 
                                    a list of positions, a list of polarities,
                                    and a list of rates
        num_polarities (int) : the number of expected polarities in the net
        timecoeff (float) : the coefficient used to build the timesurfaces for 
                            this layer
        minv (float) : the minimum float number at with two events of different 
                       timesurfaces can be considered the same one (as in the way
                       the surfaces are generated, equality is not expected)
       xdim (int), ydim (int) : dimensions of the entireplane for the dataset
                                generated in this layer
       tsurface_dim (list od 2 int) : dimensions of a single timesurface         

    Returns :
        event_list (nested lists) : The cut down version of the input event_list
    """
    # Let's create the lookup tables to store all events
    # one per each per type (timestamps and rate)
    # the positions will be encoded in the actual positions of values each matrix
    # only a single polarity is taken in account as i always need a full set of
    # syncronous polarities to feed the previous layer (except for the first but 
    # this function won't be called for that)
    list_dim=np.max(event_list[1])
    timestamp_table=np.nan*np.ones([list_dim+1,list_dim+1]) 
    rate_table= np.nan*np.ones([list_dim+1,list_dim+1]) 
    # in this list I will save index of the events I want to save with ones
    # while with 0 I will represent the events that are going to be ereased 
    check_list = np.ones(len(event_list[0]))
    # the for loop proceeds backwards
    for ind in range(len(event_list[0])-1,-1,-num_polarities):
        current_pos = event_list[1][ind]
        if not np.isnan(timestamp_table[current_pos[0],current_pos[1]]) or current_pos[0]<tsurface_dim[0]//2 \
            or current_pos[1]<tsurface_dim[1]//2 or current_pos[0] >= xdim-tsurface_dim[0]//2 \
            or current_pos[1]>=ydim-tsurface_dim[1]//2 :
            actual_rate = event_list[3][ind]
            predicted_rate = rate_table[current_pos[0],current_pos[1]]*np.exp((event_list[0][ind]-timestamp_table[current_pos[0],current_pos[1]])/timecoeff)
            if np.abs(actual_rate-predicted_rate)<=minv or current_pos[0]<tsurface_dim[0]//2 \
            or current_pos[1]<tsurface_dim[1]//2 or current_pos[0] >= xdim-tsurface_dim[0]//2 \
            or current_pos[1]>=ydim-tsurface_dim[1]//2 :
                check_list[ind-num_polarities+1:ind+1]=0
            else:
                timestamp_table[current_pos[0],current_pos[1]]=event_list[0][ind]
                rate_table[current_pos[0],current_pos[1]]=event_list[3][ind]
        else:
            timestamp_table[current_pos[0],current_pos[1]]=event_list[0][ind]
            rate_table[current_pos[0],current_pos[1]]=event_list[3][ind]
            
            
    event_list=[np.array(list(compress(event_list[0], check_list))),np.array(list(compress(event_list[1], check_list))),
                np.array(list(compress(event_list[2], check_list))),np.array(list(compress(event_list[3], check_list)))]
    
    # The output data need to be sorted like the original dataset in respect to
    # the timestamp, because the time surface functions are expecting it to be sorted.
    
    sort_ind = np.argsort(event_list[0])
    event_list[0] = event_list[0][sort_ind]
    event_list[1] = event_list[1][sort_ind]
    event_list[2] = event_list[2][sort_ind]
    event_list[3] = event_list[3][sort_ind]
    
    return event_list

def event_cutter_no_rate(event_list, num_polarities, timecoeff, mint, xdim, ydim, tsurface_dim, min_timestamp=0):
    """
    Same as event cutter but works expecting no rate information as in the 
    first layer
    In the reconstruction problem we have to move from timesurfaces to events.
    The concept of time surface is reduntant (as the same event might appear
    in different time surfaces at different moments), thus the events generated 
    by the stacked decoder tend to increase exponentially even with few layer
    this function helps too cut away redundant information, by comparing timestamps
    only (If i have same events (pos and polarity) at smiliar timestamps it removes
    them). 
    Arguments :
        event_list (nested lists) : A list containing a list of all timestamps, 
                                    a list of positions and a list of polarities
        num_polarities (int) : the number of expected polarities in the net
        timecoeff (float) : the coefficient used to build the timesurfaces for 
                            this layer
        mint (float) : the minimum float number at with two timestamps of different 
                       timesurfaces can be considered the same one (as in the way
                       the surfaces are generated, equality is not expected)
       xdim (int), ydim (int) : dimensions of the entireplane for the dataset
                                generated in this layer
       tsurface_dim (list od 2 int) : dimensions of a single timesurface         
       min_timestamp (int) : Due to numerical approximation it might appen that events
                             originated might have timestamps lower than 0 
                             or less than a mintimestamp. Thus this function will also
                             remove anything with a timestamps lower than min_timestamp
    Returns :
        event_list (nested lists) : The cut down version of the input event_list
    """
    # Let's create the lookup tables to store all events
    # one per each per type (timestamps)
    # the positions will be encoded in the actual positions of values each matrix
    # only a single polarity is taken in account as i always need a full set of
    # syncronous polarities to feed the previous layer (except for the first but 
    # this function won't be called for that)
    list_dim=np.max(event_list[1])
    timestamp_tables=[np.nan*np.ones([list_dim+1,list_dim+1])for pol in range(num_polarities)] 
    # in this list I will save index of the events I want to save with ones
    # while with 0 I will represent the events that are going to be ereased 
    check_list = np.ones(len(event_list[0]))
    # the for loop proceeds backwards
    for ind in range(len(event_list[0])-1,-1,-1):
        current_pos = event_list[1][ind]
        current_pol = event_list[2][ind]
        if not np.isnan(timestamp_tables[current_pol][current_pos[0],current_pos[1]]) or current_pos[0]<tsurface_dim[0]//2 \
            or current_pos[1]<tsurface_dim[1]//2 or current_pos[0] >= xdim+tsurface_dim[0]//2 \
            or current_pos[1]>=ydim+tsurface_dim[1]//2 or event_list[0][ind]<min_timestamp :
            if np.abs(event_list[0][ind]-timestamp_tables[current_pol][current_pos[0],current_pos[1]])<=mint or current_pos[0]<tsurface_dim[0]//2 \
            or current_pos[1]<tsurface_dim[1]//2 or current_pos[0] >= xdim+tsurface_dim[0]//2 \
            or current_pos[1]>=ydim+tsurface_dim[1]//2 or event_list[0][ind]<min_timestamp:
                check_list[ind]=0
            else:
                timestamp_tables[current_pol][current_pos[0],current_pos[1]]=event_list[0][ind]
        else:
            timestamp_tables[current_pol][current_pos[0],current_pos[1]]=event_list[0][ind]
    
    event_list=[np.array(list(compress(event_list[0], check_list))),np.array(list(compress(event_list[1], check_list))),
                np.array(list(compress(event_list[2], check_list)))]
    
    # The output data need to be sorted like the original dataset in respect to
    # the timestamp, because the time surface functions are expecting it to be sorted.
    
    sort_ind = np.argsort(event_list[0])
    event_list[0] = event_list[0][sort_ind]
    event_list[1] = event_list[1][sort_ind]
    event_list[2] = event_list[2][sort_ind]
    
    return event_list
# =============================================================================
def create_mlp(input_size, hidden_size, output_size, learning_rate):
    """
    Function used to create a small mlp used for classification porpuses 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optiomization alg.
    Returns :
        mlp (keras model) : the freshly baked network
    """
    mlp = Sequential()
    mlp.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
    mlp.add(Dense(output_size, activation='relu'))
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp.compile(optimizer=adam,
              loss='mean_squared_error',
              metrics=['accuracy'])
    return mlp
    
    

# Let's try with a small predefined network    
def create_vae(original_dim, latent_dim, intermediate_dim, learning_rate, coding_costraint):
    """
    Function used to create a small autoencoder used each layer of Var HOTS
    Arguments :
        original_dim (int) : size of the input layer
        latent_dim (int) : size of the output layer
        intermediate_dim (int) : size of the hidden layer
        learning_rate (int) : the learning rate for the optiomization alg.
        coding_costraint (float) : a Lagrange multiplier to constraint the autoencoders
                                   to move low active time surface rapresentation to smaller 
                                   absolute values of the latent variables (wich is fundamental 
                                   for data encoding with timesurfaces)
    Returns :
        vae (keras model) : the freshly baked network
        encoder (keras model) : the freshly baked encoder
        decoder (keras model) : the freshly baked decoder
        
    """
    
    n_samples_carlo = 50 
    
    def log_stdnormal(x):
        """log density of a normalized gaussian"""
        c = - 0.5 * np.log(2*np.pi)
        result = c - K.square(x) / 2
        return result


    def log_normal2(x, mean, log_var):
        """log density of a normalized gaussian of mean(mean) and variance exp(log_var)"""
        c = - 0.5 * np.log(2*np.pi)
        result = c - log_var/2 - K.square(x - mean) / (2 * K.exp(log_var) + 1e-8)
        return result
    
    def log_egg(x, R, var):
        """log density of an egg distribution (I need to find a better name for fuck sake)"""
        c = -K.abs(K.sum(K.square(x),axis=-1)-R**2)/(2*var)
        eplus=np.exp(+(R**2)/(2*var))
        eminus=np.exp(-(R**2)/(2*var))
        result = c  - np.log(eminus*(2*np.pi*var*(eplus-1))**(latent_dim/2)+eplus*(2*np.pi*var*eminus)**(latent_dim/2) + 1e-8)
        return result
    
    # network parameters
    input_shape = (original_dim, )
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='sigmoid')(inputs)
    x1 = Dense(intermediate_dim, activation='sigmoid')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x1)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    encoder.compile(loss='mean_squared_error', optimizer='adam')
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='sigmoid')(latent_inputs)
    x1 = Dense(intermediate_dim, activation='sigmoid')(x)
    outputs = Dense(original_dim, activation='sigmoid')(x1)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    decoder.compile(loss='mean_squared_error', optimizer='adam')

    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    L2_z = K.sum(K.square(z_mean),axis=-1)/latent_dim
    L2_inputs = K.sum(K.square(inputs),axis=-1)/original_dim
    # + coding_costraint*K.log(K.abs(L2_inputs-L2_z)+1) 
    # VAE loss = mse_loss + kl_loss
    reconstruction_loss = mse(inputs, outputs) #+ coding_costraint*K.abs(L2_inputs-L2_z)/(L2_z+0.0001) 
    R = 4
    std_egg = 2
    def monte_carlo_kl_div_EGG(args):
        mean, log_std = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        for k in range(n_samples_carlo):
            epsilon = K.random_normal(shape=(batch, dim))
            z = mean + K.exp(0.5*log_std+1e-8) * epsilon
            try:
                loss += K.sum(log_normal2(z, mean, log_std), -1) - log_egg(z, R, std_egg) 
            except NameError:
                loss = K.sum(log_normal2(z, mean, log_std), -1) - log_egg(z, R, std_egg) 
        return loss / n_samples_carlo
    
    def monte_carlo_kl_div(args):
        mean, log_std = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        for k in range(n_samples_carlo):
            epsilon = K.random_normal(shape=(batch, dim))
            z = mean + K.exp(0.5*log_std+1e-8) * epsilon
            try:
                loss += K.sum(log_normal2(z, mean, log_std) - log_stdnormal(z) , -1)
            except NameError:
                loss = K.sum(log_normal2(z, mean, log_std) - log_stdnormal(z) , -1)
        return loss / n_samples_carlo
    
    carlo = Lambda(monte_carlo_kl_div_EGG)([z_mean, z_log_var])
    
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + carlo)#+ kl_loss)#
    vae.add_loss(vae_loss)
    #sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    vae.compile(optimizer=adam)
    
    return vae, encoder, decoder


# Let's try with a small predefined network    
def create_sparse(original_dim, latent_dim, intermediate_dim, learning_rate, coding_costraint):
    """
    Function used to create a small autoencoder used each layer of Var HOTS
    Arguments :
        original_dim (int) : size of the input layer
        latent_dim (int) : size of the output layer
        intermediate_dim (int) : size of the hidden layer
        learning_rate (int) : the learning rate for the optiomization alg.
        coding_costraint (float) : a Lagrange multiplier to constraint the autoencoders
                                   to move low active time surface rapresentation to smaller 
                                   absolute values of the latent variables (wich is fundamental 
                                   for data encoding with timesurfaces)
    Returns :
        vae (keras model) : the freshly baked network
        encoder (keras model) : the freshly baked encoder
        decoder (keras model) : the freshly baked decoder
        
    """
    # network parameters
    input_shape = (original_dim, )
    
    # Define a regularizer
    def egg(latent_vars):
        return   0.0001*K.abs(8-K.sqrt(K.sum(K.square(latent_vars),axis=-1)))# + 0.0001*K.sum(K.abs(latent_vars),axis=-1)
        #return  0.0001/K.sum(K.square(latent_vars),axis=-1) 
    
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')    
#    norm = BatchNormalization()(inputs)
    x = Dense(intermediate_dim, activation='relu')(inputs)
#    x1 = Dense(intermediate_dim, activation='sigmoid')(x)
#    x2 = Dense(intermediate_dim, activation='sigmoid')(x1)
#    x3 = Dense(intermediate_dim, activation='sigmoid')(x2)
#    encoded = Dense(latent_dim, name='latent_vars', activity_regularizer=egg)(x)
    encoded = Dense(latent_dim, name='latent_vars', activity_regularizer=regularizers.l1(1e-9))(x)
    
    
    
    # instantiate encoder model
    encoder = Model(inputs, encoded, name='encoder')
    encoder.summary()
    encoder.compile(loss='mean_squared_error', optimizer='adam')
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='latent_vars')
#    norm = BatchNormalization()(latent_inputs)
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
#    x1 = Dense(intermediate_dim, activation='sigmoid')(x)
#    x2 = Dense(intermediate_dim, activation='sigmoid')(x1)
#    x3 = Dense(intermediate_dim, activation='sigmoid')(x2)
    outputs = Dense(original_dim, name='decoded')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    decoder.compile(loss='mean_squared_error', optimizer='adam')

    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs))
    sae = Model(inputs, outputs, name='vae_mlp')
    
    L2_z = K.sum(K.square(encoded),axis=-1)#/latent_dim
    L2_inputs = K.sum(K.square(inputs),axis=-1)/original_dim
    # + coding_costraint*K.log(K.abs(L2_inputs-L2_z)+1) 
    # VAE loss = mse_loss + kl_loss
    reconstruction_loss = mse(inputs, outputs) # 0.0001*K.abs(10*latent_dim-(K.sum(K.square(encoded),axis=-1)))  #   + 0.01*K.abs(4-L2_z)       #+ coding_costraint*K.abs(L2_inputs-L2_z) +1/(L2_z+0.0001) 
    sae.add_loss(reconstruction_loss)
    #sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sae.compile(optimizer=adam)
    
    return sae, encoder, decoder


def plot_reconstruct(xdim,ydim,surfaces_dimensions,input_surfaces,input_events):
    original_image = np.zeros([ydim,xdim])
    mean_norm = np.ones([ydim,xdim])
    # To better deal with sourfaces centered on the borders
    xoff = surfaces_dimensions[0][0]//2
    yoff = surfaces_dimensions[0][1]//2
    for i in range(len(input_events[0])):
        x0 = input_events[1][i,0]
        y0 = input_events[1][i,1]
        if x0+xoff>=xdim or x0-xoff<0 or y0+yoff>=ydim or y0-yoff<0 :
            continue
        original_image[(y0-yoff):(y0+yoff+1),(x0-xoff):(x0+xoff+1)] += input_surfaces[i].reshape(surfaces_dimensions[0][1],surfaces_dimensions[0][0])
        mean_norm[(y0-yoff):(y0+yoff+1),(x0-xoff):(x0+xoff+1)]  += input_surfaces[i].reshape(surfaces_dimensions[0][1],surfaces_dimensions[0][0]).astype(bool)
    plt.figure()
    plt.imshow(original_image)


             ## ELEPHANT GRAVEYARD, WHERE ALL THE UNUSED FUNCTIONS GO TO SLEEP, ##
              ##  UNTIL A LAZY DEVELOPER WILL DECIDE WHAT TO DO WITH THEM     ##
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================      
    
    
# Method for live printing the current set of basis during learning   
# =============================================================================
# surfaces : A list of surfaces to print
# figures : if the axes are empty, the function will generate a new set of them
#           along the axes.
# axes : if provided, the function will update them with the updated set of
#        images
# images : if provided, the function will update them with the new set of
#        timesurfaces
# fig_names : A list containing the names of the figures, only used if 
#             and mandatory, if the function is generating a new set of ones
#
# It returns the figure and the axes lists for updating the plots if needed
# =============================================================================             
def surface_live_plot(surfaces, figures=[], axes=[], images=[], fig_names=[]):
    if axes==[]:
        for surf in range(len(surfaces)):
            fig = plt.figure(fig_names[surf])
            ax = fig.add_subplot(111)
            image = ax.imshow(surfaces[surf])
            figures.append(fig)
            axes.append(ax)
            images.append(image)
        plt.pause(0.0001)
    else:
        for surf in range(len(surfaces)):
            images[surf].set_array(surfaces[surf])
            figures[surf].canvas.draw()
        plt.pause(0.0001)
    return figures, axes




