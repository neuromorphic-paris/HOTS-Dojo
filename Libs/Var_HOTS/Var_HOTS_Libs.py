"""
@author: marcorax

This file contains the Var_HOTS_Net class complementary functions that are used
inside the network that have no pratical use outside the context of the class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

To better understand the math behind classification an optimization please take a look
to the upcoming Var Hots paper
 
"""

from keras.layers import Lambda, Input, Dense
from keras.models import Model, Sequential
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

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
    mlp.add(Dense(output_size, activation='sigmoid'))
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
    # network parameters
    input_shape = (original_dim, )
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    L2_z = K.sum(K.square(z_mean),axis=-1)
    L2_inputs = K.sum(K.square(inputs),axis=-1)

    # VAE loss = mse_loss + kl_loss
    coding_costraint = 0.08
    reconstruction_loss = mse(inputs, outputs) + coding_costraint*K.abs(L2_z-L2_inputs)/(L2_inputs+0.01)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    #sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    vae.compile(optimizer=adam)
    
    return vae, encoder, decoder




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




