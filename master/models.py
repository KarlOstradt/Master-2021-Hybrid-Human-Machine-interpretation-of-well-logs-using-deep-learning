import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def vae_encoder_model(input_shape, intermediate_dim, latent_dim):
    """ Create an encoder model.
    
    Args:
        input_shape (tuple): Tuple with original dimension. Ex: '(org_dim,)'.
        intermediate_dim (list): List of intermediate dimension sizes.
        laten_dim (int): Dimensional size of latent space. 
        
    Returns:
        tensorflow.keras.model: Encoder model
    """
    
    inputs = keras.Input(shape=input_shape, name='input')
    x = layers.Dense(intermediate_dim[0], activation='relu', name='intermediate')(inputs)
    
    for i in range(1, len(intermediate_dim)):
        x = layers.Dense(intermediate_dim[i], activation='relu', name='intermediate'+str(i+1))(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_sigma = layers.Dense(latent_dim, name='z_sd')(x)
    z = layers.Lambda(vae_sampling, output_shape=(latent_dim,), name='latent_space')([z_mean, z_log_sigma])
    
    return keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
  
    
def vae_sampling(args):
    z_mean, z_log_sigma = args
    epsilon = keras.backend.random_normal(shape=(z_mean.shape[1],), seed=1)
    
    return z_mean + keras.backend.exp(z_log_sigma) * epsilon

def vae_decoder_model(latent_dim, intermediate_dim, original_dim):
    """ Create a decoder model.
    
    Args:
        laten_dim (int): Dimensional size of latent space.
        intermediate_dim (list): List of intermediate dimension sizes. 
        original_dim (int): Dimensional size of original data.  
        
    Returns:
        tensorflow.keras.model: Decoder model
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim[-1], activation='relu', name='intermediate')(latent_inputs)
    
    intermediate_dim = intermediate_dim[::-1]
    
    for i in range(1, len(intermediate_dim)):
        x = layers.Dense(intermediate_dim[i], activation='relu', name='intermediate'+str(i+1))(x)
        
    outputs = layers.Dense(original_dim, activation='sigmoid', name='output')(x)
    
    return keras.Model(latent_inputs, outputs, name='decoder')


def vae_model(original_dim, intermediate_dim, latent_dim, loss):
    """ Create a variational autoencoder model.
    
    Args:
        original_dim (int): Dimensional size of original data.
        intermediate_dim (list): List of intermediate dimension sizes.
        laten_dim (int): Dimensional size of latent space.
        loss (str): Loss function to add to the VAE model.
        
    Returns:
        tensorflow.keras.model: Encoder model
    """
    
    input_shape = (original_dim, )
    
    encoder = vae_encoder_model(input_shape, intermediate_dim, latent_dim)
    decoder = vae_decoder_model(latent_dim, intermediate_dim, original_dim)
    
    inputs = keras.Input(shape=input_shape, name='inputs')
    outputs = decoder(encoder(inputs)[2])

    vae = keras.Model(inputs, outputs, name='vae')
    vae_loss = vae_loss_func(encoder, inputs, outputs, loss, original_dim)
    vae.add_loss(vae_loss)
    
    return vae, encoder, decoder


def vae_loss_func(encoder, inputs, outputs, loss, original_dim):
    """ Define loss function.
    
    Args:
        encoder (tensorflow.keras.model): Encoder part of the VAE model.
        inputs (tensorflow.Tensor): Ground truth values.
        outputs (tensorflow.Tensor): The predicted values
        loss (str): Loss function to add to the VAE model.
        original_dim (int): Dimensional size of original data.
        
    Returns:
        tensorflow.keras.model: Encoder model
    """
    
    z_mean = encoder(inputs)[0]
    z_log_sigma = encoder(inputs)[1]
    
    if loss == 'kl-mse':
        print('using kl-mse')
        reconstruction_loss = keras.losses.MSE(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    elif loss == 'kl-bce':
        print('using kl-bce')
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    else:
        print('using only mse')
        vae_loss = keras.losses.MSE(inputs, outputs)
       
    return vae_loss        


class DeepAnt(keras.Model):
    
    def __init__(self, timesteps, original_dim, kernel_size=3, n_filters=8, padding='same', pool_size=2, dense_layer_size=24, dropout=0.25):
        super(DeepAnt, self).__init__()
  
        self.conv1d_1 = layers.Conv1D(n_filters, kernel_size, activation='relu', padding=padding, name='conv1d_1')
        self.maxpool_1 = layers.MaxPooling1D(pool_size=pool_size, name='maxpool_1')

        self.conv1d_2 = layers.Conv1D(n_filters, kernel_size, activation='relu', padding=padding, name='conv1d_2')
        self.maxpool_2 = layers.MaxPooling1D(pool_size=pool_size, name='maxpool_2')

        self.flatten = layers.Flatten(name='flatten')
        self.dense_1 = layers.Dense(dense_layer_size, name='dense_1', activation='relu')

        self.dropout = layers.Dropout(dropout, name='dropout')
        self.dense_2 = layers.Dense(original_dim, name='dense_2', activation='relu')

        
    def call(self, inputs, training=False):
        x = self.conv1d_1(inputs)
        x = self.maxpool_1(x)

        x = self.conv1d_2(x)
        x = self.maxpool_2(x)

        x = self.flatten(x)
        x = self.dense_1(x)

        x = self.dropout(x)
        x = self.dense_2(x)

        return x


class LSTM(keras.Model):
    
    def __init__(self, neurons, timesteps, original_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstms = []
        for neuron in neurons[:-1]:
            self.lstms.append(layers.LSTM(neuron, input_shape=(timesteps, original_dim), recurrent_dropout=dropout, return_sequences=True))
        self.lstms.append(layers.LSTM(neurons[-1], input_shape=(timesteps, original_dim), recurrent_dropout=dropout))
        self.dense = layers.Dense(original_dim)
        
        
    def call(self, inputs, training=False):
        x = self.lstms[0](inputs)    
        for lstm in self.lstms[1:]:
            x = lstm(x)
        x = self.dense(x)

        return x


class AE(keras.Model):

    def __init__(self, original_dim, intermediate_dims, laten_dim):
        super(AE, self).__init__()

        # Encoder
        self.input_layer = layers.Dense(original_dim,  name='input')
        self.encoder_layers = []
        for dim in intermediate_dims:
            self.encoder_layers.append(layers.Dense(dim, activation='relu'))
        self.latent_layer = layers.Dense(laten_dim, activation='relu', name='latent')

        # Decoder
        self.decoder_layers = []
        for dim in reversed(intermediate_dims):
            self.decoder_layers.append(layers.Dense(dim, activation='relu'))
        self.output_layer = layers.Dense(original_dim, activation='sigmoid', name='output')

    
    def call(self, inputs, training=False):
        # Encoder
        x = self.input_layer(inputs)
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
        x = self.latent_layer(x)
        
        # Decoder
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x)
        x = self.output_layer(x)

        return x
