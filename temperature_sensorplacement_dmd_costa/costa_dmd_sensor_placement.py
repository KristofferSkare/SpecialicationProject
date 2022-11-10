import numpy as np
import tensorflow as tf
from sensor_placement import load_sensorplacement
from dmd import load_dmd_modes
from pod_analysis import load_POD, load_limited_data
from temperature_simulation import load_simulations
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SensorPlacementPredictionGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, C, Ur, num_prediction_steps=1, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        

        nsims, nsteps, nvars = X.shape
        self.n_simulations = nsims
        self.n_steps = nsteps
        self.n_vars = nvars
        self.n_pred_steps = num_prediction_steps

        self.X = X
        self.C = C
        self.Ur = Ur

        self.Y = (C @ X.reshape(nsims*nsteps, nvars).T).T.reshape(nsims, nsteps, C.shape[0])
        self.a = (Ur @ X.reshape(nsims*nsteps, nvars).T).T.reshape(nsims, nsteps, C.shape[0])

        self.indexes = np.arange(nsims*(nsteps-num_prediction_steps))

        self.len = self.indexes.size

        self.on_epoch_end()

    
    def __len__(self):
        return self.len//self.batch_size
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        sim_nums = np.floor_divide(indexes, self.n_steps-self.n_pred_steps)
        step_nums = np.mod(indexes, self.n_steps-self.n_pred_steps)

        
        pred_steps = np.arange(self.n_pred_steps+1) + step_nums[:, np.newaxis]
   
        X_batch = self.X[sim_nums[:, np.newaxis], pred_steps, :]
        a_batch = self.a[sim_nums[:, np.newaxis], pred_steps, :]
        Y_batch = self.Y[sim_nums, step_nums, :]

        return Y_batch, [X_batch, a_batch]
    
    def on_epoch_end(self):
        if (self.shuffle):
            np.random.shuffle(self.indexes)



def create_model(
    Theta_inv, A_tilde, U,
    nsteps = 10,
    prediction_layers = [16,32,64],
    sensor_layers = [16,32,64],
    reconstruction_layers = [512,256,512,2500],
    l1_weight = 0.01,
    zero_init = False
    ):

    initilizer = tf.keras.initializers.GlorotNormal()

    if zero_init:
        initilizer = tf.keras.initializers.Zeros()

    regularizer = tf.keras.regularizers.L1(l1_weight)

    n,r = U.shape

    # Defining layers

    Theta_inv_mul = tf.keras.layers.Dense(r, use_bias=False, name="Theta_inv", kernel_initializer=tf.keras.initializers.Constant(Theta_inv.T))

    A_mul = tf.keras.layers.Dense(r, use_bias=False, name="A",kernel_initializer=tf.keras.initializers.Constant(A_tilde.T))

    U_mul = tf.keras.layers.Dense(n, use_bias=False, name="U", kernel_initializer=tf.keras.initializers.Constant(U.T))

    add = tf.keras.layers.Add()
    # Model for correcting reconstruction from sparse sensors to latent space
    sensor_source_term = tf.keras.Sequential(
        [tf.keras.layers.Dense(layer, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initilizer) for layer in [*sensor_layers, r]]
        , name="sensor_correction")

    # Model for correcting future prediction in latent space
    prediction_source_term = tf.keras.Sequential(
        [tf.keras.layers.Dense(layer, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initilizer) for layer in [*prediction_layers, r]]
        ,name="prediction_correction")
    
    # Model for correcting reconstructing from latent space to full space
    reconstruction_source_term = tf.keras.Sequential(
        [tf.keras.layers.Dense(layer, activation='relu', kernel_regularizer=regularizer, kernel_initializer=initilizer) for layer in [*reconstruction_layers, n]]
        ,name="reconstruction_correction")

    # Create model
    Y = tf.keras.layers.Input(shape=(r), name="Y")
    a_hat_i = Theta_inv_mul(Y)
    a_hat = [a_hat_i]


    
    sigma_s = sensor_source_term(a_hat_i)

    a_tilde_i = add([a_hat_i,sigma_s])
    a_tilde = [a_tilde_i]

    X_hat_i = U_mul(a_tilde_i)
    X_hat = [X_hat_i]

    sigma_r = reconstruction_source_term(X_hat_i)

    X_tilde_i = add([X_hat_i ,sigma_r])
    X_tilde= [X_tilde_i]

    for i in range(1, nsteps+1):

        a_hat_i = A_mul(a_tilde_i)
        sigma_p = prediction_source_term(a_hat_i)
        a_tilde_i = add([a_hat_i ,sigma_p])

        X_hat_i = U_mul(a_tilde_i)
        sigma_r = reconstruction_source_term(X_hat_i)
        X_tilde_i = add([X_hat_i,sigma_r])

        a_hat.append(a_hat_i)
        a_tilde.append(a_tilde_i)

        X_hat.append(X_hat_i)
        X_tilde.append(X_tilde_i)

    reshape_1n = tf.keras.layers.Reshape((1, n))
    reshape_1r = tf.keras.layers.Reshape((1, r))
    X_tilde_all = tf.keras.layers.Concatenate(axis=1, name="X")([reshape_1n(X) for X in X_tilde])
    a_tilde_all = tf.keras.layers.Concatenate(axis=1, name="a")([reshape_1r(a) for a in a_tilde])

    model = tf.keras.Model(inputs=Y, outputs=[X_tilde_all, a_tilde_all])
    # Values i a is generally 10* values in X, and loss in a is less important. scale A mse with (0.1)**2 * 0.01
    model.compile(optimizer='adam', loss='mse', loss_weights={"X": 1, "a": 0.0001})

    model.get_layer("Theta_inv").trainable = False
    model.get_layer("A").trainable = False
    model.get_layer("U").trainable = False

    model.summary()

    return model

def train_model(num_pred_steps=2, num_modes_used=8, epochs=1):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Load data
 
    data = load_simulations()
    #data = load_limited_data(max_simulations=20, max_time_steps=100)
    U, L, RIC, X_mean = load_POD()
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement()
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()

    X = data.reshape(data.shape[0], data.shape[1], np.product(data.shape[2:]))
    X = X - X_mean

    model = create_model(Theta_inv, Atilde, Ur, nsteps=num_pred_steps, 
    sensor_layers=[16,32,64,64, 64],
    prediction_layers=[16,32,64,64,64],
     reconstruction_layers=[1024,512,256,128,64],
    l1_weight=1e-1
    )

    # Train val split
    val_size = 0.2
    n = X.shape[0]
    sims = np.arange(0, n)
    np.random.shuffle(sims)
    val_cutof = int(val_size*n)
    val_sims = sims[:int(n*val_size)]
    train_sims = sims[val_cutof:]

    train_generator = SensorPlacementPredictionGenerator(X[train_sims], C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=32, shuffle=True)
    val_generator = SensorPlacementPredictionGenerator(X[val_sims], C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=32, shuffle=True)

    cb = tf.keras.callbacks.ModelCheckpoint("temperature_sensorplacement_dmd_costa/costa_models/modes_used_"+str(num_modes_used) + "/model_1/cp-{epoch:02d}.ckpt", monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
  
    model.fit(train_generator, epochs=epochs, verbose=1, callbacks=[cb],validation_data=val_generator)
    

def load_model(num_pred_steps=2, num_modes_used=8, zero_init=False):
    model_folder = "temperature_sensorplacement_dmd_costa/costa_models/modes_used_"+str(num_modes_used) + "/model_1/"
    
    U, L, RIC, X_mean = load_POD()
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement()
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()
    
    model = create_model(Theta_inv, Atilde, Ur, nsteps=num_pred_steps, 
    sensor_layers=[16,32,64,64, 64],
    prediction_layers=[16,32,64,64,64],
    reconstruction_layers=[1024,512,256,128,64],
    l1_weight=1e1,
    zero_init=zero_init
    )

    if not zero_init:
        latest_checkpoint = tf.train.latest_checkpoint(model_folder)
        model.load_weights(latest_checkpoint)

    return model

def show_predictions(num_pred_steps=50, num_modes_used=8, simulation=0, zero_init=False, interval_pred_steps=1):
    model = load_model(num_pred_steps=num_pred_steps, num_modes_used=num_modes_used, zero_init=zero_init)

    data = load_simulations()
    #data = load_limited_data(max_simulations=10, max_time_steps=100)
    U, L, RIC, X_mean = load_POD()
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement()
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()

    n_sims, n_steps, nx, ny = data.shape 
    X = data.reshape((n_sims, n_steps, np.product(data.shape[2:])))
    X = X - X_mean
    
    generator = SensorPlacementPredictionGenerator(X, C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=1, shuffle=False)
    len_of_series = n_steps-num_pred_steps
    predictions = np.zeros((num_pred_steps+1,n_steps-num_pred_steps,nx,ny))
    for i in range(len_of_series):
        y, x = generator[simulation*len_of_series + i]
        X_tilde, a_tilde = model.predict(y)
        pred = X_tilde + X_mean 
        pred = pred.reshape((num_pred_steps+1, nx, ny))
        predictions[:,i] = pred

    reconstruction_movie(data[simulation], predictions, prediction_interval=interval_pred_steps, dt=0.01)

    
def reconstruction_movie(X, X_rec,prediction_interval,dt=0.01):
    n_steps, nx, ny = X.shape
    n_pred_steps = X_rec.shape[0]-1
    
    num_cols =(n_pred_steps + 1)//prediction_interval
    num_rows = 3
    fig,ax = plt.subplots(num_rows,num_cols)
    
    min = np.min(X)
    max = np.max(X)
    error = np.zeros((n_steps,n_pred_steps+1, nx, ny))
    for i in range(n_steps-n_pred_steps-1):
        for j in range(n_pred_steps+1):
            error[i,j] = X[i+j] - X_rec[j,i]

    error_max = np.max(np.abs(error))

    sim_ims = [ax[0,i].imshow(X[i*prediction_interval], vmin=min, vmax=max, cmap="hot") for i in range(num_cols)]
    pred_ims = [ax[1,i].imshow(X_rec[i*prediction_interval,0], vmin=min, vmax=max, cmap="hot") for i in range(num_cols)]
    error_ims = [ax[2,i].imshow(X_rec[i*prediction_interval,0]-X[0], vmin=0, vmax=error_max,cmap="gray") for i in range(num_cols)]


    def updatefig(j):
        for i in range(num_cols):
            sim_ims[i].set_array(X[i*prediction_interval+j])
            pred_ims[i].set_array(X_rec[i*prediction_interval,j])
            error_ims[i].set_array(X_rec[i*prediction_interval,j]-X[i*prediction_interval+j])

        return *sim_ims, *pred_ims, *error_ims

    ani = animation.FuncAnimation(fig, updatefig, frames=range(X.shape[0] - n_pred_steps-1), interval=dt*1000, blit=True)

    plt.show()

if __name__ == "__main__":
    #train_model(num_pred_steps=40, num_modes_used=8, epochs=10)
    #show_predictions(num_pred_steps=39, num_modes_used=8, zero_init=False, simulation=100, interval_pred_steps=8)
    

    #Y,X,a= format_data(data, Ur, C, X_mean, batch_size=32, prediction_steps=50)
    #np.save("costa_data.npy",{ "Y":Y, "X":X, "a":a})

    #
    pass

    





