from dmd import load_dmd_modes, load_dmd_reconstruction
from sensor_placement import load_sensorplacement, reconstruction_movie
from pod_analysis import load_POD
from temperature_simulation import load_simulations
import numpy as np
import os
import matplotlib.pyplot as plt

absolute_path = os.path.dirname(__file__)

sensor_dmd_file =  "data/temperature_sensor_dmd_reconstruction.npy"

def sensor_dmd_prediction(C, Theta_inv, D, W, W_inv, U, X0, nsteps):
    X = np.zeros((nsteps, *X0.shape))
    Y0 = C @ X0
    a_hat_0 = Theta_inv @ Y0
    X[0] = U @ a_hat_0

    D_i = np.diag(D)
    for i in range(1, nsteps):
        X[i] = U @ W @ D_i @ W_inv @ a_hat_0
        D_i = D_i @ np.diag(D)
    return X


def sensor_dmd_analysis(modes_used, num_steps=None):
    data = load_simulations()
    nsims, nsteps, nx, ny = data.shape
    if num_steps is None:
        num_steps = nsteps
    U,L, RIC, X_mean =load_POD()
    U = U[:,:modes_used]
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()
    C, Theta_inv = load_sensorplacement()

    X0 = data[:,0,:,:].reshape((nsims, nx*ny))
    X0 = (X0 - X_mean).T
    
    X = sensor_dmd_prediction(C, Theta_inv, D, W, W_inv, U, X0, num_steps)
    X = np.rollaxis(X,2,0) + X_mean
    X = X.reshape((nsims,num_steps,nx, ny))
    np.save(sensor_dmd_file, X)
    return X
    
def load_sensor_dmd_reconstruction(file = None):
    if file is None:
        file = sensor_dmd_file
    abs_path = os.path.join(absolute_path, file)
    return np.load(abs_path)

def plot_error(sim=None, exp=2):
    X_ = load_simulations()
    X_sensor_dmd_ = load_sensor_dmd_reconstruction()
    if sim is None:
        current_sim = np.random.randint(0, X_.shape[0])
    else:
        current_sim = sim
   
    while sim is None or sim == current_sim:
        X = X_[current_sim]
        X_sensor_dmd = X_sensor_dmd_[current_sim]
        error = np.abs(X - X_sensor_dmd)**exp
        se = np.sum(error, axis=(1,2))
        plt.plot(se)
        plt.show()
        current_sim = np.random.randint(0, X.shape[0])

def plot_all_error(exp=2):
    X = load_simulations()
    X_sensor_dmd = load_sensor_dmd_reconstruction()
    error = np.abs(X - X_sensor_dmd)**exp
    se = np.sum(error, axis=(2,3))
    plt.plot(se.T)
    plt.show()
    

def show_random_reconstruction_movies():
    X = load_simulations()
    X_dmd = load_dmd_reconstruction()
    X_sensor_dmd = load_sensor_dmd_reconstruction()
    while True:
        sim_number = np.random.randint(0, X.shape[0])
        reconstruction_movie(X, X_dmd, X_sensor_dmd, sim_number=sim_number, dt=0.01)

if __name__ == "__main__":
    #sensor_dmd_analysis(8)
    plot_all_error(exp=1)
    
