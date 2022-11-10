from dmd import load_dmd_modes, load_dmd_reconstruction
from sensor_placement import load_sensorplacement, reconstruction_movie
from pod_analysis import load_POD
from temperature_simulation import load_simulations
import numpy as np

sensor_dmd_file = "temperature_sensorplacement_dmd_costa/data/temperature_sensor_dmd_reconstruction.npy"

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
    
def load_sensor_dmd_reconstruction():
    return np.load(sensor_dmd_file)

if __name__ == "__main__":
    sensor_dmd_analysis(8)
    X = load_simulations()
    X_dmd = load_dmd_reconstruction()
    X_sensor_dmd = load_sensor_dmd_reconstruction()

    while True:
        sim_number = np.random.randint(0, X.shape[0])
        reconstruction_movie(X, X_dmd, X_sensor_dmd, sim_number=sim_number, dt=0.01)
