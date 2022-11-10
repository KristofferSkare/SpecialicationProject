import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pod_analysis import load_POD, POD_file, reconstruction_file
from temperature_simulation import load_simulations
import matplotlib.animation as animation

rec_file = "temperature_sensorplacement_dmd_costa/data/temperature_reconstruction_sensors.npy"
sensor_placement_file = "temperature_sensorplacement_dmd_costa/data/temperature_sensor_placement.npy"

def qr_pivots(Psi_r, num_eigen, num_sensors=None):
    if num_sensors is None:
        num_sensors = num_eigen
    
    M = Psi_r.T
    if num_sensors > num_eigen:
        M = Psi_r @  Psi_r.T

    Q, R, P = scipy.linalg.qr(M, pivoting=True)
    return P

def find_sensor_placement(Psi_r, num_eigen, num_sensors=None):
    P = qr_pivots(Psi_r, num_eigen, num_sensors)
    C = np.zeros((num_sensors,Psi_r.shape[0]))
    C[np.arange(num_sensors),P[:num_sensors]] = 1
    return C

def create_measurements_sensor_placement(X, C, X_mean):
    Y = np.dot(C, (X - np.tile(X_mean,(1,X.shape[1]))))
    return Y

def reconstruct_from_measurements(Y, C, Psi_r, X_mean,Theta_inv = None):
    if Theta_inv is None:
        Theta_inv = np.linalg.inv(np.dot(C, Psi_r))
    X_hat = np.dot(Psi_r, np.dot(Theta_inv, Y)) + np.tile(X_mean,(1,Y.shape[1]))
    return X_hat



def find_sensor_placement_and_reconstruct(num_modes_used=None, C=None):
    data = load_simulations()
   
    nsims, nsteps, nx, ny = data.shape
    X = data.copy().reshape((nsims*nsteps, nx*ny)).T
    Phi, L, RIC, mean = load_POD()
    if num_modes_used is not None:
        Phi = Phi[:, :num_modes_used]
    X_mean = mean.reshape((nx*ny,1))
    num_components = Phi.shape[1]
    num_sensors = num_components
    if C is None:
        C = find_sensor_placement(Phi, num_components, num_sensors)
    Y = create_measurements_sensor_placement(X, C, X_mean)
    Theta_inv = np.linalg.inv(np.dot(C, Phi))
    X_hat = reconstruct_from_measurements(Y, C, Phi, X_mean, Theta_inv=Theta_inv)
    rec = X_hat.T.reshape((nsims, nsteps, nx, ny))

    np.save(rec_file, rec)
    np.save(sensor_placement_file, {"C":C, "Theta_inv": Theta_inv})
    return rec, C, Theta_inv


def reconstruction_movie(X,X_pod, X_rec, sim_number=0, dt=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    sim = X[sim_number]
    pod = X_pod[sim_number]
    rec = X_rec[sim_number]
    min = np.min((np.min(sim), np.min(rec), np.min(pod)))
    max = np.max((np.max(sim), np.max(rec), np.max(pod)))

    im = ax.imshow(sim[0], cmap="hot", vmin=min, vmax=max)
    im2 = ax2.imshow(pod[0], cmap="hot",vmin=min, vmax=max)
    im3 = ax3.imshow(rec[0], cmap="hot",vmin=min, vmax=max)

    def updatefig(j):
        im.set_array(sim[j])
        im2.set_array(pod[j])
        im3.set_array(rec[j])
        return im, im2, im3

    ani = animation.FuncAnimation(fig, updatefig, frames=range(X.shape[1]), interval=dt*1000, blit=True)

    plt.show()

def visualize_sensor_placement(C):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.sum(C, axis=0).reshape(50,50), cmap="gray")
    plt.show()

def analyze_errors(X, X_rec):
    errors = np.abs(X - X_rec)
    print("Max error: ", np.max(errors))
    print("Mean error: ", np.mean(errors))
    print("Std error: ", np.std(errors))

def load_sensorplacement():
    data = np.load(sensor_placement_file, allow_pickle=True).item()
    C = data["C"]
    Theta_inv = data["Theta_inv"]
    return C, Theta_inv

if __name__ == "__main__":
    find_sensor_placement_and_reconstruct(num_modes_used=8)
    C, Theta_inv = load_sensorplacement()
    visualize_sensor_placement(C)
    
    X = load_simulations()
    X_pod = np.load(reconstruction_file)
    X_rec = np.load(rec_file)
    analyze_errors(X, X_pod)
    analyze_errors(X, X_rec)

    while True:
       sim_number = np.random.randint(0, X.shape[0])
       reconstruction_movie(X, X_pod,X_rec, sim_number=sim_number, dt=0.01)

    pass