import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pod_analysis import load_POD
from pod_analysis import load_reconstruction as load_pod_reconstruction
from temperature_simulation import load_simulations
import matplotlib.animation as animation
import matplotlib.colors as colors
import os

absolute_path = os.path.dirname(__file__)

rec_file = "data/temperature_reconstruction_sensors.npy"
sensor_placement_file ="data/temperature_sensor_placement.npy"

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
    #plt.title("Positions of chosen sensors")
    plt.xticks([])
    plt.yticks([])
    plt.show()

def analyze_errors(X, X_rec):
    errors = np.abs(X - X_rec)
    print("Max error: ", np.max(errors))
    print("Mean error: ", np.mean(errors))
    print("Std error: ", np.std(errors))

def load_sensorplacement(file=None):
    if file is None:
        file = sensor_placement_file
    abs_path = os.path.join(absolute_path, file)
    data = np.load(abs_path, allow_pickle=True).item()
    C = data["C"]
    Theta_inv = data["Theta_inv"]
    return C, Theta_inv

def load_reconstruction(file=None):
    if file is None:
        file = rec_file 
    abs_path = os.path.join(absolute_path, file)
    return np.load(abs_path)

def plot_error(exp=2):
    X = load_simulations()
    X_rec = load_reconstruction()
    errors = np.abs(X - X_rec)**exp
    errors = np.sum(errors, axis=(2,3))
    flat_errors = errors.flatten()
    plt.figure()
    plt.hist(flat_errors, bins=np.logspace(np.log10(np.min(flat_errors)), np.log10(np.max(flat_errors)), 100))
    plt.gca().set_xscale("log")
    plt.show()

def plot_error_per_time(exp=1):
    X = load_simulations()
    X_rec = load_pod_reconstruction()
    errors = np.abs(X - X_rec)**exp
    errors = np.mean(errors, axis=(2,3))
    plt.figure()
    mean = np.mean(errors, axis=0)
    std = np.std(errors, axis=0)
    min = np.min(errors, axis=0)
    max = np.max(errors, axis=0)
    plt.plot(mean)
    plt.fill_between(np.arange(mean.size), mean-std, mean+std, alpha=0.25)
    plt.ylabel("MAE")
    plt.xlabel("Time step")
    #plt.fill_between(np.arange(mean.size), min, max, alpha=0.25)
    #plt.yscale("log")
    plt.figure()
    mean_per_sim = np.mean(errors, axis=1)
    indexes = np.argsort(mean_per_sim)
    
    mean_per_sim_sorted = mean_per_sim[indexes]
    std_per_sim = np.std(errors, axis=1)
    std_per_sim_sorted = std_per_sim[indexes]
    plt.plot(mean_per_sim_sorted)
    plt.fill_between(np.arange(mean_per_sim.size), mean_per_sim_sorted-std_per_sim_sorted, mean_per_sim_sorted+std_per_sim_sorted, alpha=0.25)
    plt.ylabel("MAE")
    plt.xlabel("Time step")
    plt.show()
    
def show_error_position(with_sensor_placement=True):
    X = load_simulations()
    X_rec = load_reconstruction()
    errors = np.abs(X - X_rec)
    errors = np.mean(errors, axis=(0,1))
    
    plt.figure()
    plt.imshow(errors, cmap="gray")
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    #plt.title("Mean absolute error per position")
    if with_sensor_placement:
        C, Theta_inv = load_sensorplacement()
        C_im = np.sum(C, axis=0).reshape(50,50)
        plt.imshow(C_im, cmap =colors.ListedColormap([(0,0,0,0), (1,0,0,1)]), vmin=0, vmax=1)
    plt.show()

def show_reconstruction(sim=0, time_step=0):
    X = load_simulations()
    X_rec = load_reconstruction()
    C, Theta_inv = load_sensorplacement()
    x = X[sim, time_step]
    x_rec = X_rec[sim, time_step]
    y = np.dot(C, x.flatten())
    y_im = (y*C.T).T.sum(axis=0).reshape(50,50).T
    fig, ax = plt.subplots(2,3)
    vmin = np.min([x, x_rec])
    vmax = np.max([x, x_rec])
    ax[0,0].imshow(x, cmap="hot", vmin=vmin, vmax=vmax)
    ax[0,0].set_title("Original")
    ax[0,1].imshow(y_im, cmap="hot", vmin=vmin, vmax=vmax)
    ax[0,1].set_title("Measurements")
    ax[0,2].imshow(x_rec, cmap="hot", vmin=vmin, vmax=vmax)
    ax[0,2].set_title("Reconstruction")

    error = np.abs(x - x_rec)
    max_error = np.max(error)
    ax[1,2].imshow(error, cmap="gray", vmin=0, vmax=max_error)
    ax[1,2].set_title("Absolute Error")

    for i in range(3):
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])

    ax[1,1].axis("off")
    ax[1,0].axis("off")

    cm_hot = plt.get_cmap("hot")
    sm = plt.cm.ScalarMappable(cmap=cm_hot, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    cm_gray = plt.get_cmap("gray")
    sm2 = plt.cm.ScalarMappable(cmap=cm_gray, norm=plt.Normalize(vmin=0, vmax=max_error))
    plt.colorbar(sm ,ax=ax[0,:])
    plt.colorbar(sm2, ax=ax[1,:])
    #plt.suptitle("Reconstruction from sparse measurements")
    plt.show()

if __name__ == "__main__":
    #show_error_position(with_sensor_placement=False)
    #show_reconstruction(sim=0, time_step=20)
    #plot_error()
    #plot_error_per_time(exp=1)
    #find_sensor_placement_and_reconstruct(num_modes_used=8)
    
    C, Theta_inv = load_sensorplacement()
    indexes = np.where(C==1)[1]
    n = 50
    print(indexes)
    positions = np.array([indexes%n, indexes//n])
    print(positions.T)
    visualize_sensor_placement(C)
    
    X = load_simulations()
    X_pod = load_pod_reconstruction()
    X_rec = load_reconstruction()
    analyze_errors(X, X_pod)
    analyze_errors(X, X_rec)

    while True:
       sim_number = np.random.randint(0, X.shape[0])
       reconstruction_movie(X, X_pod,X_rec, sim_number=sim_number, dt=0.01)
    
    pass