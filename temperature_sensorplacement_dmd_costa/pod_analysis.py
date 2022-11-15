import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.animation as animation
from temperature_simulation import load_simulations, data_folder


absolute_path = os.path.dirname(__file__)



POD_file =  "data/temperature_simulations_POD.npy"
reconstruction_file =  "data/temperature_reconstruction_POD.npy"



###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = la.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    Sigma = S[:R]
    Vr = Vh[:R,:]

    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi, Sigma, Vr, L,RIC, 

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

def load_limited_data(max_simulations = 10,max_time_steps = 10):

    simulations = []    
    simulation_count = 0
    for file in os.listdir(data_folder):
        if file.endswith(".npy"):
            simulation_count += 1
            simulations.append(np.load(data_folder + file)[:max_time_steps])
            if simulation_count >= max_simulations:
                break
    return np.array(simulations)

def find_POD(n_modes = 10, mean_center=True):
    print("Loading data...")
    data = load_simulations()
 
    nsims, nsteps, nx, ny = data.shape

    flat_data = data.reshape((nsims*nsteps, nx*ny))

    X_mean = np.zeros((nx*ny))
    if mean_center:
        X_mean = np.mean(flat_data, axis=0)
    flat_data = flat_data - X_mean
    
    print("Finding POD...")
    Phi, Sigma, Vr, L, RIC = POD(flat_data.T, n_modes)
    print("Saving POD...")
    
    print("RIC: ", RIC)

    np.save(POD_file, {"phi": Phi, "l": L, "ric": RIC, "mean": X_mean, "sigma": Sigma, "vr": Vr})
    return Phi, L, RIC, X_mean


def load_POD(file = None):
    if file is None:
        file = POD_file
    abs_path = os.path.join(absolute_path, file)
    data = np.load(abs_path, allow_pickle=True).item()
    return data["phi"], data["l"], data["ric"], data["mean"]

def load_reconstruction(file = None):
    if file is None:
        file = reconstruction_file
    abs_path = os.path.join(absolute_path, file)
    return np.load(abs_path)


def reconstruction_analysis(n_modes_used=4):
    print("Loading data...")
    data = load_simulations()
    print(data.shape)
    nsims, nsteps, nx, ny = data.shape
    flat_data = data.copy().reshape((nsims*nsteps, nx*ny))

    print("Loading POD...")
    Phi, L, RIC, mean = load_POD()
    print("Explained variance:")
    print((L[:n_modes_used]/sum(L)*100).cumsum())
    X = flat_data - mean
    Phi_used = Phi[:,:n_modes_used]
    print("Reconstructing...")

    proj = PODproj(X.T, Phi_used)

    reconstruction = PODrec(proj, Phi_used)

    reconstruction = reconstruction.T + mean

    reconstruction = reconstruction.reshape((nsims, nsteps, nx, ny))
    np.save(reconstruction_file, reconstruction)
    print("Done")
    return data, reconstruction


def reconstruction_movie(X,X_rec, sim_number=0,dt=0.1):
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    sim = X[sim_number]
    rec = X_rec[sim_number]
    min = np.min((np.min(sim), np.min(rec)))
    max = np.max((np.max(sim), np.max(rec)))
    max_error = np.max(np.abs(sim-rec))
    im = ax.imshow(sim[0], cmap="hot", vmin=min, vmax=max)
    im2 = ax2.imshow(rec[0], cmap="hot",vmin=min, vmax=max)
    im3 = ax3.imshow(np.abs(sim[0]-rec[0]), cmap="gray",vmin=0, vmax=max_error)

    def updatefig(j):
        im.set_array(sim[j])
        im2.set_array(rec[j])
        im3.set_array(np.abs(sim[j]-rec[j]))
        return im, im2, im3

    ani = animation.FuncAnimation(fig, updatefig, frames=range(X.shape[1]), interval=dt*1000, blit=True)

    plt.show()
    


if __name__ == "__main__":
    
    #find_POD(n_modes=20, mean_center=False)
    data, reconstruction = reconstruction_analysis(8)
    #data = load_simulations()
    #reconstruction = load_reconstruction()
    
    while True:
       sim_number = np.random.randint(0, data.shape[0])
       reconstruction_movie(data, reconstruction, sim_number=sim_number, dt=0.01)
        