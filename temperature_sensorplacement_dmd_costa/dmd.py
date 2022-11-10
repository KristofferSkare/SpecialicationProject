from pod_analysis import load_POD, PODproj, PODrec, POD_file, reconstruction_file
from temperature_simulation import load_simulations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DMD_file = "temperature_sensorplacement_dmd_costa/data/temperature_DMD.npy"
DMD_rec_file = "temperature_sensorplacement_dmd_costa/data/temperature_reconstruction_DMD.npy"

def format_dmd_data(data, X_mean, Vr):
    nsims, nsteps, nx, ny = data.shape
    X1 = np.zeros((nsims*(nsteps-1), nx*ny))
    X2 = np.zeros((nsims*(nsteps-1), nx*ny))
    Vr_used = np.zeros((Vr.shape[0], nsims*(nsteps-1)))
    for i in range(nsims):
        X1[i*(nsteps-1):(i+1)*(nsteps-1),:] = data[i,:-1,:,:].reshape((nsteps-1, nx*ny))
        X2[i*(nsteps-1):(i+1)*(nsteps-1),:] = data[i,1:,:,:].reshape((nsteps-1, nx*ny))
        Vr_used[:,i*(nsteps-1):(i+1)*(nsteps-1)] = Vr[:,i*(nsteps):(i+1)*nsteps -1]
    
    X1 = (X1 - X_mean).T
    X2 = (X2 - X_mean).T

    return X1, X2, Vr_used.T

def DMD(data, Ur, Sigma, Vr, X_mean):
    X1, X2, Vr_used = format_dmd_data(data, X_mean, Vr)
    Sigma_inv = np.diag(1/Sigma)
    Atilde = Ur.T @ X2 @ Vr_used @ Sigma_inv
    D, W = np.linalg.eig(Atilde)
    W_inv = np.linalg.inv(W)
    print("DMD eigenvalues: ", D)
    Phi = X2 @ Vr_used @ Sigma_inv @ W
    
    np.save(DMD_file, {"Atilde": Atilde, "D": D, "W": W,"W_inv":W_inv,  "dmd_modes": Phi})
    return Atilde, D, W, W_inv, Phi

def load_whole_POD():
    data = np.load(POD_file, allow_pickle=True).item()
    return data["phi"], data["l"], data["ric"], data["mean"], data["sigma"], data["vr"]

def dmd_prediction(D,W,W_inv,U, X0, nsteps):
    X = np.zeros((nsteps, *X0.shape))
    X[0] = U @ U.T @ X0
    D_i = np.diag(D)
    for i in range(1, nsteps):
        X[i] = U @ W @ D_i @ W_inv @ U.T @ X0
        D_i = D_i @ np.diag(D)
    return X

def load_dmd_modes():
    data = np.load(DMD_file, allow_pickle=True).item()
    return data["Atilde"], data["D"], data["W"], data["W_inv"], data["dmd_modes"]

def dmd_analysis_and_reconstruction(modes_used=None, calculate_dmd=True, num_steps=None):
    data = load_simulations()
    Phi, L, RIC, X_mean, Sigma, Vr = load_whole_POD()
    if modes_used is not None:
        Phi = Phi[:,:modes_used]
        Sigma = Sigma[:modes_used]
        Vr = Vr[:modes_used,:]
    
    if calculate_dmd:
        print("Calculating DMD modes")
        Atilde, D, W, W_inv, dmd_modes = DMD(data, Phi, Sigma, Vr, X_mean)
    else:
        Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()
        
    nsims, nsteps, nx, ny = data.shape
    if (num_steps is None) or (num_steps > nsteps):
        num_steps = nsteps
    X0 = data[:,0,:,:].reshape((nsims, nx*ny))
    X0 = (X0 - X_mean).T
    print("Calculating DMD reconstruction")
    X = dmd_prediction(D,W, W_inv,Phi, X0, num_steps)
    X = np.rollaxis(X,2,0) + X_mean
    X = X.reshape((nsims,num_steps,nx, ny))
    if num_steps == nsteps:
        np.save(DMD_rec_file, X)
    else:
        np.save(DMD_rec_file.replace(".npy", "_" + str(num_steps) +"_steps"+ ".npy"), X)
    return data, X


def load_dmd_reconstruction(num_steps=None):
    if num_steps is None:
        return np.load(DMD_rec_file)
    else:
        return np.load(DMD_rec_file.replace(".npy", "_" + str(num_steps) +"_steps"+ ".npy"))

def reconstruction_movie(X, X_rec, sim_number=0, dt=0.01):

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    sim = X[sim_number]
    rec = X_rec[sim_number]
    
    min = np.min((np.min(sim), np.min(rec)))
    max = np.max((np.max(sim), np.max(rec)))
    max_error = np.max(np.abs(sim-rec))
    im = ax.imshow(sim[0], cmap="hot",vmin=min, vmax=max)
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
    #dmd_analysis_and_reconstruction(modes_used=8,calculate_dmd=True)
    X = load_simulations()
    X_rec = load_dmd_reconstruction()
    #error = np.mean(np.abs(X-X_rec), axis=(2,3))
    

    while True:
        sim_number = np.random.randint(0, X.shape[0])
        #print(error[sim_number, :10])
        #plt.plot(error[sim_number])
        #plt.show()
        reconstruction_movie(X, X_rec, sim_number=sim_number, dt=0.01)
    
    pass