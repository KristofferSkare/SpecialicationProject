
from pod_analysis import load_POD, PODproj, PODrec, POD_file, reconstruction_file
from temperature_simulation import load_simulations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import os


absolute_path = os.path.dirname(__file__)


DMD_file =  "data/temperature_DMD.npy"
DMD_rec_file = "data/temperature_reconstruction_DMD.npy"

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

def DMD(data, Ur, Sigma, Vr, X_mean, file=None):
    X1, X2, Vr_used = format_dmd_data(data, X_mean, Vr)
    Sigma_inv = np.diag(1/Sigma)
    Atilde = Ur.T @ X2 @ Vr_used @ Sigma_inv
    D, W = np.linalg.eig(Atilde)
    W_inv = np.linalg.inv(W)
    print("DMD eigenvalues: ", D)
    Phi = X2 @ Vr_used @ Sigma_inv @ W
    if file is None:
        file = DMD_file
    np.save(os.path.join(absolute_path, file), {"Atilde": Atilde, "D": D, "W": W,"W_inv":W_inv,  "dmd_modes": Phi})
    return Atilde, D, W, W_inv, Phi

def load_whole_POD(file=None):
    if file is None:
        file = POD_file
    abs_path = os.path.join(absolute_path, file)
    data = np.load(abs_path, allow_pickle=True).item()
    return data["phi"], data["l"], data["ric"], data["mean"], data["sigma"], data["vr"]

def dmd_prediction(D,W,W_inv,U, X0, nsteps):
    X = np.zeros((nsteps, *X0.shape))
    X[0] = U @ U.T @ X0
    D_i = np.diag(D)
    for i in range(1, nsteps):
        X[i] = U @ W @ D_i @ W_inv @ U.T @ X0
        D_i = D_i @ np.diag(D)
    return X

def load_dmd_modes(file=None):
    if file is None:
        file = DMD_file
    abs_path = os.path.join(absolute_path, file)
    data = np.load(abs_path, allow_pickle=True).item()
    return data["Atilde"], data["D"], data["W"], data["W_inv"], data["dmd_modes"]

def dmd_analysis_and_reconstruction(modes_used=None, calculate_dmd=True, num_steps=None, pod_file=None, dmd_file=None, dmd_rec_file=None):
    data = load_simulations()
    Phi, L, RIC, X_mean, Sigma, Vr = load_whole_POD(file=pod_file)
    if modes_used is not None:
        Phi = Phi[:,:modes_used]
        Sigma = Sigma[:modes_used]
        Vr = Vr[:modes_used,:]
    
    if calculate_dmd:
        print("Calculating DMD modes")
        Atilde, D, W, W_inv, dmd_modes = DMD(data, Phi, Sigma, Vr, X_mean, file=dmd_file)
    else:
        Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(file=dmd_file)
        
    nsims, nsteps, nx, ny = data.shape
    if (num_steps is None) or (num_steps > nsteps):
        num_steps = nsteps
    X0 = data[:,0,:,:].reshape((nsims, nx*ny))
    X0 = (X0 - X_mean).T
    print("Calculating DMD reconstruction")
    X = dmd_prediction(D,W, W_inv,Phi, X0, num_steps)
    X = np.rollaxis(X,2,0) + X_mean
    X = X.reshape((nsims,num_steps,nx, ny))

    if dmd_rec_file is None:
        dmd_rec_file = DMD_rec_file

    file = os.path.join(absolute_path, dmd_rec_file)
    if num_steps == nsteps:
        np.save(file, X)
    else:
        np.save(file.replace(".npy", "_" + str(num_steps) +"_steps"+ ".npy"), X)
    return data, X


def load_dmd_reconstruction(num_steps=None, file=None):
    if file is None:
        file = DMD_rec_file
    abs_path = os.path.join(absolute_path, file)
    if num_steps is None:
        return np.load(abs_path)
    else:
        return np.load(abs_path.replace(".npy", "_" + str(num_steps) +"_steps"+ ".npy"))

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

def plot_error(exp=2, time_steps=None):
    file = DMD_rec_file
    X = load_simulations()
    X_rec = load_dmd_reconstruction(file=file)
    X_rec_mean = load_dmd_reconstruction(file=file.replace(".npy", "_8_modes_mean_centered.npy"))
    error = np.abs(X-X_rec)**exp
    error_m = np.abs(X-X_rec_mean)**exp
    if time_steps is not None:
        error = error[:,:time_steps]
        error_m = error_m[:,:time_steps]
    error = np.mean(error, axis=(2,3))
    error_m = np.mean(error_m, axis=(2,3))
    print(np.mean(error))
    print(np.mean(error_m))
    mean_error_t = np.mean(error, axis=0)
    std_mean = np.std(error, axis=0)

    mean_error_t_m = np.mean(error_m, axis=0)
    std_mean_m = np.std(error_m, axis=0)


    plt.plot(mean_error_t)
    plt.plot(mean_error_t_m)
    plt.legend(["DMD", "DMD mean centered"])
    plt.fill_between(np.arange(mean_error_t.size), mean_error_t - std_mean,  mean_error_t + std_mean, alpha=0.25)
 
    plt.fill_between(np.arange(mean_error_t_m.size),  mean_error_t_m - std_mean_m, mean_error_t_m + std_mean_m, alpha=0.25)
    plt.xlabel("Time step")
    plt.ylabel("MAE")
    plt.show()

def show_dmd_modes(file=DMD_file):
    nx,ny = 50,50
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(file=file)
    n = dmd_modes.shape[1]
    mode_images = dmd_modes.T.reshape((n, nx, ny)) 
    fig, ax = plt.subplots(2,4)
    ax = ax.flatten()
    vmax = np.max(mode_images)
    vmin = np.min(mode_images)
    for i in range(n):
        ax[i].imshow(mode_images[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[i].set_title("Mode " + str(i+1))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    cm = plt.cm.get_cmap('hot')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, format="%4.1e", cax=cbar_ax)
    #plt.suptitle("DMD modes")
    plt.show()

def show_dmd_matrix(file=DMD_file, exp=1/4):
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(file=file)
    adjusted = np.sign(Atilde) * ((np.abs(Atilde))**exp)
    plt.figure()
    plt.imshow(adjusted, cmap="bwr", vmin=-1, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    cb = plt.colorbar()
    N = 9
    ticks = np.linspace(-1,1,N)
    cb.set_ticks(ticks)
    cb.set_ticklabels(["{:0.3f}".format(np.sign(i)*(np.abs(i)**(1/exp))) for i in ticks])
    #plt.title("DMD matrix")
    plt.show()

def show_dmd_prediction_and_error(file=DMD_rec_file,sim=0, num_images=4, start=0, end=None):
    X = load_simulations()
    X_rec = load_dmd_reconstruction(file=file)
    if end is None:
        end = X.shape[1]
    indexes = np.linspace(start, end-1, num_images, dtype=int)
    x = X[sim, indexes]
    x_rec = X_rec[sim, indexes]
    error = np.abs(x-x_rec)
    vmax = np.max((np.max(x), np.max(x_rec)))
    vmin = np.min((np.min(x), np.min(x_rec)))
    max_error = np.max(error)
    print(x_rec.shape)
    fig, ax = plt.subplots(3,num_images)
    for i in range(num_images):
        
        ax[0,i].imshow(x[i], cmap="hot", vmin=vmin, vmax=vmax)
        
        
        ax[1,i].imshow(x_rec[i], cmap="hot", vmin=vmin, vmax=vmax)
        
        ax[2,i].imshow(error[i], cmap="gray", vmin=0, vmax=max_error)

        if i == 0:
            ax[0,i].set_title("Original: t={:d}".format(indexes[i]))
            ax[1,i].set_title("Prediction: t={:d}".format(indexes[i]))
            ax[2,i].set_title("Error: t={:d}".format(indexes[i]))
        else:
            ax[0,i].set_title("t={:d}".format(indexes[i]))
            ax[1,i].set_title("t={:d}".format(indexes[i]))
            ax[2,i].set_title("t={:d}".format(indexes[i]))
        
        for j in range(3):
            ax[j,i].set_xticks([])
            ax[j,i].set_yticks([])
    cm_hot = plt.cm.get_cmap('hot')
    sm_hot = plt.cm.ScalarMappable(cmap=cm_hot, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cm = plt.cm.get_cmap('gray')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_error))
    plt.colorbar(sm_hot, format="%4.1e", ax=ax[[0,1],:])
    plt.colorbar(sm, format="%4.1e", ax=ax[2,:])
    #plt.suptitle("DMD prediction")
    plt.show()




if __name__ == "__main__":
    
    pod_file="data/temperature_simulations_POD_mean_centered.npy"
    dmd_file="data/temperature_DMD_8_modes_mean_centered.npy"
    dmd_rec_file="data/temperature_reconstruction_DMD_8_modes_mean_centered.npy"
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(file=dmd_file)
    print(" & ".join(["{:0.3f}".format(i) for i in D]))
    #show_dmd_prediction_and_error(end=290,sim=2, num_images=6, start=0, file=dmd_rec_file
    #)
    #show_dmd_matrix(
    #    exp=1,
    #    file=dmd_file
    #)
    #dmd_analysis_and_reconstruction(modes_used=8,calculate_dmd=True, 
    #pod_file=pod_file,
    #dmd_file=dmd_file,
    #dmd_rec_file=dmd_rec_file
    #)
    #plot_error(exp=1,
    #    time_steps=None
   #)
    #show_dmd_modes(file=dmd_file
    #exp=1)
    

    # X = load_simulations()
    # X_rec = load_dmd_reconstruction(#file=dmd_rec_file
    # )
    # #error = np.mean(np.abs(X-X_rec), axis=(2,3))
    

    # while True:
    #     sim_number = np.random.randint(0, X.shape[0])
    #     #print(error[sim_number, :10])
    #     #plt.plot(error[sim_number])
    #     #plt.show()
    #     reconstruction_movie(X, X_rec, sim_number=sim_number, dt=0.01)
    
    pass