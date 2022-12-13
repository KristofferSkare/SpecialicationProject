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

def find_POD(n_modes = 10, mean_center=True, file = None):
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
    if file is None:
        file = POD_file
    file = os.path.join(absolute_path, file)
    np.save(file, {"phi": Phi, "l": L, "ric": RIC, "mean": X_mean, "sigma": Sigma, "vr": Vr})
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


def reconstruction_analysis(n_modes_used=4, pod_file = None, rec_file = None):
    if pod_file is None:
        pod_file = POD_file
    if rec_file is None:
        rec_file = reconstruction_file
    print("Loading data...")
    data = load_simulations()
    print(data.shape)
    nsims, nsteps, nx, ny = data.shape
    flat_data = data.copy().reshape((nsims*nsteps, nx*ny))

    print("Loading POD...")
    Phi, L, RIC, mean = load_POD(file = pod_file)
    print("Explained variance:")
    print((L[:n_modes_used]/sum(L)*100).cumsum())
    X = flat_data - mean
    Phi_used = Phi[:,:n_modes_used]
    print("Reconstructing...")

    proj = PODproj(X.T, Phi_used)

    reconstruction = PODrec(proj, Phi_used)

    reconstruction = reconstruction.T + mean

    reconstruction = reconstruction.reshape((nsims, nsteps, nx, ny))
    np.save(os.path.join(absolute_path, rec_file), reconstruction)
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
    
def plot_POD_modes(n_modes=8, file=None):
    nx, ny = 50, 50
    phi, l, ric, mean = load_POD(file=file)
    print("Explained variance:", (l[:n_modes]/sum(l)*100).cumsum())
    modes = phi[:,:n_modes]
    n_cols = 4
    fig, ax = plt.subplots(int(np.ceil((n_modes)/n_cols)), n_cols)
    ax = ax.flatten()
    vmax = np.max(modes)
    vmin = np.min(modes)
    for i in range(n_modes):
        ax[i].imshow(modes[:,i].reshape((nx,ny)), cmap="hot", vmin=vmin, vmax=vmax)
        ax[i].set_title("Mode " + str(i+1))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    cm = plt.cm.get_cmap('hot')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, format="%4.1e", cax=cbar_ax)
    #plt.suptitle("POD modes")
    plt.show()

def error_analysis(rec_file=reconstruction_file):
    rec = load_reconstruction(file=rec_file)
    data = load_simulations()
    error = np.abs(data-rec)
    mean_per_pixel = np.mean(error, axis=(0,1))
    plt.imshow(mean_per_pixel, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    #plt.title("Mean error per position")
    plt.show()

    mean_per_time_and_sim = np.mean(error, axis=(2,3))
    mean = np.mean(mean_per_time_and_sim, axis=0)
    std = np.std(mean_per_time_and_sim, axis=0)
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
    #plt.title("Mean absolute error over time")
    plt.xlabel("Time step")
    plt.ylabel("MAE")
    plt.show()

def analysis_of_mean_centering():
    nx, ny = 50, 50
    Phi, L, RIC, mean = load_POD()
    Phi_m, L_m, RIC_m, mean_m = load_POD(file=POD_file.replace(".npy", "_mean_centered.npy"))
    
    plt.imshow(mean_m.reshape((nx,ny)), cmap="hot")
    #plt.title("Mean")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

    n_modes = 8

    diff = np.abs(Phi[:,:n_modes] - Phi_m[:,:n_modes])
    max_diff = np.max(diff)
    print("Max difference:", np.max(diff, axis=0))
    print("Mean difference:", np.mean(diff, axis=0))
    print("Relative difference:", np.mean(diff, axis=0)/np.mean(np.abs(Phi[:,:n_modes]), axis=0))

    n_cols = 4
    fig, ax = plt.subplots(int(np.ceil((n_modes)/n_cols)), n_cols)
    ax = ax.flatten()
    for i in range(n_modes):
        ax[i].imshow(diff[:,i].reshape((nx,ny)), cmap="gray", vmin=0, vmax=max_diff)
        ax[i].set_title("Mode " + str(i+1))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    cm = plt.cm.get_cmap('gray')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_diff))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
   
    fig.colorbar(sm, format="%4.1e", cax=cbar_ax)
    #plt.suptitle("Difference between mean centered and non-mean centered POD modes")
  
    plt.show()

    rec = load_reconstruction()
    rec_m = load_reconstruction(file=reconstruction_file.replace(".npy", "_8_modes_mean_centered.npy"))
    rec_diff = np.abs(rec-rec_m)
    max_diff = np.max(rec_diff)
    print("Max difference:", max_diff)
    rec_diff_per_sim_and_time = np.mean(rec_diff, axis=(2,3))

    plt.plot(rec_diff_per_sim_and_time.T)
    plt.show()

    rec_diff_per_pixel = np.mean(rec_diff, axis=(0,1))
    plt.imshow(rec_diff_per_pixel, cmap="gray", vmin=0, vmax=np.max(rec_diff_per_pixel))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    #plt.title("Mean difference per position")
    plt.show()

def show_reconstruction_4_modes(sim=0, time_step=0):
    X = load_simulations()
    X_rec = load_reconstruction(file=reconstruction_file.replace(".npy", "_4_modes.npy"))
    x = X[sim, time_step]
    x_rec = X_rec[sim, time_step]
    fig, ax = plt.subplots(2,2)
    data = np.array([x, x_rec])
    titles = ["Original", "Reconstruction"]
    vmin = np.min(data)
    vmax = np.max(data)
    for i in range(2):
        ax[0,i].imshow(data[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[0,i].set_title(titles[i])
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
    cm = plt.cm.get_cmap('hot')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, format="%4.1e", ax=ax[0,:])

    error = np.abs(x-x_rec)

    data = np.array([error])
    titles = ["Error"]
    vmin = np.min(data)
    vmax = np.max(data)
    for i in range(1):
        ax[1,i+1].imshow(data[i], cmap="gray", vmin=vmin, vmax=vmax)
        ax[1,i+1].set_title(titles[i])
        ax[1,i+1].set_xticks([])
        ax[1,i+1].set_yticks([])
    cm = plt.cm.get_cmap('gray')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, format="%4.1e", ax=ax[1,:])
    ax[1,0].axis("off")
  
    plt.show()

def show_reconstruction(sim=0, time_step=0):
    X = load_simulations()
    X_rec = load_reconstruction()
    X_rec_m = load_reconstruction(file=reconstruction_file.replace(".npy", "_8_modes_mean_centered.npy"))
    x = X[sim, time_step]
    x_rec = X_rec[sim, time_step]
    x_rec_m = X_rec_m[sim, time_step]
    fig, ax = plt.subplots(2,3)
    data = np.array([x, x_rec, x_rec_m])
    titles = ["Original", "Reconstruction", "Reconstruction (mean centered)"]
    vmin = np.min(data)
    vmax = np.max(data)
    for i in range(3):
        ax[0,i].imshow(data[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[0,i].set_title(titles[i])
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
    cm = plt.cm.get_cmap('hot')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, format="%4.1e", ax=ax[0,:])

    error = np.abs(x-x_rec)
    error_m = np.abs(x-x_rec_m)
    data = np.array([error, error_m])
    titles = ["Error", "Error (mean centered)"]
    vmin = np.min(data)
    vmax = np.max(data)
    for i in range(2):
        ax[1,i+1].imshow(data[i], cmap="gray", vmin=vmin, vmax=vmax)
        ax[1,i+1].set_title(titles[i])
        ax[1,i+1].set_xticks([])
        ax[1,i+1].set_yticks([])
    cm = plt.cm.get_cmap('gray')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, format="%4.1e", ax=ax[1,:])
    ax[1,0].axis("off")
  
    plt.show()

def format_exp(string):
    num, exp = string.split("e")
    return num + " \cdot 10^{" + str(int(exp)) + "}"

def explained_variance(file, num_modes=8):
    Phi, l, ric, X = load_POD(file=file)
    explained_variance = np.cumsum(l)/np.sum(l)
    return explained_variance[:num_modes]

if __name__ == "__main__":
    # exp = [explained_variance(file=POD_file, num_modes=8), explained_variance(file=POD_file.replace(".npy", "_mean_centered.npy"), num_modes=8)]
    # res = [[format_exp("{:.1e}".format(1-i)) for i in res] for res in exp]
    # res = np.array(res).T
    # for i, r in enumerate(res):
    #     print(str(i+1) +" & "+ " & ".join(r) + " \\\\")


    #plot_POD_modes(8 , file = POD_file.replace(".npy", "_mean_centered.npy"))
    show_reconstruction_4_modes(time_step=0, sim=2)
    #analysis_of_mean_centering()
    #error_analysis(rec_file=reconstruction_file.replace(".npy", "_8_modes_mean_centered.npy")
    #)
    
   
    #p_file = POD_file#.replace(".npy", "_mean_centered.npy")
    # r_file = reconstruction_file.replace(".npy", "_8_modes_mean_centered.npy")
    # #find_POD(n_modes=20, mean_center=False, file = p_file)
    #data, reconstruction = reconstruction_analysis(4, pod_file = POD_file, rec_file = reconstruction_file.replace(".npy", "_4_modes.npy"))
    #plot_POD_modes(8, file = p_file)
    # data = load_simulations()
    # reconstruction = load_reconstruction()
    
    # while True:
    #    sim_number = np.random.randint(0, data.shape[0])
    #    reconstruction_movie(data, reconstruction, sim_number=sim_number, dt=0.01)
    pass