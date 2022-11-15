import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
import pod_analysis as pod_analysis
import sensor_placement as sensor_placement
import temperature_simulation
from PIL import Image
import numpy as np
import matplotlib

def load_simulations():
    return temperature_simulation.load_simulations()

def load_pod_modes():
    return pod_analysis.load_POD()

def load_pod_reconstruction():
    return np.load(pod_analysis.reconstruction_file.replace(".npy", "_5_modes.npy"))

def load_sensor_placement():
    sensor_data = np.load(sensor_placement.sensor_placement_file.replace(".npy", "_5_sensors.npy"), allow_pickle=True).item()
    C = sensor_data['C']
    Theta_inv = sensor_data['Theta_inv']
    return C, Theta_inv

def load_sensor_reconstruction():
    return np.load(sensor_placement.rec_file.replace(".npy", "_5_sensors.npy"))

def color_scheme(img, min_value=0, max_value=255):
    colors = (matplotlib.cm.hot(np.linspace(0, 1, 256))[:,0:3]*255).astype(np.uint8)
   
    color_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color_image[:,:] = colors[img[:,:]]
    
    return color_image

def format_temperature_image(data, min_value=None, max_value=None):
    if min_value is None:
        min_value = np.min(data)
    if max_value is None:
        max_value = np.max(data)
    data = (255*(data - min_value) / (max_value - min_value))
    data[data > 255] = 255
    data[data < 0] = 0
    color_image = color_scheme(data.astype(np.uint8))
    return color_image
    

def save_simulation_images(sim_num=0):
    data = load_simulations()
    nsims, nsteps, nx, ny = data.shape
    max = np.max(data[sim_num])
    min = np.min(data[sim_num])
    for i in range(nsteps):
        img = format_temperature_image(data[sim_num,i,:,:], min_value=min, max_value=max)
        img = Image.fromarray(img)
        img.save("simulations/simulation_" + str(sim_num) + "/original/frame_" +str(i) + ".png") 


def save_sensor_reconstruction_images(sim_num=0):
    data = load_simulations()
    C, Theta_inv = load_sensor_placement()
    X_rec = load_sensor_reconstruction()
    nsims, nsteps, nx, ny = data.shape
    max = np.max(data[sim_num])
    min = np.min(data[sim_num])
    for i in range(nsteps):
        rec = format_temperature_image(X_rec[sim_num,i,:,:], min_value=min, max_value=max)
        rec_img = Image.fromarray(rec)
        rec_img.save("simulations/simulation_" + str(sim_num) + "/sensor_reconstruction/frame_" +str(i) + ".png") 
        Y = C @ data[sim_num,i,:,:].reshape((nx*ny,1))
        measurement = np.sum(C*Y, axis=0).T.reshape((nx,ny))
        measurement = format_temperature_image(measurement, min_value=min, max_value=max)
        measurement_img = Image.fromarray(measurement)
        measurement_img.save("simulations/simulation_" + str(sim_num) + "/measurement/frame_" +str(i) + ".png")

def save_images_changing_sensors(sim_num=0, step_num=0):
    N = 300
    data = load_simulations()
  
    nsims, nsteps, nx, ny = data.shape
    max = np.max(data[sim_num])
    min = np.min(data[sim_num])
    C, Theta_inv = load_sensor_placement()
    num_sensors = C.shape[0]
    U,_,_,_ = load_pod_modes()
    Ur = U[:,0:num_sensors]
    sensors = np.random.choice(np.arange(nx*ny), (num_sensors,1))#np.array([np.where(C[i,:]==1)[0] for i in range(num_sensors)])
    change = np.zeros((N, num_sensors)).astype(int)
    #change = np.random.randint(0, nx*ny, (N, num_sensors))
    length = 15
    for i in range(N//length):
        sensor = np.random.randint(0, num_sensors)
        change[i*length:(i+1)*length,sensor] = np.random.choice([1,-1,ny,-ny])

    for i in range(N):
        sensors[:,0] += change[i,:]
        sensors[sensors < 0] += nx*ny
        sensors[sensors >= nx*ny] -= nx*ny
        C_new = np.zeros(C.shape)
        for j in range(num_sensors):
            C_new[j,sensors[j,:]] = 1
        Y = C_new @ data[sim_num,step_num,:,:].reshape((1, nx*ny)).T
        Theta_inv_new = np.linalg.pinv(C_new @ Ur)
        X_rec = Ur @ Theta_inv_new @ Y
        X_rec = X_rec.reshape((nx, ny))

        rec = format_temperature_image(X_rec, min_value=min, max_value=max)
        rec_img = Image.fromarray(rec)
        rec_img.save("simulations/simulation_" + str(sim_num) + "/moving_sensors/sensor_reconstruction/frame_" +str(step_num) + "_sensors_" + str(i) + ".png") 
        measurement = np.sum(C_new*Y, axis=0).T.reshape((nx,ny))
        measurement = format_temperature_image(measurement, min_value=min, max_value=max)
        measurement_img = Image.fromarray(measurement)
        measurement_img.save("simulations/simulation_" + str(sim_num) + "/moving_sensors/measurement/frame_" +str(step_num) + "_sensors_" + str(i) + ".png")
   
def save_images_pod_modes_sensors_theta_inv():
    data = load_simulations()
    nsims, nsteps, nx, ny = data.shape
    C, Theta_inv = load_sensor_placement()
    num_sensors = C.shape[0]
    U,_,_,_ = load_pod_modes()
    Ur = U[:,0:num_sensors]
    C_img = np.sum(C, axis=0).reshape((nx,ny))*255
    C_img = format_temperature_image(C_img, min_value=0, max_value=255)
    C_img = Image.fromarray(C_img)
    C_img.save("simulations/matrixes/sensor_placement.png")
    Theta_inv_img = format_temperature_image(Theta_inv, min_value=np.min(Theta_inv), max_value=np.max(Theta_inv))
    Theta_inv_img = Image.fromarray(Theta_inv_img)
   
    Theta_inv_img.resize((nx,ny), Image.Resampling.NEAREST)
    Theta_inv_img.save("simulations/matrixes/theta_inv.png")
    for i in range(num_sensors):
        Ur_img = format_temperature_image(Ur[:,i].reshape((nx,ny)), min_value=np.min(Ur), max_value=np.max(Ur))
        Ur_img = Image.fromarray(Ur_img)
        Ur_img.save("simulations/matrixes/pod_modes/pod_mode_" + str(i) + ".png")

    return

def POD_latent_space(indexes, num_pod_modes=5):
    data = load_simulations()
    nsims, nsteps, nx, ny = data.shape
    N = indexes.shape[0]
    U,_,_,_ = load_pod_modes()
    Ur = U[:,0:num_pod_modes]
    X = np.zeros((N, nx*ny))
    for i, (sim, step) in enumerate(indexes):
        X[i,:] = data[sim,step,:,:].reshape((1,nx*ny))
    X = X.T
    A = Ur.T @ X
    return A

def color_latent_space(A, n_colors=256):
    A_max = np.max(np.abs(A))
    A = (A + A_max)/(2*A_max)
    color_indexes = (A*n_colors).astype(int)
    colors = (matplotlib.cm.bwr(np.linspace(0, 1, n_colors))[:,0:3]*255).astype(np.uint8)
   
    A_colors = np.zeros((A.shape[0], A.shape[1], 3), dtype=np.uint8)
    A_colors[:,:] = colors[color_indexes]
    
    return A_colors


if __name__ == '__main__':
    #save_sensor_reconstruction_images()
    for i in range(5,6):
        save_simulation_images(sim_num=i)
    #save_images_changing_sensors()
    #save_images_pod_modes_sensors_theta_inv()
    pass
