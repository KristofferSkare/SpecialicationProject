import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = "temperature_sensorplacement_dmd_costa/data/temperature_simulations/"
data_file = "temperature_sensorplacement_dmd_costa/data/temperature_simulations.npy"


def initialize(seed):

    D = np.random.uniform(1, 100)

    Tcool = np.random.uniform(100, 500)
    Thot = np.random.uniform(Tcool*1.1, 1000)
  
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    u0 = Tcool * np.ones((nx, ny))
    ut = u0.copy()
    u = np.empty((nsteps+1, nx, ny))


    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r = np.random.uniform(1,np.min((h,w))/2)
    cx = w* (1/2 + np.random.uniform(-0.1, 0.1))
    cy = h* (1/2 + np.random.uniform(-0.1, 0.1))
    r2 = r**2
    for i in range(nx):
        for j in range(ny):
            p2 = (i*dx-cx)**2 + (j*dy-cy)**2
            if p2 < r2:
                u0[i,j] = Thot

    u[0] = u0.copy()
    return u0, ut, u, dt, D

def save_simulations_in_single_array(skip_beginning=None, skip_end=None, time_step=1):
   

    simulations = []

    for file in os.listdir(data_folder):
        if file.endswith(".npy"):
            sim_data = np.load(data_folder + file)
            if skip_beginning is not None and skip_end is not None:
                sim_data = sim_data[skip_beginning:skip_end:time_step]
            elif skip_beginning is not None:
                sim_data = sim_data[skip_beginning::time_step]
            elif skip_end is not None:
                sim_data = sim_data[:skip_end:time_step]
            elif time_step != 1:
                sim_data = sim_data[::time_step]
            simulations.append(sim_data)

    data = np.array(simulations)

    np.save(data_file, data)


def load_simulations():
    return np.load(data_file)

if __name__ == "__main__":
    ### Main settings for all simulations

    # Number of timesteps
    nsteps = 1500-1
    # plate size, mm
    w = h = 5.
    # intervals in x-, y- directions, mm
    dx = dy = 0.1

    nsimulations = 500

    nx, ny = int(w/dx), int(h/dy)

    dx2, dy2 = dx*dx, dy*dy


    for index in range(nsimulations):
        u0, ut, u, dt, D = initialize(index)

        def do_timestep(u0, u):
        # Propagate with forward-difference in time, central-difference in space
            u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
                (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
                + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

            u0 = u.copy()
            return u0, u

        for m in range(nsteps):
            u0, ut = do_timestep(u0, ut)
            u[m+1] = ut.copy()

        np.save(data_folder + str(index) +".npy", u)
    save_simulations_in_single_array(skip_beginning=50, time_step=5)


