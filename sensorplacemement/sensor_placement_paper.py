
# This is an implementation of the sensor placement algorithm described in the paper: https://ieeexplore.ieee.org/document/8361090

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.linalg

def find_basis(X, n, show=False):
    X_mean = np.mean(X, axis=1, keepdims=True) 
    X_ = X - np.tile(X_mean,(1,X.shape[1]))
    U, S, V = np.linalg.svd(X_)
    Psi_r = U[:,:n]
    
    if show:
        plt.figure()
        plt.imshow(X_mean.reshape(32,32).T, cmap="gray")
        plt.show()
        for i in range(10):
            plt.figure()
            plt.imshow(Psi_r[:,i].reshape(32,32).T, cmap="gray")
            plt.show()
    return Psi_r, X_mean

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

def find_sensor_placement_naive_constraints(Psi_r, num_eigen, num_sensors=None, constraints=None):
    if constraints is None:
        return find_sensor_placement(Psi_r, num_eigen, num_sensors)
    
    P = qr_pivots(Psi_r, num_eigen, num_sensors)
    C = np.zeros((num_sensors,Psi_r.shape[0]))

    # Naively take the p first pivots that satisfy the constraints
    # The real optimal solution with constraints is probably different 
    for con in constraints:
        P = P[np.where(con(P))]

    C[np.arange(num_sensors),P[:num_sensors]] = 1
    return C

def find_sensor_placement_naive_constraints_before(Psi_r, num_eigen, num_sensors=None, constraints=None):
    if constraints is None:
        return find_sensor_placement(Psi_r, num_eigen, num_sensors)
    
    # Remove the pixels outside the constraints before calculating the QR pivots
    # This is maybe a better solution than the naive constraints after QR pivots
    indexes = np.arange(Psi_r.shape[0])
    for con in constraints:
        indexes = indexes[np.where(con(indexes))]

    Psi_r_constrained = Psi_r[indexes,:]
    P = qr_pivots(Psi_r_constrained, num_eigen, num_sensors)
    C = np.zeros((num_sensors,Psi_r.shape[0]))

    C[np.arange(num_sensors),indexes[P[:num_sensors]]] = 1
    return C

def find_sensor_placement_matrixes(X, num_eigen, num_sensors=None, filename=None, constraints=None):
    if num_sensors is None:
        num_sensors = num_eigen
    Psi_r, X_mean = find_basis(X, num_eigen)
    C = find_sensor_placement_naive_constraints_before(Psi_r, num_eigen, num_sensors, constraints)
    Theta = C @ Psi_r
    Theta_inv = np.linalg.pinv(Theta)
    if filename is not None:
        scipy.io.savemat(filename, {'Psi_r': Psi_r, 'X_mean': X_mean, 'C': C, 'Theta': Theta, 'Theta_inv': Theta_inv})
    return C, Psi_r, X_mean, Theta, Theta_inv

def create_measurements_sensor_placement(X, C, X_mean):
    Y = np.dot(C, (X - np.tile(X_mean,(1,X.shape[1]))))
    return Y

def reconstruct_from_measurements(Y, C, Psi_r, X_mean,Theta_inv = None):
    if Theta_inv is None:
        Theta_inv = np.linalg.inv(np.dot(C, Psi_r))
    X_hat = np.dot(Psi_r, np.dot(Theta_inv, Y)) + np.tile(X_mean,(1,Y.shape[1]))
    return X_hat

def load_matrixes_from_file(filename):
    mat = scipy.io.loadmat(filename)
    Psi_r = mat['Psi_r']
    X_mean = mat['X_mean']
    C = mat['C']
    Theta = mat['Theta']
    Theta_inv = mat['Theta_inv']
    return C, Psi_r, X_mean, Theta, Theta_inv

if __name__ == "__main__":
    
    mat = scipy.io.loadmat('./data/Yale_32x32.mat')
    X = mat['fea'].T
    w, h = 32,32
    num_eigen = 100
    num_sensors = 100
    edge_percentage = 0.1
    center_percentage = 0.1
    
    cons = [
        lambda x: np.any([
            x> w*h*(1 - edge_percentage), 
            x < w*h*edge_percentage, 
            x%w > h*(1 - edge_percentage), 
            x % w < h*edge_percentage,
            (lambda x_: np.all([
                x_ > w*h*(0.5-center_percentage),
                x_ < w*h*(0.5+center_percentage),
                x_%w > h*(0.5-center_percentage),
                x_ % w < h*(0.5+center_percentage)
            ], axis=0))(x)
            ], axis=0),
    ]
    cons = None
    C, Psi_r, X_mean,  Theta, Theta_inv = find_sensor_placement_matrixes(X,num_eigen, num_sensors, filename='./data/Yale_32x32_sensor_placement.mat', constraints=cons)
    #C, Psi_r, X_mean, Theta, Theta_inv = load_matrixes_from_file(filename='./data/Yale_32x32_sensor_placement.mat')
    
    plt.imshow(np.sum(C, axis=0).reshape(32,32), cmap='gray')
    plt.show()
    

    Y = create_measurements_sensor_placement(X, C, X_mean)
    X_hat = reconstruct_from_measurements(Y, C, Psi_r, X_mean,Theta_inv)
    
    for i in np.random.choice(X.shape[1], 10):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(X[:,i].reshape(32,32).T, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(X_hat[:,i].reshape(32,32).T, cmap="gray")
        plt.show()




