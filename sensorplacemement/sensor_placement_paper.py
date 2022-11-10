
# This is an implementation of the sensor placement algorithm described in the paper: https://ieeexplore.ieee.org/document/8361090

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.linalg

import cvxpy as cvx

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


def theta_i_T_theta_i_decomposition(Psi_r):
    return np.array([Psi_r[[i],:].T @ Psi_r[[i],:] for i in range(Psi_r.shape[0])])


def find_sensor_placement_optimization(Psi_r, num_eigen, num_sensors=None, L1_regularization=None):
    if num_sensors is None:
        num_sensors = num_eigen

    n = Psi_r.shape[0]
    beta = cvx.Variable(n)
    decomposed_theta = theta_i_T_theta_i_decomposition(Psi_r)
    obj = cvx.log_det(cvx.sum([beta[i] * decomposed_theta[i] for i in range(n)], axis=0))
    if L1_regularization is not None:
        obj += -L1_regularization * cvx.norm(beta, 1)
    objective = cvx.Maximize(obj)
    constraints = [cvx.sum(beta) == num_sensors, beta >= 0, beta <= 1]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    print("Result:", result)
    
    indexes = np.argsort(beta.value)[::-1]
    

    C = np.zeros((num_sensors,Psi_r.shape[0]))
    C[np.arange(num_sensors),indexes[:num_sensors]] = 1
    return C, beta.value[indexes]

def compare_methods_from_file(data_filename, C_filename, num_eigen):

    mat = scipy.io.loadmat(data_filename)
    X = mat['fea'].T
    w, h = 32,32
    Psi_r, X_mean = find_basis(X, num_eigen)

    save_obj = np.load(C_filename, allow_pickle=True).item()
    C_opt = save_obj['C_opt']
    C_QR = save_obj['C_QR']

    print(np.sum(C_opt, axis=1))
    print(np.sum(C_QR, axis=1))
    print(np.sum(np.abs(np.sum(C_opt, axis=0) - np.sum(C_QR, axis=0))))
    Y_opt = create_measurements_sensor_placement(X, C_opt, X_mean)
    Y_QR = create_measurements_sensor_placement(X, C_QR, X_mean)
    X_hat_opt = reconstruct_from_measurements(Y_opt, C_opt, Psi_r, X_mean)
    X_hat_QR = reconstruct_from_measurements(Y_QR, C_QR, Psi_r, X_mean)

    for i in range(10):
        plt.subplot(1,3,1)
        plt.imshow(X[:,i].reshape(w,h).T, cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(X_hat_opt[:,i].reshape(w,h).T, cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(X_hat_QR[:,i].reshape(w,h).T, cmap='gray')
        plt.show()

if __name__ == "__main__":

    compare_methods_from_file('./data/Yale_32x32.mat', "optimization_50_L1.npy", 50)
    
    mat = scipy.io.loadmat('./data/Yale_32x32.mat')
    X = mat['fea'].T
    w, h = 32,32
    num_eigen = 50
    num_sensors = 50
    Psi_r, X_mean = find_basis(X, num_eigen)
    
    C_opt, betas = find_sensor_placement_optimization(Psi_r, num_eigen, num_sensors, L1_regularization=1)
    
    C_QR = find_sensor_placement(Psi_r, num_eigen, num_sensors)
    save_obj = {'C_opt': C_opt, 'C_QR': C_QR, 'betas': betas}
    np.save("optimization_50_L1.npy", save_obj)
    
    plt.figure(1)
    plt.plot(betas[:num_sensors*2])
    perfect_beats = np.zeros((num_sensors*2))
    perfect_beats[:num_sensors] = 1
    plt.plot(perfect_beats)
    plt.show()

    
    '''
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


'''

