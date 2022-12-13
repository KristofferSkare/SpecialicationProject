import numpy as np
import tensorflow as tf
from sensor_placement import load_sensorplacement
from dmd import load_dmd_modes
from pod_analysis import load_POD, load_limited_data
from temperature_simulation import load_simulations
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import json


class SensorPlacementPredictionGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, C, Ur, num_prediction_steps=1, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle
        

        nsims, nsteps, nvars = X.shape
        self.n_simulations = nsims
        self.n_steps = nsteps
        self.n_vars = nvars
        self.n_pred_steps = num_prediction_steps

        self.X = X
        self.C = C
        self.Ur = Ur

        self.Y = (C @ X.reshape(nsims*nsteps, nvars).T).T.reshape(nsims, nsteps, C.shape[0])
        self.a = (Ur @ X.reshape(nsims*nsteps, nvars).T).T.reshape(nsims, nsteps, C.shape[0])

        self.indexes = np.arange(nsims*(nsteps-num_prediction_steps))

        self.len = self.indexes.size

        self.on_epoch_end()

    
    def __len__(self):
        return self.len//self.batch_size
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        sim_nums = np.floor_divide(indexes, self.n_steps-self.n_pred_steps)
        step_nums = np.mod(indexes, self.n_steps-self.n_pred_steps)

        
        pred_steps = np.arange(self.n_pred_steps+1) + step_nums[:, np.newaxis]
   
        X_batch = self.X[sim_nums[:, np.newaxis], pred_steps, :]
        a_batch = self.a[sim_nums[:, np.newaxis], pred_steps, :]
        Y_batch = self.Y[sim_nums, step_nums, :]

        return Y_batch, [X_batch, a_batch]
    
    def on_epoch_end(self):
        if (self.shuffle):
            np.random.shuffle(self.indexes)



def create_model(
    Theta_inv, A_tilde, U,
    nsteps = 10,
    prediction_layers = [16,32,64],
    sensor_layers = [16,32,64],
    reconstruction_layers = [512,256,512,2500],
    l1_weight = 0.01,
    dt = 0.1,
    zero_init = False,
    loss_weights = {"X": 1, "a": 0.0001},
    ):
    initilizer = [tf.keras.initializers.GlorotNormal() for i in range(3)]
    if isinstance(zero_init, bool):
        if zero_init:
            initilizer = [tf.keras.initializers.Zeros() for i in range(3)]
    else:
        for i in range(3):
            if zero_init[i]:
                initilizer[i] = tf.keras.initializers.Zeros()
    
    regularizer = tf.keras.regularizers.L1(l1_weight)

    n,r = U.shape

    # Defining layers

    Theta_inv_mul = tf.keras.layers.Dense(r, use_bias=False, name="Theta_inv", kernel_initializer=tf.keras.initializers.Constant(Theta_inv.T), trainable=False)
    
    dmd_pred_dot = A_tilde
    A_mul = tf.keras.layers.Dense(r, use_bias=False, name="A",kernel_initializer=tf.keras.initializers.Constant(dmd_pred_dot.T), trainable=False)

    U_mul = tf.keras.layers.Dense(n, use_bias=False, name="U", kernel_initializer=tf.keras.initializers.Constant(U.T), trainable=False)

    add = tf.keras.layers.Add()
    # Model for correcting reconstruction from sparse sensors to latent space
    sensor_source_term = tf.keras.Sequential(
    [tf.keras.layers.Dense(layer, activation=None if i == len(sensor_layers) else "relu", kernel_regularizer=regularizer, kernel_initializer=initilizer[0]) for i, layer in enumerate([*sensor_layers, r])]
        , name="sensor_correction")

    # Model for correcting future prediction in latent space
    prediction_source_term = tf.keras.Sequential(
        [tf.keras.layers.Dense(layer, activation=None if i == len(prediction_layers) else "relu", kernel_regularizer=regularizer, kernel_initializer=initilizer[1]) for i, layer in enumerate([*prediction_layers, r])]
        ,name="prediction_correction")
    
    # Model for correcting reconstructing from latent space to full space
    reconstruction_source_term = tf.keras.Sequential(
        [tf.keras.layers.Dense(layer, activation=None if i == len(reconstruction_layers) else "relu", kernel_regularizer=regularizer, kernel_initializer=initilizer[2]) for i, layer in enumerate([*reconstruction_layers, n])]
        ,name="reconstruction_correction")

    # Create model
    Y = tf.keras.layers.Input(shape=(r), name="Y")
    a_hat_0 = Theta_inv_mul(Y)


    sigma_s = sensor_source_term(a_hat_0)

    a_tilde_i = add([a_hat_0,sigma_s])
    a_tilde = [a_tilde_i]

    X_hat_i = U_mul(a_tilde_i)

    sigma_r = reconstruction_source_term(X_hat_i)

    X_tilde_i = add([X_hat_i ,sigma_r])
    X_tilde= [X_tilde_i]

    for i in range(1, nsteps+1):

        a_hat_i = A_mul(a_tilde_i)
        sigma_p = prediction_source_term(a_hat_i)
        a_tilde_i = add([dt*sigma_p, a_hat_i])

        X_hat_i = U_mul(a_tilde_i)
        sigma_r = reconstruction_source_term(X_hat_i)
        X_tilde_i = add([X_hat_i,sigma_r])

        a_tilde.append(a_tilde_i)

        X_tilde.append(X_tilde_i)

    reshape_1n = tf.keras.layers.Reshape((1, n))
    reshape_1r = tf.keras.layers.Reshape((1, r))
    X_tilde_all = tf.keras.layers.Concatenate(axis=1, name="X")([reshape_1n(X) for X in X_tilde])
    a_tilde_all = tf.keras.layers.Concatenate(axis=1, name="a")([reshape_1r(a) for a in a_tilde])

    model = tf.keras.Model(inputs=Y, outputs=[X_tilde_all, a_tilde_all])
    # Values i a is generally 10* values in X, and loss in a is less important. scale A mse with (0.1)**2 * 0.01
    loss = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights)

    model.get_layer("Theta_inv").trainable = False
    model.get_layer("A").trainable = False
    model.get_layer("U").trainable = False

    model.summary()

    return model

def train_model(num_pred_steps=None, num_modes_used=None, epochs=1, model_folder=None):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Load data

    
    config_file = model_folder + "/config.json"
    with open(config_file) as json_file:
        config = json.load(json_file)

    
    if num_pred_steps is None:
        num_pred_steps = config["num_pred_steps"]
    
    if num_modes_used is None:
        num_modes_used = config["num_modes_used"]
    
    data = load_simulations(config["data_file"])
    U, L, RIC, X_mean = load_POD(config["pod_file"])
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement(config["sensor_placement_file"])
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(config["dmd_file"])
    
    X = data.reshape(data.shape[0], data.shape[1], np.product(data.shape[2:]))
    X = X - X_mean
    #data = load_limited_data(max_simulations=20, max_time_steps=100)
    model, latest_cp = load_model(model_folder, num_pred_steps=num_pred_steps, num_modes_used=num_modes_used, config=config)

    # Train val split
    val_size = 0.2
    n = X.shape[0]
    sims = np.arange(0, n)
    np.random.shuffle(sims)
    val_cutof = int(val_size*n)
    val_sims = sims[:int(n*val_size)]
    train_sims = sims[val_cutof:]

    train_generator = SensorPlacementPredictionGenerator(X[train_sims], C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=config["batch_size"], shuffle=True)
    val_generator = SensorPlacementPredictionGenerator(X[val_sims], C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=config["batch_size"], shuffle=True)
    
    cb = tf.keras.callbacks.ModelCheckpoint(model_folder + "/cp-{epoch:02d}.ckpt", monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
    
    initial_epoch = 0
    if latest_cp is not None:
        initial_epoch = int(latest_cp.split("cp-")[1].split(".")[0])

    model.fit(train_generator, epochs=epochs, verbose=1, callbacks=[cb],validation_data=val_generator, initial_epoch=initial_epoch)
    

def load_model(model_folder, num_pred_steps=2, num_modes_used=8, zero_init=False, config=None):
    if config is None:
        config_file = model_folder + "config.json"
        with open(config_file) as json_file:
            config = json.load(json_file)
        
    
    U, L, RIC, X_mean = load_POD(config["pod_file"])
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement(config["sensor_placement_file"])
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(config["dmd_file"])
    
    model = create_model(Theta_inv, Atilde, Ur, nsteps=num_pred_steps, 
    sensor_layers=config["sensor_layers"],
    prediction_layers=config["prediction_layers"],
    reconstruction_layers=config["reconstruction_layers"],
    l1_weight=config["l1_weight"],
    dt = config["dt"],
    zero_init=zero_init,
    loss_weights=config["loss_weights"],
    )
    latest_checkpoint = None
    latest_checkpoint = tf.train.latest_checkpoint(model_folder)
    print("Latest checkpoint", latest_checkpoint)
    if isinstance(zero_init, bool):
        if zero_init == False and (latest_checkpoint is not None):
            model.load_weights(latest_checkpoint)
    elif (latest_checkpoint is not None):
            model.load_weights(latest_checkpoint)
            layer_names = ["sensor_correction", "prediction_correction", "reconstruction_correction"]
            for i,zero in enumerate(zero_init):
                layer_name = layer_names[i]
                if zero:
                    layer = model.get_layer(layer_name)
                    weights = layer.get_weights()
                    layer.set_weights([np.zeros_like(w) for w in weights])
                    print(layer.name, "weights set to zero", np.sum(np.linalg.norm(w) for w in layer.get_weights()))
    

    return model, latest_checkpoint

def show_predictions(num_pred_steps=50, num_modes_used=8, simulation=0, zero_init=False, interval_pred_steps=1, model_folder=None):
    model, latest_cp = load_model(model_folder, num_pred_steps=num_pred_steps, num_modes_used=num_modes_used, zero_init=zero_init)

    data = load_simulations()
    #data = load_limited_data(max_simulations=10, max_time_steps=100)
    U, L, RIC, X_mean = load_POD()
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement()
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes()

    n_sims, n_steps, nx, ny = data.shape 
    X = data.reshape((n_sims, n_steps, np.product(data.shape[2:])))
    X = X - X_mean
    
    generator = SensorPlacementPredictionGenerator(X, C, Ur.T, num_prediction_steps=num_pred_steps, batch_size=1, shuffle=False)
    len_of_series = n_steps-num_pred_steps
    predictions = np.zeros((num_pred_steps+1,n_steps-num_pred_steps,nx,ny))
    for i in range(len_of_series):
        y, x = generator[simulation*len_of_series + i]
        X_tilde, a_tilde = model.predict(y)
        pred = X_tilde + X_mean 
        pred = pred.reshape((num_pred_steps+1, nx, ny))
        predictions[:,i] = pred

    reconstruction_movie(data[simulation], predictions, prediction_interval=interval_pred_steps, dt=0.01)

    
def reconstruction_movie(X, X_rec,prediction_interval,dt=0.01):
    n_steps, nx, ny = X.shape
    n_pred_steps = X_rec.shape[0]-1
    
    num_cols =(n_pred_steps + 1)//prediction_interval
    num_rows = 3
    fig, ax = plt.subplots(num_rows,num_cols)
    
    min = np.min(X)
    max = np.max(X)
    error = np.zeros((n_steps,n_pred_steps+1, nx, ny))
    for i in range(n_steps-n_pred_steps-1):
        for j in range(n_pred_steps+1):
            error[i,j] = X[i+j] - X_rec[j,i]

    error_max = np.max(np.abs(error))

    sim_ims = [ax[0,i].imshow(X[i*prediction_interval], vmin=min, vmax=max, cmap="hot") for i in range(num_cols)]
    pred_ims = [ax[1,i].imshow(X_rec[i*prediction_interval,0], vmin=min, vmax=max, cmap="hot") for i in range(num_cols)]
    error_ims = [ax[2,i].imshow(X_rec[i*prediction_interval,0]-X[0], vmin=0, vmax=error_max,cmap="gray") for i in range(num_cols)]


    def updatefig(j):
        for i in range(num_cols):
            sim_ims[i].set_array(X[i*prediction_interval+j])
            pred_ims[i].set_array(X_rec[i*prediction_interval,j])
            error_ims[i].set_array(X_rec[i*prediction_interval,j]-X[i*prediction_interval+j])

        return *sim_ims, *pred_ims, *error_ims

    ani = animation.FuncAnimation(fig, updatefig, frames=range(X.shape[0] - n_pred_steps-1), interval=dt*1000, blit=True)

    plt.show()

def draw_thresholded_graph(w_s, biases, threshold=0):
  
    all_weights = np.concatenate([w.flatten() for w in [*w_s, *biases]])
    log10_weights = np.log10(np.abs(all_weights))
    log10_weights[log10_weights < -10] = -10
    plt.figure()
    
    hist_vals, bins, _ = plt.hist(log10_weights, bins=200)
    min_bin, max_bin = int(np.floor(np.min(bins))), int(np.ceil(np.max(bins)))
    max_value = np.max(hist_vals)
    plt.plot([np.log10(threshold)]*2,[0, max_value], "r")
    #plt.title("Histogram of weights and biases")
    plt.xlabel("Absolute value of weights and biases")
    plt.ylabel("Number of parameters")
    plt.xticks([*range(min_bin+1, max_bin+1, 2), np.log10(threshold)], [*[str(10**(i)) for i in range(min_bin+1, max_bin+1,2)], "threshold"])
    plt.show()
    num_out = w_s[-1].shape[1]
    num_layers = len(w_s) + 1 
    edges_per_layer = np.array([w.shape[0]*w.shape[1] for w in w_s])
    edges_per_layer_single_out = edges_per_layer.copy()
    edges_per_layer_single_out[-1] = edges_per_layer_single_out[-1]//num_out
    max_num_edges = np.sum(edges_per_layer_single_out)
    max_num_edges_total = np.sum(edges_per_layer)
    max_num_nodes = np.sum([w.shape[0] for w in w_s]) + 1
    max_num_nodes_total = max_num_nodes + num_out - 1
    print("Max number of nodes: {:}".format(max_num_nodes_total))
    print("Max number of edges: {:}".format(max_num_edges_total))

    total_graph = nx.DiGraph()
    graphs = [nx.DiGraph() for i in range(num_out)]
    poses = [{} for n in range(num_out+1)]
    print("Adding every node to the graph")
    for i in reversed(range(len(w_s))):
        J, K = w_s[i].shape
        for j in range(J):
            for n in range(num_out):
                graphs[n].add_node("{:}_{:}".format(i,j))
            total_graph.add_node("{:}_{:}".format(i,j))
        if i == len(w_s)-1:
            for n in range(num_out):
                graphs[n].add_node("{:}_{:}".format(i+1,n))
                total_graph.add_node("{:}_{:}".format(i+1,n))
    
    print("Adding edges to the graph")
    num_edges =0
    for i in reversed(range(len(w_s))):
        J, K = w_s[i].shape
        for j in range(J):
            for k in range(K):
                weight = w_s[i][j,k]
                if np.abs(weight) >= threshold:
                    num_edges += 1
                    for n in range(num_out):
                        graphs[n].add_edge("{:}_{:}".format(i,j), "{:}_{:}".format(i+1,k), weight=weight)
                    total_graph.add_edge("{:}_{:}".format(i,j), "{:}_{:}".format(i+1,k), weight=weight)
    print("Added {:} edges".format(num_edges))
    
    for n in range(num_out):
        for j in range(num_out):
            if j!= n:
                node = "{:}_{:}".format(num_layers-1,j)
                if (graphs[n].has_node(node)):
                    graphs[n].remove_node(node)

    print("Removing nodes with no incoming or outgoing edges")
    for g in [*graphs, total_graph]:
        contin = True
        while contin:
            contin = False
            nodes = [*g.nodes()]
            for node in nodes:
                in_degree = g.in_degree(node)
                out_degree = g.out_degree(node)
                layer = int(node.split("_")[0])
                number = int(node.split("_")[1])
                removed = False
                if in_degree == 0 and layer != 0:
                    bias = biases[layer-1][number]
                    if np.abs(bias) < threshold:
                        g.remove_node(node)
                        removed = True
                        contin = True
                if out_degree == 0 and layer != len(w_s) and not removed:
                    g.remove_node(node)
                    contin = True
                
    # output_biases = biases[-1]
    # for i in range(num_out):
    #     if np.abs(output_biases[i]) >= threshold:
    #         total_graph.add_node("{:}_{:}".format(len(w_s),i))

    print("Sorting nodes by layer and number")

    for n, g in enumerate([*graphs, total_graph]):
        nodes = g.nodes()
        layered_nodes = {}
        for node in nodes:
            layer = int(node.split("_")[0])
            number = int(node.split("_")[1])
            if layer not in layered_nodes:
                layered_nodes[layer] = []
            layered_nodes[layer].append(number)
        
        for layer in layered_nodes:
            sorted_nodes = np.sort(layered_nodes[layer])
            for i, node in enumerate(sorted_nodes):
                poses[n]["{:}_{:}".format(layer,node)] = (layer,i)

    total_nodes = total_graph.nodes()
    parameters_per_layer = [[0,0] for i in range(num_layers)]
    for node in total_nodes:
        layer = int(node.split("_")[0])
        parameters_per_layer[layer][0] += 1
        in_degree = total_graph.in_degree(node)
        parameters_per_layer[layer][1] += in_degree

    print("Drawing graphs")
    for n in range(len(graphs)):
        nodes = graphs[n].nodes()
        graph_label_dict = {}
        for node in nodes:
            number = int(node.split("_")[1])
            graph_label_dict[node] = "{:}".format(number)
        num_nodes = len(nodes)
        num_edges = len(graphs[n].edges())
        print("Number of nodes: {:}, ({:}%)".format(num_nodes, num_nodes/max_num_nodes*100))
        print("Number of edges: {:}, ({:.2f}%)".format(num_edges, num_edges/max_num_edges*100))
        if (num_nodes == 1 or num_edges == 0):
            continue

        plt.figure()

        nx.draw(graphs[n],pos=poses[n], with_labels=True, labels=graph_label_dict )
    
    nodes = total_graph.nodes()
    graph_label_dict = {}
    for node in nodes:
        number = int(node.split("_")[1])
        graph_label_dict[node] = "{:}".format(number)
    
    num_nodes = len(nodes)
    num_edges = len(total_graph.edges())
    print("\nTotal graph:")
    print("Number of nodes: {:}, ({:}%)".format(num_nodes, num_nodes/(max_num_nodes_total)*100))
    print("Number of edges: {:}, ({:.2f}%)".format(num_edges, num_edges/(max_num_edges_total)*100))

    print("Parameters per layer:")
    for i in range(num_layers):
        print("Layer {:}: {:} nodes, {:} edges".format(i, parameters_per_layer[i][0], parameters_per_layer[i][1]))
    plt.figure()
  
    nx.draw(total_graph,pos=poses[num_out], with_labels=True, labels=graph_label_dict )

    plt.show()


def analyze_weights(model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/", thresholds = [10**(-4), 10**(-3), 10**(-4)]):
    model,_ = load_model(model_folder, num_modes_used=8,num_pred_steps=1, zero_init=False)
    layer_names = ["sensor_correction", "prediction_correction", "reconstruction_correction"]

    layers = [model.get_layer(name).get_weights() for name in layer_names]
    weights = [w[::2] for w in layers]
    biases = [w[1::2] for w in layers]
    for i in range(len(layers)):
        draw_thresholded_graph(weights[i],biases[i], threshold=thresholds[i])
    
        
        
def compare_models(sim=0, time_step=0, prediction_steps=1, model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_2/", num_show_plot=6, weight_thresholds=None):
    
    config = {}
    config_file = os.path.join(model_folder, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    C, Theta = load_sensorplacement(file=config["sensor_placement_file"])
    X = load_simulations(file=config["data_file"])
    n_sims, n_steps, nx,ny = X.shape
    config["num_pred_steps"] = prediction_steps
    X_truth = X[sim, time_step:time_step+prediction_steps+1]
    x0 = X[sim, time_step].reshape(1,-1)
    y0 = x0 @ C.T
    model,_ = load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=False)
    
    for n, name in enumerate(["sensor_correction", "prediction_correction", "reconstruction_correction"]):
        layer = model.get_layer(name)
        print(name, np.sum([np.linalg.norm(w) for w in layer.get_weights()]))
        if weight_thresholds is not None:
            weights = layer.get_weights()
            print("Before: ", np.sum([np.linalg.norm(w, ord=1) for w in weights]))
            for i in range(len(weights)):
                weights[i][np.abs(weights[i]) < weight_thresholds[n]] = 0
            print("After: ", np.sum([np.linalg.norm(w, ord=1) for w in weights]))
            layer.set_weights(weights)


  
    model_zero,_ = load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=True)
   
    X_pred, a_pred = model(y0.copy())
    X_pred_zero, a_pred_zero = model_zero(y0.copy())

    X_pred = X_pred.numpy().reshape((prediction_steps +1, nx,ny))
    X_pred_zero = X_pred_zero.numpy().reshape((prediction_steps +1, nx,ny))
    
    error = np.abs(X_truth - X_pred)
    error_zero = np.abs(X_truth - X_pred_zero)
    
    error_t = np.mean(error, axis=(1,2))
    error_zero_t = np.mean(error_zero, axis=(1,2))
    plt.figure()
    plt.plot(error_t)
    plt.plot(error_zero_t)
    plt.legend(["error", "error_zero"])

    fig, ax = plt.subplots(5, num_show_plot)
    indexes = np.linspace(0, prediction_steps, num_show_plot, dtype=int)
    X_truth_show = X_truth[indexes]
    X_pred_show = X_pred[indexes]
    X_pred_zero_show = X_pred_zero[indexes]
    error_show = error[indexes]
    error_zero_show = error_zero[indexes]


    vmax = np.max([np.max(X_truth_show), np.max(X_pred_show), np.max(X_pred_zero_show)])
    vmin = np.min([np.min(X_truth_show), np.min(X_pred_show), np.min(X_pred_zero_show)])

    max_error = np.max([np.max(error_show), np.max(error_zero_show)])
    for i, index in enumerate(indexes):
        ax[0,i].imshow(X_truth_show[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[1,i].imshow(X_pred_show[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[2,i].imshow(X_pred_zero_show[i], cmap="hot", vmin=vmin, vmax=vmax)
        ax[3,i].imshow(error_show[i], cmap="gray", vmin=0, vmax=max_error)
        ax[4,i].imshow(error_zero_show[i], cmap="gray", vmin=0, vmax=max_error)
        for j in range(5):
            ax[j,i].set_xticks([])
            ax[j,i].set_yticks([])
    plt.show()


    pass

def evaluate_model(model_folder, num_modes_used=8, original=True, zero_init=False, weight_thresholding=False, weight_thresholds=None, prediction_steps=1):  
    models = []
    names = []
    if original:
        model,_ = load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=False)
        models.append(model)
        names.append("Full model")
    if zero_init:
        model_zero,_ = load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=True)
        models.append(model_zero)
        names.append("Zero init")
    
    if weight_thresholding and weight_thresholds is not None:
        names.append("Thresholded")
        model_threshold = load_thresholded_model(model_folder, num_modes_used=8,prediction_steps=prediction_steps, weight_thresholds=weight_thresholds)
        models.append(model_threshold)
    
    config = {}
    config_file = os.path.join(model_folder, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    
    data = load_simulations(config["data_file"])
    U, L, RIC, X_mean = load_POD(config["pod_file"])
    Ur = U[:,:num_modes_used]
    C, Theta_inv = load_sensorplacement(config["sensor_placement_file"])
    Atilde, D, W, W_inv, dmd_modes = load_dmd_modes(config["dmd_file"])
    
    X = data.reshape(data.shape[0], data.shape[1], np.product(data.shape[2:]))
    X = X - X_mean

    generator = SensorPlacementPredictionGenerator(X, C, Ur.T, num_prediction_steps=prediction_steps, batch_size=config["batch_size"], shuffle=True)

    for i, model in enumerate(models):
        print("\n\nEvaluating model: ", names[i])
        results = model.evaluate(generator, verbose=1)
        print("\n\n")

def plot_prediction_error_graph():

    data = load_simulations()
    nsims, ntime, nx, ny = data.shape
    C, Theta_inv = load_sensorplacement()
    num_sims = nsims
    X0 = data[:num_sims,0].reshape(num_sims, nx*ny)
    Y0 =  X0 @ C.T

    
    prediction_steps = ntime - 1 
    model_folder_l1 = "temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/"
    model_folder = "temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_3/"
    zero,_= load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=True)
    model_l1, _ = load_model(model_folder_l1, num_modes_used=8, num_pred_steps=prediction_steps, zero_init=False)
    model_l1_threshold = load_thresholded_model(model_folder_l1, num_modes_used=8, prediction_steps=prediction_steps, weight_thresholds=[10, 10**(-2.8), 10])
    model,_= load_model(model_folder, num_modes_used=8,num_pred_steps=prediction_steps, zero_init=False)

    
    
    models = [model, model_l1, model_l1_threshold, zero]
    names = ["CoSTA", "L1 CoSTA", "L1 threshold CoSTA", "No CoSTA"]
    styles = [["blue", "solid"], ["orange", "solid"], ["green", (0,(9,10))], ["red", "solid"]]

    preds = []
    for model in models:
        pred, _ = model(Y0.copy())
        pred = pred.numpy()
        pred = pred.reshape(num_sims, prediction_steps +1, nx, ny)
        preds.append(pred)
    

  

    errors = [np.abs(data[:num_sims,:prediction_steps+1] - pred) for pred in preds]
    mean_error = [np.mean(e) for e in errors]
    print(mean_error)
    error_per_pixel = [np.mean(e, axis=(0,1)) for e in errors]
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    cm = plt.cm.get_cmap('gray')
    
    for i, a in enumerate(ax):
        vmin = np.min(error_per_pixel[i])
        vmax = np.max(error_per_pixel[i])
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        a.imshow(error_per_pixel[i], cmap="gray", vmin=vmin, vmax=vmax)
        fig.colorbar(sm, ax=a)
        a.set_title(names[i])
        a.set_xticks([])
        a.set_yticks([])
    #plt.suptitle("MAE per spacial postition")
        
    error_over_pixels = [np.mean(e, axis=(2,3)) for e in errors]
    error_t = [np.mean(e, axis=(0)) for e in error_over_pixels]
    stds = [np.std(e, axis=(0)) for e in error_over_pixels]
    plt.figure()
    for i, e in enumerate(error_t):
        plt.plot(e, color=styles[i][0], linestyle=styles[i][1])
    plt.legend(names)

    for i, e in enumerate(error_t):
        std = stds[i]
        plt.fill_between(np.arange(e.shape[0]) ,e-std, e+std, alpha=0.1, color=styles[i][0], linestyle=styles[i][1])
    #plt.title("Absolute error per prediction step")
    plt.xlabel("Num steps predicted")
    plt.ylabel("MAE")
    plt.figure()
    for i, e in enumerate(error_t[:-1]):
        plt.plot(e, color=styles[i][0], linestyle=styles[i][1])
    plt.legend(names[:-1])
    
    for i, e in enumerate(error_t[:-1]):
        std = stds[i]
        plt.fill_between(np.arange(e.shape[0]) ,e-std, e+std, alpha=0.1, color=styles[i][0], linestyle=styles[i][1])
    #plt.title("Absolute error per prediction step")
    plt.xlabel("Num steps predicted")
    plt.ylabel("MAE")
    plt.show()

def load_thresholded_model(model_folder, weight_thresholds, prediction_steps=1, num_modes_used=8):
    model_threshold, _ = load_model(model_folder, num_modes_used=num_modes_used, num_pred_steps=prediction_steps, zero_init=False)
    for n, name in enumerate(["sensor_correction", "prediction_correction", "reconstruction_correction"]):
            total = 0
            layer = model_threshold.get_layer(name)
            print("Layer: ", layer.name)
           
            weights = layer.get_weights()
            before = []
            after = [] 
            for i in range(len(weights)):
                size = weights[i].size
                before.append(size)
                indexes = np.abs(weights[i]) < weight_thresholds[n]
                num_zero = np.sum(indexes)
                num_non_zero = size - num_zero
                after.append(num_non_zero)
                total += num_non_zero
                weights[i][indexes] = 0
                layer.set_weights(weights)
            print("Before: ", before)
            print("After: ", after)
            print("Total non-zero weights: ", total)
    return model_threshold

def show_prediction(sim=0, t=0, steps=289, num_steps_shown=6):
    model_folder1 = "temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/"
    model_folder2 = "temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_3/"
    model_threshold = load_thresholded_model(model_folder1, weight_thresholds=[10, 10**(-2.8), 10], prediction_steps=steps, num_modes_used=8)
    model,_ = load_model(model_folder2, num_modes_used=8, num_pred_steps=steps, zero_init=False)
    zero,_ = load_model(model_folder1, num_modes_used=8, num_pred_steps=steps, zero_init=True)

    data = load_simulations()
    nsims, nt, nx, ny = data.shape
    X = data[sim, t:t+steps+1]
    X0 = X[0].reshape(1, nx*ny)
    C, Theta_inv = load_sensorplacement()
    Y0 = X0 @ C.T
    models = [model, model_threshold, zero]
    preds = []
    for m in models:
        pred,_ = m(Y0.copy())
        pred = pred.numpy().reshape(steps+1, nx, ny)
        preds.append(pred)
    errors = [np.abs(X - p)for p in preds] 
    model_names = ["CoSTA", "CoSTA thresholded", "No CoSTA"]
    indexes = np.linspace(0, steps, num_steps_shown, dtype=int)
    X = X[indexes]
    preds = [p[indexes] for p in preds]
    errors = [e[indexes] for e in errors]

    for i in range(len(models)):
        max_error = np.max(errors[i]) 
        max_val = np.max([np.max(X), np.max(preds[i])])
        min_val = np.min([np.min(X), np.min(preds[i])])
        fig, ax = plt.subplots(3,num_steps_shown)
        for j in range(num_steps_shown):
            ax[0,j].imshow(X[j], cmap="hot", vmin=min_val, vmax=max_val)
            ax[1,j].imshow(preds[i][j], cmap="hot", vmin=min_val, vmax=max_val)
            ax[2,j].imshow(errors[i][j], cmap="gray", vmin=0, vmax=max_error)

            if j == 0:
                ax[0,j].set_title("Original: t=" + str(t+j))
                ax[1,j].set_title(model_names[i] + ": t=" + str(t+indexes[j]))
                ax[2,j].set_title("Error: t=" + str(t+j))
            else:
                for n in range(3):
                    ax[n,j].set_title("t=" + str(t+indexes[j]))

            for n in range(3):
                ax[n,j].set_xticks([])
                ax[n,j].set_yticks([])
        cm_hot = plt.cm.get_cmap('hot')
        sm_hot = plt.cm.ScalarMappable(cmap=cm_hot, norm=plt.Normalize(vmin=min_val, vmax=max_val))
        cm = plt.cm.get_cmap('gray')
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_error))
        plt.colorbar(sm_hot, format="%4.1e", ax=ax[[0,1],:])
        plt.colorbar(sm, format="%4.1e", ax=ax[2,:])
        #plt.suptitle(model_names[i] + " prediction")
    plt.show()

    
    pass 


if __name__ == "__main__":
    #show_prediction(sim=2, t=0, steps=289, num_steps_shown=6)
    #plot_prediction_error_graph()
    #weight_thresholds=[10**(-3),10**(-2.8),10**(-3)]
    evaluate_model(model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/", 
    original=False, zero_init=False,
    weight_thresholding=True, weight_thresholds=[10**(-3),10**(-2.8),10**(-3)], prediction_steps=100)

    #compare_models(sim=0, time_step=0, prediction_steps=50, 
    #model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/", 
    #weight_thresholds=weight_thresholds
    #)
    
    #analyze_weights(model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_1/",
    #thresholds=[10**(-3), 10**(-2.8),10**(-3)])
    
    #train_model(epochs=30, model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_3/")
    #show_predictions(model_folder="temperature_sensorplacement_dmd_costa/costa_models/modes_used_8/model_2/", num_pred_steps=59, num_modes_used=8, zero_init=False, simulation=0, interval_pred_steps=10)
    

    #Y,X,a= format_data(data, Ur, C, X_mean, batch_size=32, prediction_steps=50)
    #np.save("costa_data.npy",{ "Y":Y, "X":X, "a":a})

    #
    pass

    





