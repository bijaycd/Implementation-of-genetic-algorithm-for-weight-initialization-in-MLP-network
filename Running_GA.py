import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
from skimage import io
from sklearn.decomposition import PCA
from tqdm import tqdm
import csv

# # Define the global variables
# image1_data = None
# image2_data = None
# solution_per_population = None
# num_parents_mating = None
# num_generations = None
# conv_threshold = None
# mutation_percent = None
# crossover_prob = None
# hidden_layers = None
# hidden_nodes = []
# low_val = None
# high_val = None

# def user_inputs(csv_data, image1_data, image2_data, solution_per_population, num_parents_mating,num_generations,conv_threshold, 
# mutation_percent,crossover_prob,hidden_layers,hidden_nodes,low_val,high_val):
#     csv_data = csv_data
#     image1_data = image1_data
#     image2_data = image2_data
#     solution_per_population = solution_per_population
#     num_parents_mating = num_parents_mating
#     num_generations = num_generations
#     conv_threshold = conv_threshold
#     mutation_percent = mutation_percent
#     crossover_prob = crossover_prob
#     hidden_layers = hidden_layers
#     hidden_nodes = hidden_nodes
#     low_val = low_val
#     high_val = high_val
    

def user_inputs(image1_data, image2_data, solution_per_population, num_parents_mating,num_generations,conv_threshold, 
mutation_percent,crossover_prob,hidden_layers,hidden_nodes,low_val,high_val):
    
    # IMAGE Data
    data = image1_data
    label = image2_data

    data = data[sorted(data.keys())[-1]]
    label = label[sorted(label.keys())[-1]]
    label_no = np.unique(label)
    d = dict(enumerate(label_no.flatten(),0))
    def return_key(val):
        for key, value in d.items():
            if key==val:
                return value
        return('Key Not Found')

    # x_train = np.array(image1)
    # y_train = np.array(image2)

    ########################################################################################################################
    # Choose three bands for RGB display
    band1 = 29  # Choose band 29
    band2 = 19  # Choose band 19
    band3 = 9  # Choose band 9

    # Extract the selected bands from the image data
    image_rgb = np.stack((data[:, :, band1], data[:, :, band2], data[:, :, band3]), axis=-1)

    # # Normalize the RGB image data to [0, 255]
    image_rgb = (image_rgb - np.min(image_rgb)) / (np.max(image_rgb) - np.min(image_rgb)) * 255
    image_rgb = image_rgb.astype(np.uint8)

    # Display the RGB image
    plt.imshow(image_rgb)
    plt.title('RGB Image')
    plt.show()

###########################################################################################################################


    X = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    data1 = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    print('Original data shape:', data1.shape)
    n_components = 10

    # Apply PCA to the data
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(data1)


    print('Transformed data shape:', X.shape)

    # X = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    y = label.ravel()
    print(X.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def normalise(data):
        data_min = np.min(data)
        data_max = np.max(data)
        data_norm = (data - data_min) / (data_max - data_min)
        return data_norm

    x_train = normalise(x_train)

    print(np.shape(x_train))
    print(np.shape(y_train))

    
    # # Load iris dataset
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # # Split data into train and test sets
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def sigmoid(x):
        return 1/(1 + np.exp(-x)) 

    def relu(x):
        return np.maximum(0, x)
    
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=0))
        s_z = e_z / e_z.sum(axis=0)
        return s_z.flatten()

    def tanh(x):
        return np.tanh(x)

    def forward(x, w, activation):
        return activation(np.matmul(x, w))

    def accuracy_fn(y, y_hat):
        return (np.where(y == y_hat)[0].size / y_hat.size)
    
    def mse_loss(y,y_hat):
        n = y.size
        loss = np.sum((y - y_hat)**2) / n
        return loss

    def predict(x, y_hat, weights):
        predictions = np.zeros(shape=(x.shape[0]))
        loss = 0
        for idx in range(x.shape[0]):
            r1 = x[idx, :]
            layer=1
            for curr_weights in weights:
                if layer <=  hidden_layers:
                    activation=relu
                    r1 = forward(r1, curr_weights, activation)
                else:
                    activation=tanh
                    r1 = forward(r1, curr_weights, activation)
                layer+=1
            predictions[idx] = return_key(np.where(r1 == np.max(r1))[0][0])
            loss += mse_loss(y_hat[idx], r1)

        accuracy = accuracy_fn(predictions, y_hat)
        return accuracy, loss, predictions
        
    def fitness(x, y_hat, weights):                     
        accuracy = np.empty(shape=(weights.shape[0]))
        for idx in range(weights.shape[0]):
            accuracy[idx],_, _ = predict(x, y_hat, weights[idx, :])
        return accuracy

    # Convert the nD weight matrices into a 1D weight vector
    def mat_to_vector(mat_pop_weights):
        weights_vector = []
        for idx in range(mat_pop_weights.shape[0]):
            curr_vector = []
            for layer_idx in range(mat_pop_weights.shape[1]):
                for val in mat_pop_weights[idx, layer_idx]:
                    curr_vector.extend(val)
            weights_vector.append(curr_vector)
        return np.array(weights_vector)

    # Convert the  1D weight vector into a nD weight matrices
    def vector_to_mat(vector_weights, mat_pop_weights):
        mat_weights = []
        for idx in range(mat_pop_weights.shape[0]):
            start = 0
            end = 0
            for layer_idx in range(mat_pop_weights.shape[1]):
                end = end + mat_pop_weights[idx, layer_idx].size
                curr_vector = vector_weights[idx, start:end]
                mat_layer_weights = np.reshape(curr_vector, newshape=(mat_pop_weights[idx, layer_idx].shape))
                mat_weights.append(mat_layer_weights)
                start = end
        return np.reshape(mat_weights, newshape=mat_pop_weights.shape)

    def mating_pool(pop, fitness, num_parents):
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99
        return parents

    def crossover(parents, offspring_size, crossover_prob):
        offspring = np.empty(offspring_size)
        crossover_point = np.uint32(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            
            parent1_idx = k%parents.shape[0]
            parent2_idx = (k+1)%parents.shape[0]
            
            if np.random.rand() < crossover_prob:
                offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            else:
                # If crossover is not performed, copy one of the parents
                offspring[k, :] = parents[parent1_idx, :]
            
        return offspring

    def mutation(offspring_crossover, mutation_percent):
        num_mutations = np.uint32((mutation_percent*offspring_crossover.shape[1]))
        mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
        
        for idx in range(offspring_crossover.shape[0]):
            random_value = np.random.uniform(low_val, high_val, 1)
            offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
        
        return offspring_crossover

    # solution_per_population = 20
    # num_parents_mating = 10
    # num_generations = 200
    # mutation_percent = 0.02
    # crossover_prob = 0.90
    # hidden_layers = 2
    # hidden_nodes = [32,64]
    # low_val = -3
    # high_val = 3

    input_shape = x_train.shape[1]
    output_shape = len(np.unique(y_train))


    initial_weights = []
    for curr_sol in np.arange(0, solution_per_population):
        # Initialize weights for input layer
        w1 = np.random.uniform(low_val,high_val, size=(input_shape, hidden_nodes[0]))
        initial_weights.append(w1)
    
        # Initialize weights for hidden layers
        for i in range(hidden_layers - 1):
            w2 = np.random.uniform(low_val, high_val, size=(hidden_nodes[i], hidden_nodes[i+1]))
            initial_weights.append(w2)
    
        # Initialize weights for output layer
        w3 = np.random.uniform(low_val, high_val, size=(hidden_nodes[-1], output_shape))
        initial_weights.append(w3)

    initial_weights = np.array(initial_weights).reshape((solution_per_population, len(hidden_nodes)+1))

    weights_mat = np.array(initial_weights)
    weights_vector = mat_to_vector(weights_mat)

    accuracy_values = []
    accuracies = np.empty(shape=(num_generations))
    


    for generation in tqdm(range(num_generations)):

        # vector to matrix
        weights_mat = vector_to_mat(weights_vector, weights_mat)

        # fitness of the population
        fit = fitness(x_train, y_train, weights_mat)
        _,_, predictions = predict(x_train, y_train, vector_to_mat(weights_vector, weights_mat)[0,:])
        accuracy = accuracy_fn(y_train, predictions)
        accuracy_values.append(accuracy)
        if accuracy >= conv_threshold:
            print("Convergence reached after {} generations!".format(generation))
            break
        
        # assign first fitness to the array
        accuracies[generation] = fit[0]

        # selecting mating parents from pool
        parents = mating_pool(weights_vector, fit.copy(), num_parents_mating)

        # generate new population using crossover
        offspring_crossover = crossover(parents, offspring_size=(weights_vector.shape[0]-parents.shape[0], weights_vector.shape[1]), crossover_prob =crossover_prob)

        # adding mutation to the population
        offspring_mutation = mutation(offspring_crossover, mutation_percent=mutation_percent)
        
        # new population combining parents of crossver and mut
        weights_vector[0:parents.shape[0], :] = parents
        weights_vector[parents.shape[0]:, :] = offspring_mutation

        
    weights_mat = vector_to_mat(weights_vector, weights_mat)
    best_weights = weights_mat [0, :]
    acc,loss, predictions = predict(x_train, y_train, best_weights)
    print("Accuracy of the best solution is : ", acc)
    print("loss of the best solution is : ", loss)

    # # Create a Tkinter root window
    # root1 = tk.Tk()
    # root1.withdraw()

    # # Open a file dialog to get the file name and path
    # file_path1 = filedialog.asksaveasfilename(defaultextension='.png')


    plt.plot(range(len(accuracy_values)), accuracy_values)
    plt.xlabel('Generation (accuracy = {})'.format(acc))
    plt.ylabel('Accuracy')
    # plt.xticks(np.arange(0, num_generations+1, 100))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.title("Convergence of Genetic Algorithm")
    fig = plt.gcf()
    plt.show()

    ####################################################
    # # Define colormap for pseudocoloring
    # cmap = plt.cm.get_cmap('jet', np.max(label) + 1)  # Choose a colormap (e.g., 'jet') and specify the number of colors based on the maximum label value
    # label_colored = cmap(label)

    # # Create a figure with a 2x2 grid of subplots
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot the first image
    # cmap = plt.cm.get_cmap('jet', np.max(label) + 1)
    # label_colored = cmap(label)
    # pcm = axs[0].imshow(label_colored)
    # axs[0].set_title('Pseudocolored True Label Image')
    # cbar = fig.colorbar(pcm, ax=axs[0], ticks=range(np.max(label) + 1), orientation='vertical')
    # cbar.ax.set_yticklabels(range(np.max(label) + 1))
    # cbar.set_label('Labels')

    # # Plot the second image
    # x_total = normalise(X)
    # _, _, pred = predict(x_total, y, best_weights)
    # y_champ_ovo = pred.reshape(data.shape[0], data.shape[1])
    # cmap = plt.cm.get_cmap('jet', np.max(y_champ_ovo) + 1)
    # label_colored1 = cmap(y_champ_ovo)
    # pcm = axs[1].imshow(label_colored1)
    # axs[1].set_title('Pseudocolored predict Label Image')
    # cbar = fig.colorbar(pcm, ax=axs[1], ticks=range(np.max(label) + 1), orientation='vertical')
    # cbar.ax.set_yticklabels(range(np.max(label) + 1))
    # cbar.set_label('Labels')

    # plt.show()

    #####################################################
    

    def save_data():
        # prompt the user to select a file location and name for the CSV file
        csv_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        # check if the user canceled the dialog
        if csv_filename:
            # open the file for writing
            with open(csv_filename, 'w', newline='') as csvfile:
                # create a CSV writer object
                writer = csv.writer(csvfile)
                # write each row of the matrix as a CSV row
                for row in best_weights:
                    writer.writerow(row)
            print(f"Matrix saved to {csv_filename}")
        
        # prompt the user to select a file location and name for the image file
        img_filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        # check if the user canceled the dialog
        if img_filename:
            # save the plot as an image file
            fig.savefig(img_filename)
            print(f"Plot saved to {img_filename}")

    # create a GUI button to trigger the save function
    root = tk.Tk()
    root.title("Save Output Files")
    root.geometry("100x100")
    save_button = tk.Button(root, text="Save Outputs", command=save_data)
    save_button.pack()

    root.mainloop()
