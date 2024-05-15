import tkinter as tk
from tkinter import filedialog
import pandas as pd
from Running_GA_6 import user_inputs
from scipy.io import loadmat
import os

# Create the main window
window = tk.Tk()
#Set the geometry of tkinter frame
window.geometry("410x350")
window.title("Genetic Algorithm Configuration")

# Define the global variables
solution_per_population_GUI = 10
num_parents_mating_GUI = 4
num_generations_GUI = 100
conv_threshold_GUI = 0.95
mutation_percent_GUI = 0.02
crossover_prob_GUI = 0.9
hidden_layers_GUI = 2
hidden_nodes_GUI = [22,22]
low_val_GUI = -1
high_val_GUI = 1

# solution_per_population = tk.IntVar(value=10)
# num_parents_mating = tk.IntVar(value=3)
# num_generations = tk.IntVar(value=100)
# conv_threshold = tk.IntVar(value=0.98)
# mutation_percent = tk.IntVar(value=0.04)
# crossover_prob = tk.IntVar(value=0.7)
# hidden_layers = tk.IntVar(value=8)
# hidden_nodes = tk.IntVar(value=[10,20])
# low_val = tk.IntVar(value=-1) 
# high_val = tk.IntVar(value=1)

# # Define a function to load a CSV file
# def load_csv():
#     global csv_data
#     # Show a file dialog to select a CSV file
#     file_path = filedialog.askopenfilename(initialdir = "D:\\GNR Sem 2\\ASIP\\Programming Assignment", title="Select CSV file", filetypes=[("CSV files", "*.csv")])
#     # Load the CSV file using pandas and store the data in the global variable
#     csv_data = pd.read_csv(file_path)

# Define the function to load the first TIFF file
def load_image1():
    global image1_data
    image1_data =  loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='SELECT indian_pines_corrected file'))
    # if file_path:
    #     image1_data = file_path
    #     print("First TIFF file loaded:", image1_data)

# Define the function to load the second TIFF file
def load_image2():
    global image2_data
    image2_data = loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='SELECT indian_pines_gt file'))
    # if file_path:
    #     image2_data = file_path
    #     print("Second TIFF file loaded:", image2_data)


def update_values():
    #Retrive the user given values
    user_solution_per_population = int(solution_per_population_entry.get())
    user_num_parents_mating = int(num_parents_mating_entry.get())
    user_num_generations = int(num_generations_entry.get())
    user_conv_threshold = float(conv_threshold_entry.get())
    user_mutation_percent = float(mutation_percent_entry.get())
    user_crossover_prob = float(crossover_prob_entry.get())
    user_hidden_layers = int(hidden_layers_entry.get())
    user_hidden_nodes = list(map(int, hidden_nodes_entry.get().split()))
    user_low_val = int(low_val_entry.get())
    user_high_val = int(high_val_entry.get())

    global solution_per_population
    solution_per_population = user_solution_per_population
    global num_parents_mating
    num_parents_mating = user_num_parents_mating
    global num_generations
    num_generations = user_num_generations
    global conv_threshold
    conv_threshold = user_conv_threshold
    global mutation_percent
    mutation_percent = user_mutation_percent
    global crossover_prob
    crossover_prob = user_crossover_prob
    global hidden_layers
    hidden_layers = user_hidden_layers
    global hidden_nodes
    hidden_nodes = user_hidden_nodes
    global low_val
    low_val = user_low_val
    global high_val
    high_val = user_high_val

    print("User Given Values are:\n")
    print(solution_per_population)
    print(num_parents_mating)
    print(num_generations)
    print(conv_threshold)
    print(mutation_percent)
    print(crossover_prob)
    print(hidden_layers)
    print(hidden_nodes)
    print(low_val)
    print(high_val)

#     return solution_per_population, num_parents_mating, num_generations, conv_threshold, 
# mutation_percent, crossover_prob, hidden_layers, hidden_nodes, low_val, high_val


# Define the function to run the genetic algorithm with user-specified parameters and loaded data
def run_algorithm():
    #TODO: Implement the genetic algorithm using the user inputs and loaded data
    print("Running genetic algorithm with the following parameters:")



    # # update the value of the variable
    # solution_per_population.set(user_solution_per_population)
    # num_parents_mating.set(user_num_parents_mating)
    # num_generations.set(user_num_generations)
    # conv_threshold.set(user_conv_threshold)
    # mutation_percent.set(user_mutation_percent)
    # crossover_prob.set(user_crossover_prob)
    # hidden_layers.set(user_hidden_layers)
    # hidden_nodes.set(user_hidden_nodes)
    # low_val.set(user_low_val)
    # high_val.set(user_high_val)

    print("- Solution per population:", solution_per_population_entry.get())
    print("- Number of parents mating:", num_parents_mating_entry.get())
    print("- Number of generations:", num_generations_entry.get())
    print("- Convergence Threshold:", conv_threshold_entry.get())
    print("- Mutation percent:", mutation_percent_entry.get())
    print("- Crossover probability:", crossover_prob_entry.get())
    print("- Hidden layers:", hidden_layers_entry.get())
    print("- Hidden nodes:",list(map(int, hidden_nodes_entry.get().split())))
    print("- Low value:", low_val_entry.get())
    print("- High value:", high_val_entry.get())
    # print("- CSV data:", csv_data)
    print("- Image 1 data:", image1_data)
    print("- Image 2 data:", image2_data)

    # solution_per_population = int(solution_per_population_entry.get())
    # num_parents_mating = int(num_parents_mating_entry.get())
    # num_generations = int(num_generations_entry.get())
    # conv_threshold = float(conv_threshold_entry.get())
    # mutation_percent = float(mutation_percent_entry.get())
    # crossover_prob = float(crossover_prob_entry.get())
    # hidden_layers = int(hidden_layers_entry.get())
    # hidden_nodes = hidden_nodes_entry.get()
    # low_val = int(low_val_entry.get())
    # high_val = int(high_val_entry.get())

    window.destroy()

# Create the widgets for loading the CSV file
# load_csv_button = tk.Button(window, text="Load CSV File", command=load_csv)

# Create the widgets for loading the TIFF files
load_image1_button = tk.Button(window, text="Load First Image File", command=load_image1)
load_image2_button = tk.Button(window, text="Load Second Image File", command=load_image2)

# Create the widgets for user inputs
solution_per_population_label = tk.Label(window, text="Population Size:")
solution_per_population_entry = tk.Entry(window, textvariable=solution_per_population_GUI)

num_parents_mating_label = tk.Label(window, text="Number of parents mating:")
num_parents_mating_entry = tk.Entry(window, textvariable=num_parents_mating_GUI)

num_generations_label = tk.Label(window, text="Number of generations:")
num_generations_entry = tk.Entry(window, textvariable=num_generations_GUI)

conv_threshold_label = tk.Label(window, text="Convergence Threshold:")
conv_threshold_entry = tk.Entry(window, textvariable=conv_threshold_GUI)

mutation_percent_label = tk.Label(window, text="Mutation Probability:")
mutation_percent_entry = tk.Entry(window, textvariable=mutation_percent_GUI)

crossover_prob_label = tk.Label(window, text="Crossover Probability:")
crossover_prob_entry = tk.Entry(window, textvariable=crossover_prob_GUI)

hidden_layers_label = tk.Label(window, text="Hidden layers:")
hidden_layers_entry = tk.Entry(window, textvariable=hidden_layers_GUI)

hidden_nodes_label = tk.Label(window, text="Number of nodes in each hidden layer (Space-separated):")
hidden_nodes_entry = tk.Entry(window, textvariable=hidden_nodes_GUI)

low_val_label = tk.Label(window, text="Lower bound for initial population values:")
low_val_entry = tk.Entry(window, textvariable=low_val_GUI)

high_val_label = tk.Label(window, text="Upper bound for initial population values:")
high_val_entry = tk.Entry(window, textvariable=high_val_GUI)

# Create a "Sumit New Values" button that calls the update_values function when clicked
submit_values_button = tk.Button(window, text="Submit Values", command=update_values)

# Create a "Run" button that calls the run_algorithm function when clicked
run_button = tk.Button(window, text="Run", command=run_algorithm)

# Use the grid layout to place the widgets in the window
# load_csv_button.grid(row=0, column=0)

solution_per_population_label.grid(row=1, column=0)
solution_per_population_entry.grid(row=1, column=1)

num_parents_mating_label.grid(row=2, column=0)
num_parents_mating_entry.grid(row=2, column=1)

num_generations_label.grid(row=3, column=0)
num_generations_entry.grid(row=3, column=1)

conv_threshold_label.grid(row=4, column=0)
conv_threshold_entry.grid(row=4, column=1)

mutation_percent_label.grid(row=5, column=0)
mutation_percent_entry.grid(row=5, column=1)

crossover_prob_label.grid(row=6, column=0)
crossover_prob_entry.grid(row=6, column=1)

hidden_layers_label.grid(row=7, column=0)
hidden_layers_entry.grid(row=7, column=1)

hidden_nodes_label.grid(row=8, column=0)
hidden_nodes_entry.grid(row=8, column=1)

low_val_label.grid(row=9, column=0)
low_val_entry.grid(row=9, column=1)

high_val_label.grid(row=10, column=0)
high_val_entry.grid(row=10, column=1)

submit_values_button.grid(row=11, column=0, columnspan=2, pady=10)

load_image1_button.grid(row=12, column=0)
load_image2_button.grid(row=13, column=0)


run_button.grid(row=14, column=0, columnspan=2, pady=10)



# Start the tkinter event loop
window.mainloop()



print(solution_per_population)
print(num_parents_mating)
print(num_generations)
print(conv_threshold)
print(mutation_percent) 
print(crossover_prob)
print(hidden_layers)
print(hidden_nodes)
print(low_val)
print(high_val)

user_inputs(image1_data, image2_data, solution_per_population, num_parents_mating,num_generations,conv_threshold, 
mutation_percent,crossover_prob,hidden_layers,hidden_nodes,low_val,high_val)