#relevant imports
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import scipy.stats 
import statistics
import csv
from matplotlib.pyplot import figure

#gather the metadata
stats = pd.read_csv('readouts/train_stats.csv')
column_names = stats.values.T.tolist()[0]
print(column_names[2])

#extract the useful information, and normalise it (so the inputs are of the same format as what the network was trained on)
means_raw = stats['mean']
means_raw = np.array(means_raw)
#means = np.delete(means, [0])
def norm(x):
    return (x - stats['mean']) / stats['std']
means = norm(means_raw)
means = np.array(means)

stds_raw = stats['std']
stds_raw = np.array(stds_raw)



#generate varied inputs around a particular index of author information. All other values are held at mean.
def generate_varied_values(index, interval_in_stds, num_to_either_side):
    set_of_inputs = []
    set_of_inputs.append(np.ndarray.tolist(means))

    set_of_values = [means[index]]
    
    for interval_index in range(num_to_either_side):
        forward_set = np.ndarray.tolist(means)
        backward_set = np.ndarray.tolist(means)


        forward_set[index] += interval_in_stds * (interval_index+1) #as data normalised, so σ=1
        set_of_inputs.append(forward_set)
        set_of_values.append(forward_set[index])

        backward_set[index] -= interval_in_stds * (interval_index+1) #as data normalised, so σ=1
        set_of_inputs.insert(0, backward_set)
        set_of_values.insert(0, backward_set[index])

    return np.array(set_of_inputs), set_of_values

#load the network model
new_model = tf.keras.models.load_model('saved_model/DCM_h_index_prediction_model')

#generate a graph of varied network prediction as a single variable is changed
def show_graph_of_varied_input(index_of_input, save_graph_bool):
    varied_input_name = column_names[index_of_input]
    index_varied_inputs, index_varied = generate_varied_values(index_of_input, 0.1, 20)
    varied_predictions = new_model.predict(index_varied_inputs).flatten()
    
    plt.plot(index_varied, varied_predictions, marker='x', markersize=0.0)
    #plt.xlabel('varied inputs in '+varied_input_name+" (standard deviations from mean)")
    plt.ylabel('h index predictions')
    #plt.axvline(x=means[index_of_input], ymin=0, ymax=float(new_model.predict(np.array([means])).flatten()), label="mean")

    axes1 = plt.gca()
    axes2 = axes1.twiny()

    axes2.set_xticks([x*1.0 for x in range(-2, 3)])
    tick_values=[(means_raw[index_of_input]+i*stds_raw[index_of_input]*0.5) for i in range(-2, 3)]
    tick_values = [str(x)[0:5] for x in tick_values]
    axes2.set_xticklabels(tick_values)

    axes1.set_xlabel('varied inputs in '+varied_input_name+" (standard deviations from mean)")
    axes2.set_xlabel('varied inputs in '+varied_input_name+" (true value)")
    
    if save_graph_bool:
        name_of_graph = "figs/varied_inputs/variation_of_h_with_"+varied_input_name+".pdf"
        plt.savefig(name_of_graph)
    plt.close()

    #plt.show()

#create graphs for the varied network predictions, and save the graphs. Counter is in range 14 for the 13 predictions.
for i in range(1, 14):
    show_graph_of_varied_input(i, True)

