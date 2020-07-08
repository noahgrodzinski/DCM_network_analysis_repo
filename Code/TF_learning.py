
#import all the relevant libraries
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import scipy.stats 
from scipy.stats import spearmanr
import statistics


#extract the data from the csv file
raw_dataset = pd.read_csv('author_stats.csv')
dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset = dataset.drop(["id", "name_of_author", "institute", "country", "institute_ID", "citation_count", "average_citations", "total_papers", "paper_freq", 'author_classification'], axis=1) #remove irrelivant data from each author
#print(dataset.tail())

#generate test and train datasets.
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Descipbe & visualise the stats
sns.pairplot(train_dataset, y_vars=["h_index"], x_vars=["average_DCM_papers_of_coauthors", "connected_institutes", "connections", "second_order_connections", "average_h_index_of_coauthors"], diag_kind="kde", kind='reg')
#plt.savefig('figs/variables_with_h.pdf')
#plt.show()
plt.close()

train_stats = train_dataset.describe()
train_stats.pop("h_index")
train_stats = train_stats.transpose()
#train_stats.to_csv('readouts&stats/train_stats.csv')

#split features from labels & normalise data
train_labels = train_dataset.pop('h_index')
test_labels = test_dataset.pop('h_index')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#create and train the network
retrain = input("Would you like to retrain the neural network (reccomended no)(type Y for yes): ")
if retrain =='Y':
  def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
      layers.Dense(32, activation='relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model
  model = build_model()


  EPOCHS = 1000

  history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  #print(hist.tail())

  plt.close()
  plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

  plotter.plot({'Basic': history}, metric = "mae")
  plt.ylim([0, 20])
  plt.ylabel('MAE [h_index]')
  plt.savefig('figs/learning.pdf')
  plt.show()
  plt.close()

  #model.save('saved_model/DCM_h_index_prediction_model') 

#load the network. This will be the newly trained network if Y was entered above
new_model = tf.keras.models.load_model('saved_model/DCM_h_index_prediction_model')

#use the network to generate predictions
test_predictions = new_model.predict(normed_test_data).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [h index]')
plt.ylabel('Predictions based on network factors [h index]')
lims = [0, 100]
plt.xlim(lims)
plt.ylim(lims)
b = plt.plot(lims, lims)
plt.savefig('figs/accuracy_scatter_plot_largelims.pdf')
plt.show()
plt.close()
# 

#analyse network accuracy
coef, p = spearmanr(test_predictions, test_labels)

#analyse network precision
error = test_predictions - test_labels
sns.kdeplot(error, shade=True)
plt.xlabel("Prediction Error [h_index]")
plt.ylabel("Count (normalised)")
#plt.savefig("figs/error_KDE_plot.pdf")
#plt.show()
plt.close()


abs_error = abs(error)
standard_dev = statistics.stdev(abs_error)
mean = statistics.mean(abs_error)
median = statistics.median(abs_error)

perc_error = abs((test_predictions - test_labels)/test_labels)*100
median_rel = statistics.median(perc_error)

#generate readout file with network information
with open('readouts/network_accuracy.txt', 'w') as printout:
    stat_sig = "network has Spearman's Rank correlation coefficient of "+str(coef)+", corresponding to a probability of "+str(p)
    accur = "the network has a median error of "+str(median)+", a mean error of "+str(mean)+", and a standard deviation of error of "+str(standard_dev)
    rel_accur = "the median relative error was "+str(median_rel)
    printout.writelines([stat_sig, accur, rel_accur])


