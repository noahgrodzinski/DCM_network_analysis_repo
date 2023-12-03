# import all the relevant libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

# print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots as tf_plt
import tensorflow_docs.modeling
from scipy.stats import spearmanr
import statistics


def norm(x, train_stats):
    return (x - train_stats["mean"]) / train_stats["std"]


def create_dataframes():
    # extract the data from the csv file
    raw_dataset = pd.read_csv("code/author_stats.csv")
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset = dataset.drop(
        [
            "id",
            "name_of_author",
            "institute",
            "country",
            "institute_ID",
            "citation_count",
            "average_citations",
            "total_papers",
            "paper_freq",
            "author_classification",
        ],
        axis=1,
    )  # remove irrelivant data from each author
    # print(dataset.tail())

    # Create squared input vectors
    for x in dataset:
        dataset[f"{x}_squared"] = dataset[x] ** 2
    # generate test and train datasets.
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Descipbe & visualise the stats
    sns.pairplot(
        train_dataset,
        y_vars=["h_index"],
        x_vars=[
            "average_DCM_papers_of_coauthors",
            "connected_institutes",
            "connections",
            "second_order_connections",
            "average_h_index_of_coauthors",
        ],
        diag_kind="kde",
        kind="reg",
    )
    plt.savefig("code/figs/variables_with_h.pdf")
    # plt.show()
    plt.close()

    train_stats = train_dataset.describe()
    train_stats.pop("h_index")
    train_stats = train_stats.transpose()
    train_stats.to_csv("code/readouts&stats/train_stats.csv")

    # split features from labels & normalise data
    train_labels = train_dataset.pop("h_index")
    test_labels = test_dataset.pop("h_index")

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    return (
        train_labels,
        test_labels,
        train_dataset,
        test_dataset,
        normed_train_data,
        normed_test_data,
    )


# Create and train the network
def train_new_model(
    train_labels,
    train_dataset,
    normed_train_data,
    EPOCHS=10000,
    model_save_path: str = "code/saved_model/",
):
    def build_model():
        model = keras.Sequential(
            [
                layers.Dense(256, activation="leaky_relu"),
                layers.Dense(128, activation="leaky_relu"),
                layers.Dense(16, activation="leaky_relu"),
                layers.Dense(16, activation="leaky_relu"),
                layers.Dense(1, activation="relu"),
            ]
        )

        # RMSoptimizer = keras.optimizers.RMSprop(0.0001)
        SGDoptimizer = keras.optimizers.SGD(
            learning_rate=0.00001, momentum=0.99
        )  # This worked really well with patience=200 and (1024,128,16,16,1) mae~0.7
        # ADAMoptimizer = keras.optimizers.Adam(
        #     learning_rate=5e-5,
        # )

        model.compile(loss="mse", optimizer=SGDoptimizer, metrics=["mae", "mse"])
        return model

    model = build_model()

    history = model.fit(
        normed_train_data,
        train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tfdocs.modeling.EpochDots(),
            keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True),
        ],
    )

    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    # print(hist.tail())

    plt.close()
    plotter = tf_plt.HistoryPlotter(smoothing_std=2)

    plotter.plot({"Basic": history}, metric="mae")
    plt.ylim([0, 5])
    plt.ylabel("MAE [h_index]")
    plt.savefig("code/figs/learning.pdf")
    # plt.show()
    plt.close()

    model.save(model_save_path, overwrite=True)
    return model_save_path


def make_network_statistics_and_graphs(
    test_labels, normed_test_data, saved_model_path: str = "code/saved_model/"
):
    """ "
    Function to make network statistics and graphs
    """
    # load the network. This will be the newly trained network if Y was entered above
    new_model = tf.keras.models.load_model(saved_model_path)

    # use the network to generate predictions
    test_predictions = new_model.predict(normed_test_data).flatten()
    test_predictions = [round(x, 0) for x in test_predictions]

    # Create graphs and stats
    plt.axes(aspect="equal")
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True Values [h index]")
    plt.ylabel("Predictions based on network factors [h index]")
    lims = [0, 100]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig("code/figs/accuracy_scatter_plot_largelims.pdf")
    # plt.show()
    plt.close()

    plt.axes(aspect="equal")
    plt.scatter(test_labels, test_predictions)
    plt.xlabel("True Values [h index]")
    plt.ylabel("Predictions based on network factors [h index]")
    lims = [0, 60]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig("code/figs/accuracy_scatter_plot.pdf")
    # plt.show()
    plt.close()

    # analyse network accuracy
    coef, p = spearmanr(test_predictions, test_labels)

    # analyse network precision
    error = test_predictions - test_labels
    sns.kdeplot(error, fill=True)
    plt.xlabel("Prediction Error [h_index]")
    plt.ylabel("Count (normalised)")
    plt.savefig("code/figs/error_KDE_plot.pdf")
    # plt.show()
    plt.close()

    abs_error = abs(error)
    standard_dev = statistics.stdev(abs_error)
    mean = statistics.mean(abs_error)
    median = statistics.median(abs_error)

    perc_error = abs((test_predictions - test_labels) / test_labels) * 100
    median_rel = statistics.median(perc_error)

    # generate readout file with network information
    with open("code/readouts&stats/network_accuracy.txt", "w") as printout:
        stat_sig = (
            "network has Spearman's Rank correlation coefficient of "
            + str(coef)
            + ", corresponding to a probability of "
            + str(p)
            + "\n"
        )
        accur = (
            "the network has a median error of "
            + str(median)
            + ", a mean error of "
            + str(mean)
            + ", and a standard deviation of error of "
            + str(standard_dev)
            + "\n"
        )
        rel_accur = "the median relative error was " + str(median_rel)
        printout.writelines([stat_sig, accur, rel_accur])


if __name__ == "__main__":
    (
        train_labels,
        test_labels,
        train_dataset,
        test_dataset,
        normed_train_data,
        normed_test_data,
    ) = create_dataframes()
    # retrain = input("retrain model? (y/n): ")
    retrain = "y"
    if retrain == "y":
        model_path = train_new_model(train_labels, train_dataset, normed_train_data)
    elif retrain.lower() == "n":
        model_path = None
    else:
        print("Invalid answer, not training")
        model_path = None

    if model_path:
        make_network_statistics_and_graphs(test_labels, normed_test_data, model_path)
    else:
        model_path = input("Specify path to old model to make new statistics: ")
        if model_path:
            make_network_statistics_and_graphs(
                test_labels, normed_test_data, model_path
            )
        else:
            print("Using previous model")
            make_network_statistics_and_graphs(
                test_labels, normed_test_data
            )  # Use last model instead
