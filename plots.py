"""
This file contains reusable functions for making plots of data.
The functions assume the data is passed as a dictionary where the keys are strings specifying the countries and the
values are the variable of interest for the corresponding nation.

TODO: Use ax.hline in ECDF.
TODO: Get clipping to work correctly.
TODO: Use walrus operator ;).
"""


import matplotlib.pyplot as plt
import numpy as np
import random


def create_ECDF(data_dict, x_label="X", y_label="Y", title="ECDF Plot", log_x=False, marks=[]):
    """
    Plots the empirical cumulative distribution of the values in the data dict.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted, and a
    list of nations to mark on the graph.
    """

    # create the X and Y data
    X = sorted(data_dict.values())
    if log_x:
        X = np.log(X)
    # X = np.clip(X, np.nanmin(X[X != -np.inf]), np.nanmax(X[X != np.inf]))
    Y = np.arange(len(X)) / len(X)

    # plot the ECDF
    plt.plot(X, Y, '.', markersize=20)
    if len(marks) > 0:
        for nation in marks:
            nation_X = data_dict[nation]
            if log_x:
                nation_X = np.log(nation_X)
            plt.plot([nation_X, nation_X], [0, 1], '--', label=nation)
        plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_histogram(data_dict, x_label="X", y_label="Count", title="ECDF Plot", log_x=False, bins="auto"):
    """
    Plots a histogram of the values in the data dict.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted,
    and bin size.
    """

    # create the X and Y data
    X = list(data_dict.values())
    if log_x:
        X = np.log(X)
    # X = np.clip(X, np.nanmin(X[X != -np.inf]), np.nanmax(X[X != np.inf]))

    # plot the histogram
    plt.hist(X, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def create_scatter_plot(x_data_dict, y_data_dict, x_label="X", y_label="Y", title="Scatter Plot", log_x=False, log_y=False, marks=[]):
    """
    Plots a scatter plot of the values in the data dicts.

    The required argument is a data dict (as described at the beginning of the file).
    Optional arguments are axis labels, plot title, a boolean determining whether the log of x values are plotted, and a
    list of nations to mark on the graph.
    """

    # create the X and Y data
    countries = set(list(x_data_dict.keys()) + list(y_data_dict.keys()))
    countries = [x for x in countries if x in x_data_dict.keys() and x in y_data_dict.keys()]
    X = [x_data_dict[key] for key in countries if key not in marks]
    Y = [y_data_dict[key] for key in countries if key not in marks]
    marked_X = [x_data_dict[key] for key in marks]
    marked_Y = [y_data_dict[key] for key in marks]
    if log_x:
        X = np.log(X)
        marked_X = np.log(marked_X)
    if log_y:
        Y = np.log(Y)
        marked_Y = np.log(marked_Y)
    # X = np.clip(X, np.nanmin(X[X != -np.inf]), np.nanmax(X[X != np.inf]))
    # Y = np.clip(Y, np.nanmin(Y[Y != -np.inf]), np.nanmax(Y[Y != np.inf]))

    # plot the scatter plot
    fig, ax = plt.subplots()
    plt.scatter(X, Y)
    if len(marks) > 0:
        plt.scatter(marked_X, marked_Y)
        for mark in marks:
            mark_X = x_data_dict[mark]
            mark_Y = y_data_dict[mark]
            if log_x:
                mark_X = np.log(mark_X)
            if log_y:
                mark_Y = np.log(mark_Y)
            mark_Y += max(max(Y), max(marked_Y))/60
            ax.annotate(mark,
                        (mark_X, mark_Y),
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def generate_sample_data(num_samples=100, num_marks=5):
    """Returns sample data with which to test plotting functions."""

    X = np.random.normal(100, 25, (num_samples,))
    Y = [f"Nation {i}" for i in range(num_samples)]
    data_dict = dict(zip(Y, X))
    nations_to_mark = [random.choice(list(data_dict.keys())) for i in range(num_marks)]

    return data_dict, nations_to_mark


# running this file will produce samples of each plot
if __name__ == "__main__":

    # create sample data
    sample_data_dict_1, sample_marks_1 = generate_sample_data()
    sample_data_dict_2, _ = generate_sample_data()

    # create an ECDF
    create_ECDF(sample_data_dict_1, marks=sample_marks_1)

    # create a histogram
    create_histogram(sample_data_dict_1)

    # create a scatter plot
    create_scatter_plot(sample_data_dict_1, sample_data_dict_2, marks=sample_marks_1)


"""
 # fit the classifier
        self.classifier.fit(x_train, y_train)
        print(f"Got a mean cross validation accuracy of {self.classifier.best_score_:.4f}.")
        print(f"The parameters that worked best were {self.classifier.best_params_}.")

        # test the brain
        FIG_SAVE_PATH = "confusion_matrix.png"
        y_pred = self.classifier.predict(x_test)
        print("Here is a classification report on the testing data...")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print("Here is the confusion matrix of the testing data.")
        print(cm)

        # save a confusion matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm_ax = ax.matshow(np.log(cm + .01))  # we add .01 to avoid dividing by 0 error
        ax.set_xticklabels([""] + list(self.classifier.classes_))
        ax.set_yticklabels([""] + list(self.classifier.classes_))
        fig.colorbar(cm_ax)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        fig.savefig(FIG_SAVE_PATH)
        print(f"Saved confusion matrix to '{FIG_SAVE_PATH}'.")
"""