import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(37)

from regression_plot import exploratory_plots
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_models import ml_weights
from regression_models import regularised_ml_weights
from regression_models import linear_model_predict
from regression_train_test import root_mean_squared_error
from regression_train_test import train_and_test
from regression_plot import plot_train_test_errors
from regression_models import construct_knn_approx
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import expand_to_monomials
from regression_models import  prediction_function
from regression_models import  regularised_least_squares_weights


def main( ifname, delimiter=None, columns=None, has_header=True, test_fraction=0.25):
    data, field_names = import_data( ifname, delimiter=delimiter, has_header=has_header, columns=columns)

    # Exploratory Data Analysis (EDA)
    raw_data = pd.read_csv('datafile.csv', sep=";")
    # view correlation efficieny result where |r|=1 has the strongest relation and |r|=0 the weakest
    df = pd.DataFrame(data=raw_data)
    print(df.corr())

    # view data if it is normally distributed
    plt.hist(raw_data["quality"], range=(1, 10), edgecolor='black', linewidth=1)
    plt.xlabel('quality')
    plt.ylabel('amount of samples')
    plt.title("distribution of red wine quality")

    # feature selection
    import scipy.stats as stats
    class ChiSquare:
        def __init__(self, dataframe):
            self.df = dataframe
            self.p = None  # P-Value
            self.chi2 = None  # Chi Test Statistic
            self.dof = None

            self.dfObserved = None
            self.dfExpected = None

        def _print_chisquare_result(self, colX, alpha):
            result = ""
            if self.p < alpha:
                result = "{0} is IMPORTANT for Prediction".format(colX)
            else:
                result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

            print(result)

        def TestIndependence(self, colX, colY, alpha=0.05):
            X = self.df[colX].astype(str)
            Y = self.df[colY].astype(str)

            self.dfObserved = pd.crosstab(Y, X)
            chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
            self.p = p
            self.chi2 = chi2
            self.dof = dof

            self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

            self._print_chisquare_result(colX, alpha)
            print('self:%s' % (self), self.chi2, self.p)

    # Initialize ChiSquare Class
    cT = ChiSquare(raw_data)

    # Feature Selection
    testColumns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                   "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    for var in testColumns:
        cT.TestIndependence(colX=var, colY="quality")

    # split data into inputs and targets
    inputs = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    targets = data[:, 11]

    # mean normalisation
    fixed_acidity = inputs[:, 0]
    volatile_acidity = inputs[:, 1]
    citric_acid = inputs[:, 2]
    residual_sugar = inputs[:, 3]
    chlorides = inputs[:, 4]
    free_sulfur_dioxide = inputs[:, 5]
    total_sulfur_dioxide = inputs[:, 6]
    density = inputs[:, 7]
    ph = inputs[:, 8]
    sulphates = inputs[:, 9]
    alcohol = inputs[:, 10]
    # normalize data
    inputs[:, 0] = (fixed_acidity - np.mean(fixed_acidity)) / np.std(fixed_acidity)
    inputs[:, 1] = (volatile_acidity - np.mean(volatile_acidity)) / np.std(volatile_acidity)
    inputs[:, 2] = (citric_acid - np.mean(citric_acid)) / np.std(citric_acid)
    inputs[:, 3] = (residual_sugar - np.mean(residual_sugar)) / np.std(residual_sugar)
    inputs[:, 4] = (chlorides - np.mean(chlorides)) / np.std(chlorides)
    inputs[:, 5] = (free_sulfur_dioxide - np.mean(free_sulfur_dioxide)) / np.std(free_sulfur_dioxide)
    inputs[:, 6] = (total_sulfur_dioxide - np.mean(total_sulfur_dioxide)) / \
                   np.std(total_sulfur_dioxide)
    inputs[:, 7] = (density - np.mean(density)) / np.std(density)
    inputs[:, 8] = (ph - np.mean(ph)) / np.std(ph)
    inputs[:, 9] = (sulphates - np.mean(sulphates)) / np.std(sulphates)
    inputs[:, 10] = (alcohol - np.mean(alcohol)) / np.std(alcohol)

    #draw plot of data set
    normalised_data = np.column_stack((inputs, targets))
    exploratory_plots(normalised_data, field_names)

    # add a colum of x0.ones
    x0 = np.ones(len(targets))
    inputs = np.column_stack((x0, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                              free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol))

    # run all experiments on the same train-test split of the data
    train_part, test_part = train_and_test_split(inputs.shape[0], test_fraction=test_fraction)

    # another evaluation function
    def rsquare(test_targets, test_predicts):
        y_mean = np.mean(test_targets)
        ss_tot = sum((test_targets - y_mean) ** 2)
        ss_res = sum((test_targets - test_predicts) ** 2)
        rsquare = 1 - (ss_res / ss_tot)
        return rsquare

    print('---------------------------Linear Regression-----------------------------------')

    # linear regression
    #train_part, test_part = train_and_test_split(inputs.shape[0], test_fraction=test_fraction)
    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets,
                                                                                      train_part, test_part)
    weights = ml_weights(train_inputs, train_targets)
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)
    print("LR-train_weights", weights)
    print("LR-train_error", train_error)
    print("LR-test_error", test_error)
    print("LR-rsquare score", rsquare(test_targets, test_predicts))
    print("LR-prediction:", test_predicts[:20], "LR-original", test_targets[:20])

    print('----------------Regularised Linear Regression-----------------------------')

    # regularised linear regression
    reg_params = np.logspace(-15, -4, 11)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        # print("RLR-Evaluating reg_para " + str(reg_param))
        train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets,
                                                                                          train_part, test_part)
        reg_weights = regularised_ml_weights(train_inputs, train_targets, reg_param)
        train_predicts = linear_model_predict(train_inputs, reg_weights)
        test_predicts = linear_model_predict(test_inputs, reg_weights)
        train_error = root_mean_squared_error(train_targets, train_predicts)
        test_error = root_mean_squared_error(test_targets, test_predicts)
        train_errors.append(train_error)
        test_errors.append(test_error)

    # best lambda
    test_errors = np.array(test_errors)
    best_l = np.argmin(test_errors)
    print("RLR-Best joint choice of parameters:")
    print("RLR-lambda = %.2g" % (reg_params[best_l]))
    # plot train_test_errors in different reg_params
    fig, ax = plot_train_test_errors("$\lambda$", reg_params, train_errors, test_errors)
    ax.set_xscale('log')
    reg_weights = regularised_ml_weights(train_inputs, train_targets, best_l)
    print("RLR-train_weights", reg_weights)
    print("RLR-train_error", train_errors[best_l])
    print("RLR-test_error", test_errors[best_l])
    print("RLR-rsquare score", rsquare(test_targets, test_predicts))
    print("RLR-prediction:", test_predicts[:20], "RLR-original", test_targets[:20])

    print('-----------------------------kNN Regression------------------------------------')

    # KNN-regression
    # tip out the x0=1 column
    inputs = inputs[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_errors = []
    test_errors = []
    K = range(2, 9)
    for k in K:
        train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(inputs, targets,
                                                                                          train_part, test_part)
        knn_approx = construct_knn_approx(train_inputs, train_targets, k)
        train_knn_predicts = knn_approx(train_inputs)
        train_error = root_mean_squared_error(train_knn_predicts, train_targets)
        test_knn_predicts = knn_approx(test_inputs)
        test_error = root_mean_squared_error(test_knn_predicts, test_targets)
        train_errors.append(train_error)
        test_errors.append(test_error)
        # print("knn_predicts: ", np.around(test_knn_predicts), "knn-original", test_targets)

    # best k
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    best_k = np.argmin(test_errors)
    print("Best joint choice of parameters:")
    print("k = %.2g" % (K[best_k]))
    fig, ax = plot_train_test_errors("K", K, train_errors, test_errors)
    ax.set_xticks(np.arange(min(K), max(K) + 1, 1.0))

    print("kNN-train_error", train_errors[-1])
    print("kNN-test_error", test_errors[-1])
    knn_approx = construct_knn_approx(train_inputs, train_targets, k=3)
    test_predicts = knn_approx(test_inputs)
    print("kNN-rsquare score", rsquare(test_targets, test_predicts))
    print("kNN-y_predicts", test_predicts[:20], 'y_original', test_targets[:20])

    print('----------------------------RBF Function-------------------------------------')

    # Radinal Basis Functions
    # for the centres of the basis functions sample 15% of the data
    sample_fraction = 0.15
    p = (1 - sample_fraction, sample_fraction)
    centres = inputs[np.random.choice([False, True], size=inputs.shape[0], p=p), :]
    print("centres.shape = %r" % (centres.shape,))
    # of the basis functions
    scales = np.logspace(0, 2, 17)
    # choices of regularisation strength
    reg_params = np.logspace(-15, -4, 11)
    # create empty 2d arrays to store the train and test errors
    train_errors = np.empty((scales.size, reg_params.size))
    test_errors = np.empty((scales.size, reg_params.size))
    # iterate over the scales
    for i, scale in enumerate(scales):
        # i is the index, scale is the corresponding scale
        # we must recreate the feature mapping each time for different scales
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs)
        # partition the design matrix and targets into train and test
        train_designmtx, train_targets, test_designmtx, test_targets = \
            train_and_test_partition(designmtx, targets, train_part, test_part)
        # iteratre over the regularisation parameters
        for j, reg_param in enumerate(reg_params):
            # j is the index, reg_param is the corresponding regularisation
            # parameter
            # train and test the data
            train_error, test_error = train_and_test(train_designmtx, train_targets, test_designmtx, test_targets,
                                                     reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_errors[i, j] = train_error
            test_errors[i, j] = test_error
    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    best_i = np.argmin(np.argmin(test_errors, axis=1))
    best_j = np.argmin(test_errors[i, :])
    print("Best joint choice of parameters:")
    print("\tscale= %.2g and lambda = %.2g" % (scales[best_i], reg_params[best_j]))
    # now we can plot the error for different scales using the best
    # regulariation choice
    fig, ax = plot_train_test_errors("scale", scales, train_errors[:, best_j], test_errors[:, best_j])
    ax.set_xscale('log')
    # ...and the error for  different regularisation choices given the best
    # scale choice
    fig, ax = plot_train_test_errors("$\lambda$", reg_params, train_errors[best_i, :], test_errors[best_i, :])
    ax.set_xscale('log')
    feature_mapping = construct_rbf_feature_mapping(centres, scales[best_i])
    reg_weights = regularised_ml_weights(train_designmtx, train_targets, reg_params[best_j])
    # test function
    test_predicts = np.matrix(test_designmtx) * np.matrix(reg_weights).reshape((len(reg_weights), 1))
    test_predicts = np.array(test_predicts).flatten()

    print("RBF-train_error", train_errors[best_i, best_j])
    print("RBF-test_error", test_errors[best_i, best_j])
    print("RBF-rsquare score", rsquare(test_targets, test_predicts))
    print('RBF_y_predicts: ', test_predicts[:20], 'rbf_y_originals: ', test_targets[:20])

    print('-----------------------------Polynomial---------------------------------------')

    # Polynomial Basis Function
    # set input features as 'alcohol'
    degrees = range(1, 10)
    train_errors = []
    test_errors = []
    for degree in degrees:
        processed_inputs = 0
        for i in range(inputs.shape[1]):
            processed_input = expand_to_monomials(inputs[:, i], degree)
            processed_inputs += processed_input
        processed_inputs = np.array(processed_inputs)
        # split data into train and test set
        processed_train_inputs, train_targets, processed_test_inputs, test_targets = train_and_test_partition \
            (processed_inputs, targets, train_part, test_part)
        train_error, test_error = train_and_test(processed_train_inputs, train_targets,
                                                 processed_test_inputs, test_targets, reg_param=None)
        weights = regularised_least_squares_weights(processed_train_inputs, train_targets, reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    print("Polynomial-train error: ", train_errors[-1])
    print("Polynomial-test error: ", test_errors[-1])
    best_d = np.argmin(test_errors)
    print("Best joint choice of degree:")
    final_degree = degrees[best_d]
    print("degree = %.2g" % (final_degree))
    fig, ax = plot_train_test_errors("Degree", degrees, train_errors, test_errors)
    ax.set_xticks(np.arange(min(degrees), max(degrees) + 1, 1.0))

    # test functionality with the final degree
    processed_inputs = 0
    for i in range(inputs.shape[1]):
        processed_input = expand_to_monomials(inputs[:, i], final_degree)
        processed_inputs += processed_input
    processed_inputs = np.array(processed_inputs)

    processed_train_inputs, train_targets, processed_test_inputs, test_targets = train_and_test_partition \
        (processed_inputs, targets, train_part, test_part)
    train_error, test_error = train_and_test(processed_train_inputs, train_targets,
                                             processed_test_inputs, test_targets, reg_param=None)
    weights = regularised_least_squares_weights(processed_train_inputs, train_targets, reg_param)
    # print("processed_train_inputs.shape", processed_train_inputs.shape)
    # print('weights: ', weights, 'weights shape: ', weights.shape)
    test_predicts = prediction_function(processed_test_inputs, weights, final_degree)
    print("Polynomial-rsquare score", rsquare(test_targets, test_predicts))
    print('Polynomial-y_predicts: ', test_predicts[:20], 'Polynomial-y_original: ', test_targets[:20])
    plt.show()


def import_data(ifname, delimiter=None, has_header=False, columns=None):
    # columns -- counting from 0
    # return:
    # data_as_array -- the data as a numpy.array object
    # field_names -- a list of strings of the the field names imported. Otherwise, it is a None object.
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if has_header:
            field_names = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # print("row = %r" % (row,))
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    print(field_names)
    return data_as_array, field_names


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        main()  # calls the main function with no arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(ifname=sys.argv[1])
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(ifname=sys.argv[1], delimiter=sys.argv[2])
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(",")))
        main(ifname=sys.argv[1], delimiter=sys.argv[2], columns=columns)