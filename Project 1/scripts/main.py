#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import tensorflow as tf
from BayesianDecision import DiscriminantAnalysis
from confusion_matrix import ConfusionMatrix

discriminant_analysis = DiscriminantAnalysis()
confusion_matrix = ConfusionMatrix()


def main():
    # Read and process the dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    discriminant_analysis.fit(x_train, y_train)

    yhat, (number_of_correct_predictions, number_of_incorrect_predictions, accuracy, error) = \
        discriminant_analysis.predict(x_test, y_test)

    confusion = confusion_matrix.calculate(y_test, yhat)
    confusion_matrix.plot('Confusion Matrix for MNIST Dataset tested using Discriminant Analysis',
                          confusion, confusion.shape[0])

    print('Number of Correct Predictions: ' + str(number_of_correct_predictions) + ' / ' +
          str(number_of_correct_predictions + number_of_incorrect_predictions))
    print('Number of Incorrect Predictions: ' + str(number_of_incorrect_predictions) + ' / ' +
          str(number_of_correct_predictions + number_of_incorrect_predictions))
    print('Accuracy: ' + str(accuracy) + '%' + '\t' + 'Error: ' + str(error))


if __name__ == '__main__':
    main()
