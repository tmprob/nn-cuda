/**
 * @file gradientdescend.cuh
 *
 * @brief Functions for performing gradient descent on CPU/sequential.
 */
#ifndef GRADIENTDESCEND_CUH
#define GRADIENTDESCEND_CUH

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

struct NeuralNetwork;
#include "neuralnetwork.cuh"
#include "progress.cuh"
#include "utils.cuh"

/**
 * @brief Calculates the derivative of the loss with respect to the weights.
 *
 * @param delta The delta values from current layer.
 * @param output The output values of the forward.
 * @param gradient The Array to save the resulting derivatives in.
 * @param n_delta The size of the delta array/the current layer.
 * @param n_output The size of the output array/the layer before.
 */
void calc_loss_weight_derivative(float *delta, float *output, float *gradient, int n_delta, int n_output);

/**
 * @brief Calculates the gradient of the NeuralNetwork parameters.
 *
 * @param nn The NeuralNetwork model.
 * @param input The input data.
 * @param ground_truth The ground truth labels.
 * @return The calculated gradient.
 */
float *calc_gradient(NeuralNetwork *nn, float *input, float *ground_truth);

/**
 * @brief Updates the neural network parameters using the calculated gradients and learning rate.
 *
 * @param nn The neural network model.
 * @param batch_data_x The input data batch.
 * @param batch_data_y The label data batch.
 * @param learning_rate The learning rate for the update.
 * @param batch_size The size of the batch.
 */
void update(NeuralNetwork *nn, float **batch_data_x, float **batch_data_y, float learning_rate, int batch_size);

/**
 * @brief Performs sequential training of the NeuralNetwork.
 *
 * @param nn The NeuralNetwork model.
 * @param training_data_x The training data input array.
 * @param training_data_y The training data label array.
 * @param test_data_x The testing data input array.
 * @param test_data_y The testing data label array.
 * @param learning_rate The learning rate for the update.
 * @param n_samples The total number of training data samples.
 * @param n_test_data The number of testing data samples.
 * @param batch_size The size of the training batch.
 * @param n_epochs The number of training epochs.
 */
void sequential_training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos);

#endif // GRADIENTDESCEND_CUH
