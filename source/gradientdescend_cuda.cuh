/**
 * @file gradientdescend_cuda.cuh
 *
 * @brief Implementation of gradient descent using CUDA for NeuralNetwork training.
 * @note Assume: Due to implementation batch size has to be <= BLOCKSIZE_X*80, BLOCKSIZE_X<=32 and BLOCKSIZE_Y<=32. Due to Amandus specifications.
 */
#ifndef GRADIENTDESCEND_CUDA_CUH
#define GRADIENTDESCEND_CUDA_CUH

#include "neuralnetwork.cuh"
#include "progress.cuh"
#include "kernels.cuh"

#include <stdlib.h>
#include <time.h>

/**
 * @brief Calculates the gradients of the NeuralNetwork parameters using CUDA.
 * Uses kernel functions for parallisation. Look into kernels.cu for more information on specifics.
 *
 * @param dev_nn The NeuralNetwork model on the device.
 * @param nn The NeuralNetwork model on the host.
 * @param dev_batch_data_y The batch data labels on the device.
 * @param dev_gradients The loss weight derivatives on the device.
 * @param batchsize The size of the batch.
 * @param dev_output The device array to store the intermediate outputs.
 */
void cuda_calc_gradient(NeuralNetwork *dev_nn, NeuralNetwork *nn, float *dev_batch_data_y, float *dev_gradients, int batchsize, float **dev_output);

/**
 * @brief Updates the device NeuralNetwork weights using the calculated gradients and learning rate with CUDA.
 *
 * @param dev_nn The NeuralNetwork model on the device.
 * @param nn The NeuralNetwork model on the host.
 * @param dev_batch_data_y The batch data input on the device.
 * @param dev_batch_data_y The batch data labels on the device.
 * @param dev_gradients The loss weight derivatives on the device.
 * @param learning_rate The learning rate for the update.
 * @param batch_size The size of the batch.
 * @param dev_output The device array storing the intermediate outputs.
 */
void cuda_update(NeuralNetwork *dev_nn, NeuralNetwork *nn, float *dev_batch_data_x, float *dev_batch_data_y, float *dev_gradients, float learning_rate, int batch_size, float **dev_output);

/**
 * @brief Performs the training of the NeuralNetwork using CUDA.
 * Main function for the training with cuda.
 * Allocates most of the device memory needed.
 * Handles the epochs and batch iterations including shuffling of data.
 * If data not divisible in batches, last batch is computed with less data.
 * Calls cuda_update for each batch.
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
 * @param n_epos The number of training epochs.
 */
void cuda_training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos);

#endif // GRADIENTDESCEND_CUDA_H
