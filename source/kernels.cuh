/**
 * @file kernels.cuh
 *
 * @brief CUDA kernel functions for various operations used in NeuralNetwork computations.
 * @note Assume: Due to implementation batch size has to be <= BLOCKSIZE_X*80, BLOCKSIZE_X<=32 and BLOCKSIZE_Y<=32. Due to Amandus specifications.\n
 * We implemented the possibility of parallelization over two dimensions for two kernels. In our test runs increasing BLOCKSIZE_Y was not conclusive towards improving performance.
 * Thus, we did not implement this for any other kernels, due to the limited time. It would take more time to make run time tests and searching for bottlenecks.
 *
 * This file contains CUDA kernels that implement core neural network operations. Most functions require implementation.
 * Each kernel operates on batches of data in parallel, with threads typically handling different data samples.
 * 
 * Concepts to understand:
 * - Thread indexing: Use blockIdx.x * blockDim.x + threadIdx.x to get unique thread ID
 * - Bounds checking: Always verify idx_data < batch_size before processing
 * - Memory layout: Weights are stored as flattened arrays, outputs as 2D arrays
 * - Synchronization: Use __syncthreads() to ensure all threads complete before continuing
 * 
 * General Neural Network Flow:
 * 1. Forward pass: kernel_calc_layer -> kernel_run_activation (repeat for each layer)
 * 2. Loss calculation: kernel_vector_substraction 
 * 3. Backward pass: kernel_run_del_activation -> kernel_elementwise_multiplication -> kernel_transpose_matrix_vector_multiplication
 * 4. Gradient calculation: kernel_calc_loss_weight_derivative -> kernel_reduce
 * 5. Weight update: kernel_update_weights
 */
#ifndef KERNELS_H
#define KERNELS_H

#include "math.h"

struct NeuralNetwork;
#include "neuralnetwork.cuh"
#include "activation.cuh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 1

/**
 * @brief CUDA kernel to calculate the forward pass of a single NeuralNetwork layer.
 *
 * @param output The output array for the current layer.
 * @param current_idx_input The current index of the input layer.
 * @param current_idx_output The current index of the output layer.
 * @param dev_nn The NeuralNetwork model on the device.
 * @param idx_weights The index of the weight matrix for the current layer.
 * @param n_input The size of the input layer.
 * @param n_output The size of the output layer.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_calc_layer(float **output, int current_idx_input, int current_idx_output, NeuralNetwork *dev_nn, int idx_weights, int n_input, int n_output, int batch_size);

/**
 * @brief CUDA kernel to apply activation function on the output of a layer.
 *
 * @param dev_output The output array for the current layer.
 * @param current_idx_output The current index of the output layer.
 * @param n The size of the layer.
 * @param activation_func Enum value defining the activation function to be applied.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_run_activation(float **dev_output, int current_idx_output, int n, int activation_func, int batch_size);

/**
 * @brief CUDA kernel to apply the derivative of the activation function.
 *
 * @param dev_output The output array for the current layer.
 * @param current_idx_output The current index of the output layer.
 * @param n The size of the layer.
 * @param activation_func Enum value defining the derived activation function to be applied.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_run_del_activation(float **dev_output, int current_idx_output, int n, int activation_func, int batch_size);

/**
 * @brief CUDA kernel to perform elementwise multiplication between two arrays.
 *
 * @param output The output array.
 * @param idx_output The index in the output array.
 * @param n The size of the array.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_elementwise_multiplication(float **output, int idx_output, int n, int batch_size);

/**
 * @brief CUDA kernel to subtract two vectors element-wise.
 *
 * @param output The output array.
 * @param idx_output The index in the output array.
 * @param ground_truth The ground truth vector.
 * @param n The size of the vectors.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_vector_substraction(float **output, int idx_output, float *ground_truth, int n, int batch_size);

/**
 * @brief CUDA kernel to implicitly perform transposed-matrix vector multiplication.
 *
 * @param output The output array.
 * @param idx_delta_k The index of the delta in the output array for the current layer.
 * @param idx_delta_k_plus_1 The index of the delta in the output array for the next layer.
 * @param dev_nn The NeuralNetwork model on the device.
 * @param idx_weights The index of the weight matrix for the current layer.
 * @param n Number rows of weight matrix (size of delta k).
 * @param m Number cols of weight matrix - 1 (size delta k+1).
 * @param batch_size The size of the batch.
 */
__global__ void kernel_transpose_matrix_vector_multiplication(float **output, int idx_delta_k, int idx_delta_k_plus_1, NeuralNetwork *dev_nn, int idx_weights, int n, int m, int batch_size);

/**
 * @brief CUDA kernel to calculate the derivative of the loss function with respect to the weights.
 * @note Implemented for 2D block sizes. Parallelization over data samples and neurons (deltas).
 *
 * @param output The output array.
 * @param idx_h The index of the hidden layer.
 * @param idx_delta The index of the delta array for the current layer.
 * @param dev_gradient The gradient array for the weights.
 * @param idx_del The index of the del array.
 * @param n_delta The size of the delta array.
 * @param n_input The size of the input layer.
 * @param n_weights The size of the weight matrix.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_calc_loss_weight_derivative(float **output, int idx_h, int idx_delta, float *dev_gradient, int idx_del, int n_delta, int n_input, int n_weights, int batch_size);

/**
 * @brief CUDA kernel to perform reduction operation to sum columns of a matrix.
 * @note Implemented for 2D block sizes. Parallelization over data samples and weights.
 *
 * @param dev_gradient The gradient matrix to be reduced (in memory as array).
 * @param n_weights The number of weights -> rows of matrix.
 * @param batch_size The size of the batch -> cols of matrix.
 */
__global__ void kernel_reduce(float *dev_gradient, int n_weights, int batch_size);

/**
 * @brief CUDA kernel to update the weights of the NeuralNetwork using the calculated gradients.
 *
 * @param dev_nn The NeuralNetwork model on the device.
 * @param dev_gradient The gradient array for the weights.
 * @param learning_rate The learning rate for the weight update.
 * @param batch_size The size of the batch.
 */
__global__ void kernel_update_weights(NeuralNetwork *dev_nn, float *dev_gradient, float learning_rate, int batch_size);

/**
 * @brief CUDA kernel to copy input data from device memory to device output memory for the X dimension.
 *
 * @param dev_output Pointer to the device output memory for the X dimension.
 * @param dev_batch_data_x Pointer to the device input memory for the X dimension.
 * @param n_input_features Number of neurons in input layer.
 * @param batch_size Number of batches.
 */
__global__ void kernel_copy_x(float **dev_output, float *dev_batch_data_x, int n_input_features, int batch_size);

/**
 * @brief CUDA kernel to calculate the mean squared error loss.
 * @note Not used. Intended to be used for printing the progress.
 *
 * @param predictions The prediction array.
 * @param expected The expected values array.
 * @param idx_output The index of the output layer.
 * @param n_data_samples The total number of data samples.
 * @param n_output_classes The number of output classes.
 * @param loss The calculated loss.
 */
__global__ void kernel_loss_mse(float **predictions, float *expected, int idx_output, int n_data_samples, int n_output_classes, float *loss);

/**
 * @brief CUDA kernel to calculate the accuracy of the predictions.
 * @note Not used. Intended to be used for printing the progress.
 *
 * @param predictions The prediction array.
 * @param expected The expected values array.
 * @param idx_output The index of the output layer.
 * @param n_data_samples The total number of data samples.
 * @param n_output_classes The number of output classes.
 * @param accuracy The calculated accuracy.
 */
__global__ void kernel_calc_accuracy(float **predictions, float *expected, int idx_output, int n_data_samples, int n_output_classes, float *accuracy);

#endif // KERNELS_H
