/**
 * @file neuralnetwork.cuh
 *
 * @brief Declarations for NeuralNetworkBuilder and NeuralNetwork structures, along with related functions 
 * for building, training, and managing neural networks.
 * 
 * @details This file provides a complete neural network framework with both CPU and GPU implementations.
 * The builder pattern is used for network construction, and the framework supports variable network 
 * architectures with different activation functions per layer (you are ask to use only the preimplemented identity activation function, but you are free to add others if you want).
 * 
 * @note For usage examples and implementation patterns, see the files in ./examples/.
 */
#ifndef NEURALNETWORK_CUH
#define NEURALNETWORK_CUH

#include "activation.cuh"
#include "gradientdescend.cuh"
#include "gradientdescend_cuda.cuh"
#include "data_handling.cuh"

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * @brief Struct for building a NeuralNetwork.
 */
typedef struct NeuralNetworkBuilder
{
    int *layer_sizes;
    int n_layer;
    int *activation_func;
} NeuralNetworkBuilder;

/**
 * @brief Struct for NeuralNetwork.
 * Weights between two layers k -> k+1 are saved as a flattened matrix W^k, where w_ij corresponds to the weight from neuron j in layer k to neuron i in
 * layer k+1. The bias weights are included in the last column of that matrix, e.g. b_i=w_i(m+1). \n
 * Idx_weights[k] := first weight of weight matrix W^k where k=0, ..., n_layer-2. Idx_weights[n_layer-1] = n_weights. This is similar to the CRS matrix format. \n
 * Layer_sizes is saved as a vector with the size of each layer.
 */
typedef struct NeuralNetwork
{
    float *weights;
    int *idx_weights;
    int n_weights;
    int *layer_sizes;
    int n_layer;
    int n_neurons;
    int *activation_func;
} NeuralNetwork;

/**
 * @brief Initializes a NeuralNetworkBuilder.
 *
 * @return A pointer to the initialized NeuralNetworkBuilder.
 */
NeuralNetworkBuilder *initNN();

/**
 * @brief Adds a layer to the NeuralNetworkBuilder.
 *
 * @param nnb The NeuralNetworkBuilder to add the layer to.
 * @param n_neurons The number of neurons in the layer.
 * @param activation_func The activation function for the layer.
 */
void addLayer(NeuralNetworkBuilder *nnb, int n_neurons, int activation_func);

/**
 * @brief Destroys a NeuralNetworkBuilder and frees its resources.
 *
 * @param nnb The NeuralNetworkBuilder to destroy.
 */
void destroy_NeuralNetworkBuilder(NeuralNetworkBuilder *nnb);

/**
 * @brief Builds a NeuralNetwork based on the provided NeuralNetworkBuilder.
 * @note Checks if the NeuralNetwork is valid.
 *
 * @param nnb The NeuralNetworkBuilder containing the configuration of the NeuralNetwork.
 * @return A pointer to the built NeuralNetwork.
 */
NeuralNetwork *buildNN(NeuralNetworkBuilder *nnb);

/**
 * @brief Prints a summary of the NeuralNetwork.
 *
 * @param nn The NeuralNetwork to summarize.
 */
void model_summary(NeuralNetwork *nn);

/**
 * @brief Calculates the output of a layer in the NeuralNetwork.
 *
 * @param input The input to the layer.
 * @param weights The weights of the layer.
 * @param output The output of the layer.
 * @param n_input The number of inputs to the layer.
 * @param n_output The number of outputs from the layer.
 */
void calc_layer(float *input, float *weights, float *output, int n_input, int n_output);

/**
 * @brief Trains the NeuralNetwork using the provided training data.
 * @note CUDA is not using test data during training at the moment due to high data transfer to GPU (see cuda_print_evaluation()).
 *
 * @param nn The NeuralNetwork to train.
 * @param training_data_x The input data for training.
 * @param training_data_y The label data for training.
 * @param test_data_x The input data for testing.
 * @param test_data_y The label data for testing.
 * @param learning_rate The learning rate for training.
 * @param n_samples The number of training samples.
 * @param n_test_data The number of testing samples.
 * @param batch_size The batch size for training.
 * @param n_epos The number of epochs for training.
 * @param use_cuda Specifies whether to use CUDA for training.
 */
void training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos, bool use_cuda);

/**
 * @brief Destroys a NeuralNetwork and frees its resources.
 *
 * @param nn The NeuralNetwork to destroy.
 */
void destroy_NeuralNetwork(NeuralNetwork *nn);

#endif // NEURALNETWORK_CUH
