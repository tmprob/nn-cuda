/**
 * @file main.cu
 * @brief Comprehensive example demonstrating neural network training and evaluation.
 * 
 * **Dataset**: Make-moons binary classification problem
 * **Architecture**: 2 → 1 → 2 (input → hidden → output)
 * **Training**: Supports both sequential (CPU) and CUDA (GPU) implementations. ONLY SEQUENTIAL VERSION RUNS!
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "../source/neuralnetwork.cuh"
#include "../source/data_handling.cuh"
#include "../source/activation.cuh"
#include "../source/evaluation.cuh"
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    // ========================================
    // CONFIGURATION AND SETUP
    // ========================================
    
    /**
     * Dataset and Network Configuration
     * - n_data_samples: Total number of data points in the dataset
     * - n_input_features: Dimensionality of input space (2D for make-moons)
     * - n_output_classes: Number of classification classes (binary classification)
     */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int n_data_samples, n_input_features, n_output_classes;
    n_data_samples = 1000;      // Total dataset size
    n_input_features = 2;       // 2D input space (x, y coordinates)
    n_output_classes = 2;       // Binary classification (2 classes)

    // ========================================
    // NEURAL NETWORK ARCHITECTURE DESIGN
    // ========================================
    
    /**
     * Network Architecture: 2 → 1 → 2
     * - Input layer: 2 neurons (for 2D input features)
     * - Hidden layer: 1 neuron (bottleneck for feature compression)
     * - Output layer: 2 neurons (for binary classification probabilities)
     * 
     * Using Identity activation functions for simplicity (linear transformations only)
     */
    NeuralNetworkBuilder *nnb = initNN();
    addLayer(nnb, n_input_features, 0);     // Input layer (no activation)
    addLayer(nnb, 1, 1);               // Hidden layer (1 neuron)
    addLayer(nnb, n_output_classes, 1); // Output layer (2 classes)
    
    // Build the complete neural network from the configuration
    NeuralNetwork *nn = buildNN(nnb);

    // ========================================
    // DATA LOADING AND PREPROCESSING
    // ========================================
    
    /**
     * Load Dataset from CSV Files
     * - input.csv: Contains 2D coordinates (x, y) for each data point
     * - output.csv: Contains one-hot encoded class labels [1,0] or [0,1]
     * 
     * File path construction uses realpath() to get absolute path for reliability
     */
    float **data_x, **data_y;
    char *path_prefix = realpath(".", NULL);  // Get current directory absolute path
    // strcat(path_prefix, "/.."); // Uncomment for different directory structures (e.g., when debugging)
    
    // Load input features and target labels from CSV files
    data_x = read_data_csv(path_prefix, "/data/input.csv", n_data_samples, n_input_features);
    data_y = read_data_csv(path_prefix, "/data/output.csv", n_data_samples, n_output_classes);

    // ========================================
    // TRAIN/TEST DATA SPLIT
    // ========================================
    
    /**
     * Create Training and Testing Sets
     * 
     * Split the dataset into training (80%) and testing (20%) portions:
     * - Training data: Used for learning network parameters
     * - Testing data: Used for unbiased performance evaluation
     * 
     * The split is randomized
     */
    float **test_data_x, **test_data_y;
    float training_split = 0.8;  // 80% for training, 20% for testing
    
    int n_training_data = train_test_split_percentage(data_x, data_y, &test_data_x, &test_data_y, 
                                                     n_data_samples, training_split);
    int n_test_data = n_data_samples - n_training_data;

    // ========================================
    // PRE-TRAINING ANALYSIS
    // ========================================
    
    /**
     * Examine Network State Before Training
     * 
     * Display initial random weights and make a sample prediction to establish
     * baseline performance. Helpful for:
     * - Check weight initialization
     * - Check network architecture
     * - Check data loading
     */
    printf("\nBefore training: \n");
    printf("Weights: \n");
    for (int i = 0; i < nn->n_weights; i++)
    {
        printf("%f ", nn->weights[i]);  // Show all initialized weights
    }
    printf("\n");

    // Make a prediction on sample #40 to see untrained network behavior
    float *prediction = single_predict(nn, data_x[40]);
    printf("Single prediction for sample #40: \n");
    for (int i = 0; i < nn->layer_sizes[nn->n_layer - 1]; i++)
    {
        printf("Label %d: Predicted=%f, Actual=%f\n", i, prediction[i], data_y[40][i]);
    }
    printf("\n");

    // ========================================
    // TRAINING CONFIGURATION AND EXECUTION
    // ========================================
    
    /**
     * Training Hyperparameters
     * 
     * These parameters control the learning process:
     * - learning_rate: Step size for gradient descent (0.2 = aggressive but stable)
     * - batchsize: Number of samples processed simultaneously (400 = large batches)
     * - n_epochs: Number of complete passes through the training data
     * - use_cuda: Toggle between CPU (false) and GPU (true) implementations
     */
    float learning_rate = 0.2;   // Learning rate for gradient descent
    int batchsize = 100000;         // Mini-batch size for training
    int n_epochs = 250;          // Number of training epochs
    bool use_cuda = true;      // Set to true for GPU acceleration (requires CUDA implementation)

    /**
     * Execute Training Process
     * 
     * Training mode is determined by use_cuda flag:
     * - false: Sequential CPU implementation (gradientdescend.cu)
     * - true: Parallel GPU implementation (gradientdescend_cuda.cu)
     */
    training(nn, data_x, data_y, test_data_x, test_data_y, learning_rate, 
             n_training_data, n_test_data, batchsize, n_epochs, use_cuda);

    // ========================================
    // POST-TRAINING ANALYSIS
    // ========================================
    
    /**
     * Examine Network State After Training
     * 
     * Compare the trained network with its initial state to verify learning:
     * - Weight values should have changed
     * - Predictions should be much closer
     * - Network should give improvement in accuracy
     */
    printf("\nAfter training: \n");
    printf("Weights: \n");
    for (int i = 0; i < nn->n_weights; i++)
    {
        printf("%f ", nn->weights[i]);  // Show trained weights
    }
    printf("\n");

    // Re-evaluate the same sample to see improvement
    free(prediction);  // Free previous prediction memory
    prediction = single_predict(nn, data_x[40]);
    printf("Single prediction for sample #40 after training: \n");
    for (int i = 0; i < nn->layer_sizes[nn->n_layer - 1]; i++)
    {
        printf("Label %d: Predicted=%f, Actual=%f\n", i, prediction[i], data_y[40][i]);
    }
    printf("\n");

    // ========================================
    // COMPREHENSIVE MODEL EVALUATION
    // ========================================
    printf("Evaluating model performance on test set (%d samples):\n", n_test_data);
    float **predictions = multiple_predict(nn, test_data_x, n_test_data);
    
    // Display detailed performance analysis including confusion matrix
    calc_conf_matrix(predictions, test_data_y, n_test_data);

    // ========================================
    // MEMORY CLEANUP AND PROGRAM TERMINATION
    // ========================================
    
    /**
     * Resource Deallocation
     * 
     * Properly free all dynamically allocated memory to prevent memory leaks:
     * - Neural network structures and weights
     * - Data arrays and predictions
     * - Builder objects and temporary arrays
     * 
     * Note: you do not have to care about handling the memory if you use the preimplemented structure of the NN
     */
    
    // Free prediction results
    for (int i = 0; i < n_test_data; i++)
    {
        free(predictions[i]);  // Free each prediction array
    }
    free(predictions);         // Free prediction pointer array
    free(prediction);          // Free single prediction

    // Free neural network structures
    destroy_NeuralNetworkBuilder(nnb);  // Clean up builder
    destroy_NeuralNetwork(nn);          // Clean up trained network

    // Free dataset arrays
    for (int i = 0; i < n_data_samples; i++)
    {
        free(data_x[i]);  // Free each input sample
        free(data_y[i]);  // Free each label sample
    }
    free(data_x);  // Free input data pointer array
    free(data_y);  // Free label data pointer array
    
    // Free path string
    free(path_prefix);

    printf("\nTraining and evaluation completed successfully!\n");
    printf("Neural network framework demonstration finished.\n");
    cudaDeviceSynchronize(); // Wait for all GPU work to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total GPU training time: %.3f ms\n", milliseconds);
    
    return 0;  // Successful program termination
}
