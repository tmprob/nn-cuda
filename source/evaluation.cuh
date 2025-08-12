/**
 * @file evaluation.cuh
 *
 * @brief Functions for evaluating and analyzing neural network performance and predictions.
 * 
 */
#ifndef EVALUATION_CUH
#define EVALUATION_CUH

#include "neuralnetwork.cuh"

/**
 * @brief Generates a prediction for a single input sample using the neural network.
 * 
 * @details Performs a complete forward pass through the neural network for one input sample.
 * The function applies each layer's weights and activation functions sequentially to produce
 * the final output.
 * 
 * **Process:**
 * 1. Copy input data to working memory
 * 2. For each layer: apply weights → apply activation function
 * 3. Return final layer output
 * 
 * @param nn Pointer to the trained NeuralNetwork model.
 * @param x Pointer to the input feature vector.
 * @return Pointer to dynamically allocated array containing the predicted output values.
 */
float *single_predict(NeuralNetwork *nn, float *x);

/**
 * @brief Generates predictions for multiple input samples using the neural network.
 * 
 * @details Efficiently processes multiple samples by calling single_predict() for each input.
 * This function is useful for batch evaluation and testing on entire datasets.
 * 
 * @param nn Pointer to the trained NeuralNetwork model.
 * @param x 2D array of input samples (each row is one sample).
 * @param n_data_samples Number of input samples to process.
 * @return 2D array containing predicted outputs (not class labels) for all samples.
 */
float **multiple_predict(NeuralNetwork *nn, float **x, int n_data_samples);

/**
 * @brief Computes the Mean Squared Error (MSE) loss between predictions and expected outputs.
 * 
 * @details Calculates the average squared difference between predicted and actual outputs
 * across all samples and output classes.
 * 
 * @param predictions 2D array of model predictions.
 * @param expected 2D array of ground truth values.
 * @param n_data_samples Number of data samples.
 * @param n_output_classes Number of output classes/neurons.
 * @return The computed MSE loss value.
 */
float loss_mse(float **predictions, float **expected, int n_data_samples, int n_output_classes);

/**
 * @brief Calculates classification accuracy by comparing predicted labels with expected labels.
 * 
 * @details Converts continuous prediction outputs to discrete class labels using argmax,
 * then computes the fraction of correctly classified samples.
 * 
 * **Process:**
 * 1. Convert predictions to class labels using get_labels()
 * 2. Check if predicted label matches the true class (one-hot expected format)
 * 3. Calculate fraction of correct predictions
 *  
 * **Return Value:**
 * - Range: [0.0, 1.0] where 1.0 = 100% accuracy
 * - For binary classification: >0.5 indicates better than random guessing
 * 
 * @param predictions 2D array of model predictions (raw network outputs).
 * @param expected 2D array of one-hot encoded ground truth labels.
 * @param n_data_samples Number of data samples to evaluate.
 * @param n_output_classes Number of output classes.
 * @return Accuracy as a fraction between 0.0 and 1.0.
 */
float calc_accuracy(float **predictions, float **expected, int n_data_samples, int n_output_classes);

/**
 * @brief Calculates and displays a detailed confusion matrix for binary classification.
 * 
 * @details Generates a comprehensive performance analysis including confusion matrix,
 * accuracy, precision, recall, and F1-score.
 * 
 * **Output Includes:**
 * - 2x2 confusion matrix with True/False Positives/Negatives
 * - Accuracy: (TP + TN) / (TP + TN + FP + FN)
 * - Precision: TP / (TP + FP) - quality of positive predictions
 * - Recall: TP / (TP + FN) - completeness of positive predictions
 * - F1-Score: harmonic mean of precision and recall
 * 
 * @note This function is primarily for detailed analysis and reporting. Use calc_accuracy()
 * for simple accuracy computation during training.
 * 
 * @param predictions 2D array of model predictions.
 * @param expected 2D array of one-hot encoded ground truth labels.
 * @param n_data_samples Number of data samples to analyze.
 */
void calc_conf_matrix(float **predictions, float **expected, int n_data_samples);

/**
 * @brief Converts continuous prediction outputs to discrete class labels.
 * 
 * @details Performs argmax operation on each prediction vector to determine the most
 * likely class.
 * 
 * @note This function can be called from both CPU (__host__) and GPU (__device__) contexts.
 * @note The returned array must be freed by the caller.
 * 
 * @param predictions 2D array of raw network predictions.
 * @param n_data_samples Number of data samples.
 * @param n_output_classes Number of possible output classes.
 * @return Dynamically allocated array of predicted class labels (range: 0 to n_output_classes-1).
 */
__host__ __device__ int *get_labels(float **predictions, int n_data_samples, int n_output_classes);

#endif // EVALUATION_CUH