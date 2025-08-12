/**
 * @file progress.cuh
 *
 * @brief Functions for displaying progress and evaluation during the NeuralNetwork training process.
 * @note Combine print_progress/_time/_evaluation to get a full progress bar (See gradientdescend.cu).
 */
#ifndef PROGRESS_CUH
#define PROGRESS_CUH

#include "neuralnetwork.cuh"
#include "kernels.cuh"
#include "evaluation.cuh"

#include <stdio.h>
#include <time.h>

/**
 * @brief Prints the progress during the training process.
 *
 * @param current_batch The current batch number.
 * @param total_n_batches The total number of batches.
 * @param n_signs The number of signs to display for progress visualization.
 */
void print_progress(int current_batch, int total_n_batches, int n_signs);

/**
 * @brief Prints the time taken for a batch during the training process.
 *
 * @param before The clock time before starting gradientdescend of this batch.
 * @param after The clock time after gradientdescend of this batch.
 * @param total_n_batches The total number of batches.
 */
void print_time(clock_t before, clock_t after, int total_n_batches);

/**
 * @brief Prints evaluation metrics (loss, accuracy) for a NeuralNetwork.
 *
 * @param nn The NeuralNetwork model.
 * @param test_data_x The testing data input array.
 * @param test_data_y The testing data label array.
 * @param n_test_data The number of testing data samples.
 */
void print_evaluation(NeuralNetwork *nn, float **test_data_x, float **test_data_y, int n_test_data);

#endif // PROGRESS_CUH