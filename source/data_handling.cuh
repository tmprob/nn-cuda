/**
 * @file data_handling.cuh
 *
 * @brief Comprehensive data preprocessing and manipulation utilities.
 * 
 * @details This file provides essential functions for data loading, preprocessing, and dataset
 * management commonly needed in machine learning workflows. Functions handle CSV file I/O,
 * data shuffling, train/test splitting, and random number generation.
 * 
 * **Key Features:**
 * - CSV file parsing with configurable delimiters
 * - Multiple shuffling algorithms for data randomization
 * - Flexible train/test splitting with percentage or absolute counts
 * - Seeded random number generation for reproducible results
 */
#ifndef DATA_HANDLING_CUH
#define DATA_HANDLING_CUH

#include <sys/time.h>
#include <stdio.h>

/**
 * @brief Reads numerical data from a CSV file into a 2D array structure.
 * 
 * @details Parses CSV files with comma-separated values and converts them to floating-point
 * numbers. The function handles file path construction, memory allocation, and data parsing
 * in a single operation. This is optimized for machine learning datasets where all values
 * are numerical.
 * 
 * **File Format Requirements:**
 * - Comma-separated values (CSV format)
 * - All values must be convertible to float
 * - Maximum line length: 10,000 characters
 * - No header row expected (pure data)
 * 
 * @param path_prefix Directory path where the CSV file is located.
 * @param path_suffix Filename of the CSV file to read.
 * @param n_data_samples Expected number of rows (data samples) in the file.
 * @param n_values Expected number of columns (features) per row.
 * @return Pointer to 2D array containing the loaded data, or NULL on failure.
 */
float **read_data_csv(const char *path_prefix, const char *path_suffix, int n_data_samples, int n_values);

/**
 * @brief Generates a random floating-point number within a specified range.
 * 
 * @details Produces uniformly distributed random numbers for weight initialization and
 * other stochastic operations. Uses system time for seeding to ensure different sequences
 * across program runs.
 * 
 * **Typical Usage:**
 * - Weight initialization: randfrom(-1.0, 1.0)
 * - Dropout rates: randfrom(0.0, 1.0)
 * - Learning rate scheduling: randfrom(0.001, 0.1)
 * 
 * @note The quality of randomness depends on RAND_MAX. For cryptographic applications
 * or when n >> RAND_MAX, consider using more sophisticated random number generators.
 * 
 * @param min Lower bound of the random range (inclusive).
 * @param max Upper bound of the random range (exclusive).
 * @return Random floating-point value in the range [min, max).
 */
float randfrom(float min, float max);

/**
 * @brief Shuffles an integer array using the Fisher-Yates algorithm.
 * 
 * @details Performs in-place uniform shuffling of integer arrays, commonly used for
 * creating randomized indices for data samples. This enables random sampling without
 * moving the actual data arrays.
 * 
 * @param array Pointer to the integer array to shuffle (modified in-place).
 * @param n Number of elements in the array.
 */
void shuffle(int *array, int n);

/**
 * @brief Shuffles corresponding elements in two 2D arrays simultaneously.
 * 
 * @details Performs synchronized shuffling of input features and labels, maintaining
 * the correspondence between samples. This is essential for randomizing training data
 * while preserving the input-output relationships.
 * 
 * @param data_x 2D array of input features (each row is one sample).
 * @param data_y 2D array of corresponding labels/targets.
 * @param n_data_samples Number of samples (rows) in both arrays.
 */
void shuffle_data(float **data_x, float **data_y, int n_data_samples);

/**
 * @brief Splits datasets into training and testing portions using absolute counts.
 * 
 * @details Divides input datasets into training and testing subsets by reassigning array
 * pointers. The original data remains in place, but pointers are redirected to create
 * logical splits. Data is shuffled before splitting to ensure random distribution.
 * 
 * **Layout After the Split:**
 * ```
 * Original: [sample0, sample1, ..., sampleN-1]
 * Training: [sample0, sample1, ..., sample(n_training-1)]
 * Testing:  [sample(n_training), ..., sampleN-1]
 * ```
 * 
 * @param data_x Input feature array to split.
 * @param data_y Input label array to split.
 * @param test_data_x Pointer to pointer that will reference the test features.
 * @param test_data_y Pointer to pointer that will reference the test labels.
 * @param n_data_samples Total number of samples in the input arrays.
 * @param n_training_data Number of samples to allocate for training (remainder goes to testing).
 */
void train_test_split(float **data_x, float **data_y, float ***test_data_x, float ***test_data_y, int n_data_samples, int n_training_data);

/**
 * @brief Splits datasets into training and testing portions using percentage ratios.
 * 
 * @details Convenience wrapper around train_test_split() that calculates the absolute
 * number of training samples based on a percentage. This is more intuitive for common
 * splits like 80/20 or 70/30.
 * 
 * 
 * @param data_x Input feature array to split.
 * @param data_y Input label array to split.
 * @param test_data_x Pointer to pointer that will reference the test features.
 * @param test_data_y Pointer to pointer that will reference the test labels.
 * @param n_data_samples Total number of samples in the input arrays.
 * @param training_split Fraction of data to use for training (range: 0.0 to 1.0).
 * @return Number of samples allocated to the training set.
 */
int train_test_split_percentage(float **data_x, float **data_y, float ***test_data_x, float ***test_data_y, int n_data_samples, float training_split);

#endif // DATA_HANDLING_CUH