/**
 * @file utils.cuh
 *
 * @brief CPU-based mathematical utility functions for neural network computations.
 * 
 * @details This file provides sequential (CPU) implementations of core mathematical operations
 * used in neural network training and inference. These functions serve as reference implementations
 * and are used by the sequential training algorithm.
 * 
 * @note Similar GPU-optimized functions are available in kernels.cuh for parallel execution.
 * You hav to use CPU versions for:
 * - try the sequential training implementation
 * - debug and verificy your GPU results
 */
#ifndef UTILS_CUH
#define UTILS_CUH

#include "math.h"

/**
 * @brief Performs element-wise multiplication between two arrays (Hadamard product).
 * 
 * @details Computes the element-wise multiplication of two input arrays and stores the result
 * in an output array. This operation is fundamental in neural networks for combining
 * gradients with activation derivatives during backpropagation.
 * 
 * **Mathematical Operation:** output[i] = array_a[i] * array_b[i] for i = 0 to n-1
 * 
 * @param array_a Pointer to the first input array.
 * @param array_b Pointer to the second input array.
 * @param output Pointer to the output array where results are stored.
 * @param n Number of elements to process in each array.
 */
void elementwise_multiplication(float *array_a, float *array_b, float *output, int n);

/**
 * @brief Performs element-wise subtraction between two arrays.
 * 
 * @details Computes the element-wise difference between two input arrays, storing the result
 * in an output array. This operation is commonly used for computing prediction errors
 * and gradient calculations in neural network training.
 * 
 * **Mathematical Operation:** output[i] = array_a[i] - array_b[i] for i = 0 to n-1
 * 
 * @param array_a Pointer to the minuend array (values being subtracted from).
 * @param array_b Pointer to the subtrahend array (values being subtracted).
 * @param output Pointer to the output array where differences are stored.
 * @param n Number of elements to process in each array.
 */
void vector_substraction(float *array_a, float *array_b, float *output, int n);

/**
 * @brief Performs standard matrix-vector multiplication: output = matrix_a * array_b.
 * 
 * @details Computes the matrix-vector product where the matrix is stored in row-major format
 * as a flattened 1D array.
 * 
 * @param matrix_a Pointer to the matrix stored as a flattened 1D array (row-major).
 * @param array_b Pointer to the input vector to multiply.
 * @param output Pointer to the output vector (pre-allocated with n elements).
 * @param n Number of rows in the matrix (size of output vector).
 * @param m Number of columns in the matrix (size of input vector).
 */
void matrix_vector_multiplication(float *matrix_a, float *array_b, float *output, int n, int m);

/**
 * @brief Performs transposed matrix-vector multiplication with bias weight exclusion.
 * 
 * @details Computes the matrix-vector product where the matrix is implicitly transposed
 * during the computation.
 * 
 * **Attention:**
 * - **Implicit Transposition:** No actual matrix transposition; achieved through modified indexing
 * - **Bias Exclusion:** Skips the last column of the original matrix (bias weights)
 * 
 * **Indexing Pattern:**
 * - Original matrix: n rows × (m+1) columns (including bias column)
 * - Transposed view: m rows × n columns (excluding bias column)
 * - Access pattern skips bias weights in the last column
 * 
 * **Backpropagation Context:**
 * ```
 * Original matrix (n×(m+1)): [w11 w12 w13 b1]    Transposed view (m×n): [w11 w21 w31]
 *                            [w21 w22 w23 b2] →                        [w12 w22 w32]
 *                            [w31 w32 w33 b3]                          [w13 w23 w33]
 * ```
 * 
 * @param matrix_a Pointer to the weight matrix (row-major, includes bias column).
 * @param array_b Pointer to the error/gradient vector from the next layer.
 * @param output Pointer to the output vector for backpropagated errors.
 * @param n Number of rows in original matrix (size of input vector array_b).
 * @param m Number of columns in original matrix minus 1 (excludes bias, size of output).
 */
void transpose_matrix_vector_multiplication(float *matrix_a, float *array_b, float *output, int n, int m);

#endif // UTILS_CUH