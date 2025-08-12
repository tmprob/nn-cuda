/**
 * @file activation.cuh
 *
 * @brief Activation functions, their derivatives, and supporting enumeration types.
 * 
 * @details This file provides both host (CPU) and device (GPU) implementations of activation
 * functions commonly used in neural networks. Functions marked with __host__ __device__ can
 * be called from both CPU and GPU code, while others are specific to their execution context.
 * 
 */
#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <stdio.h>
#include <math.h>

/**
 * @name Activation functions
 * @brief Applies the specific activation function or derivative element-wise on an array.
 * @note Can be called from CPU->CPU and GPU->GPU.
 *
 * @param x The input array.
 * @param n The size of the input array.
 */
///@{
__host__ __device__ void identity(float *x, int n);

__host__ __device__ void del_identity(float *x, int n);

///@}

/**
 * @brief Applies the specified activation function element-wise on an array.
 * @note Only callable from CPU->CPU. GPU function is in kernels.cuh
 *
 * @param current_output The input array.
 * @param n The size of the input array.
 * @param activation_func The activation function to apply.
 */
void run_activation(float *current_output, int n, int activation_func);

/**
 * @brief Computes the derivative of the specified activation function element-wise on an array.
 * @note Only callable from CPU->CPU. GPU function is in kernels.cuh
 *
 * @param current_output The input array.
 * @param n The size of the input array.
 * @param activation_func The activation function for which to compute the derivative.
 */
void run_del_activation(float *current_output, int n, int activation_func);

/**
 * @brief Prints the specified activation function.
 *
 * @param activation_func The activation function to print.
 */
void print_activation(int activation_func);

#endif // ACTIVATION_CUH
