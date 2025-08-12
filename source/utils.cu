#include "utils.cuh"

void elementwise_multiplication(float *array_a, float *array_b, float *output, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = array_a[i] * array_b[i];
    }
}

void vector_substraction(float *array_a, float *array_b, float *output, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = array_a[i] - array_b[i];
    }
}

void matrix_vector_multiplication(float *matrix_a, float *array_b, float *output, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = 0;
        for (int j = 0; j < m; j++)
        {
            output[i] += matrix_a[i * m + j] * array_b[j];
        }
    }
}

void transpose_matrix_vector_multiplication(float *matrix_a, float *array_b, float *output, int n, int m)
{
    // m=1, n=2 matrix 2x2 [1 2 3 4] => [1 3]: We skip the bias-weights
    // Implicit transposition of matrix
    for (int i = 0; i < m; i++)
    {
        output[i] = 0;
        for (int j = 0; j < n; j++)
        {
            output[i] += matrix_a[i + j * (m + 1)] * array_b[j];
        }
    }
}