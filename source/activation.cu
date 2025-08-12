#include "activation.cuh"

__host__ __device__ void identity(float *x, int n)
{
    return;
}

__host__ __device__ void del_identity(float *x, int n)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = 1;
    }
}

void run_activation(float *current_output, int n, int activation_func)
{
    if (activation_func == 1)
    {
        identity(current_output, n);
    }
    else
    {
        printf("Error: No activation function specified!!!\n");
    }
}

void run_del_activation(float *current_output, int n, int activation_func)
{
    if (activation_func == 1)
    {
        del_identity(current_output, n);
    }
    else
    {
        printf("Error: No activation function specified!!!\n");
    }
}

void print_activation(int activation_func)
{
    if (activation_func == 1)
    {
        printf("Identity");
    }
    else
    {
        printf("Error: No activation function specified!!!\n");
    }
}