#include "neuralnetwork.cuh"

NeuralNetworkBuilder *initNN()
{
    NeuralNetworkBuilder *nnb = (NeuralNetworkBuilder *)malloc(sizeof(NeuralNetworkBuilder));
    nnb->layer_sizes = (int *)malloc(0 * sizeof(int));
    nnb->activation_func = (int *)malloc(0 * sizeof(int));
    nnb->n_layer = 0;
    return nnb;
}

void addLayer(NeuralNetworkBuilder *nnb, int n_neurons, int activation_func)
{
    nnb->n_layer += 1;
    nnb->layer_sizes = (int *)realloc(nnb->layer_sizes, nnb->n_layer * sizeof(int));
    nnb->layer_sizes[nnb->n_layer - 1] = n_neurons;

    nnb->activation_func = (int *)realloc(nnb->activation_func, nnb->n_layer * sizeof(int));
    nnb->activation_func[nnb->n_layer - 1] = activation_func;
}

void destroy_NeuralNetworkBuilder(NeuralNetworkBuilder *nnb)
{
    free(nnb->layer_sizes);
    free(nnb->activation_func);
    free(nnb);
}

NeuralNetwork *buildNN(NeuralNetworkBuilder *nnb)
{
    // Check if NeuralNetwork is buildable
    if (nnb->n_layer < 2)
    {
        printf("Error: NeuralNetwork has not enough layers!\n");
        exit(0);
    }

    // Calculate number of weights and start index of each layer in weights matrix
    int n_weights = 0, i;
    int *idx_weights;
    idx_weights = (int *)malloc(nnb->n_layer * sizeof(int));
    idx_weights[0] = n_weights;
    for (int i = 1; i < nnb->n_layer; i++)
    {
        n_weights += (nnb->layer_sizes[i - 1] + 1) * nnb->layer_sizes[i];
        idx_weights[i] = n_weights;
    }

    // Initialize weights with random numbers
    float *weights;
    weights = (float *)malloc(n_weights * sizeof(float));
    for (i = 0; i < n_weights; i++)
    {
        weights[i] = randfrom(-1.0, 1.0);
    }

    // Copy layer_size
    int *layer_size;
    layer_size = (int *)malloc(nnb->n_layer * sizeof(int));
    for (i = 0; i < nnb->n_layer; i++)
    {
        layer_size[i] = nnb->layer_sizes[i];
    }

    // Copy activation_func
    int *activation_func;
    activation_func = (int *)malloc(nnb->n_layer * sizeof(int));
    for (i = 0; i < nnb->n_layer; i++)
    {
        activation_func[i] = nnb->activation_func[i];
    }

    // Calculate n_neurons
    int n_neurons = 0;
    for (i = 0; i < nnb->n_layer; i++)
    {
        n_neurons += nnb->layer_sizes[i];
    }

    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->activation_func = activation_func;
    nn->idx_weights = idx_weights;
    nn->layer_sizes = layer_size;
    nn->n_layer = nnb->n_layer;
    nn->n_weights = n_weights;
    nn->weights = weights;
    nn->n_neurons = n_neurons;
    return nn;
}

void model_summary(NeuralNetwork *nn)
{
    int n_weights;

    printf("\n           #Neurals   #Weights  Activation\n");
    printf("==========================================\n");
    printf("Layer %2d | %8d | %8d |  ", 1, nn->layer_sizes[0], 0);
    print_activation(nn->activation_func[0]);
    printf("\n");
    for (int i = 1; i < nn->n_layer; i++)
    {
        n_weights = nn->idx_weights[i] - nn->idx_weights[i - 1];
        printf("------------------------------------------\n");
        printf("Layer %2d | %8d | %8d |  ", i + 1, nn->layer_sizes[i], n_weights);
        print_activation(nn->activation_func[i]);
        printf("\n");
    }
    printf("==========================================\n");
    printf("Total    | %8d | %8d \n\n", nn->n_neurons, nn->n_weights);
}

void calc_layer(float *input, float *weights, float *output, int n_input, int n_output)
{
    for (int i = 0; i < n_output; i++)
    {
        // Bias-weight
        output[i] = weights[i * (n_input + 1) + n_input];
        for (int j = 0; j < n_input; j++)
        {
            output[i] += weights[i * (n_input + 1) + j] * input[j];
        }
    }
}

void training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos, bool use_cuda)
{
    if (use_cuda)
    {
        cuda_training(nn, training_data_x, training_data_y, test_data_x, test_data_y, learning_rate, n_samples, n_test_data, batch_size, n_epos);
    }
    else
    {
        sequential_training(nn, training_data_x, training_data_y, test_data_x, test_data_y, learning_rate, n_samples, n_test_data, batch_size, n_epos);
    }
}

void destroy_NeuralNetwork(NeuralNetwork *nn)
{
    free(nn->activation_func);
    free(nn->layer_sizes);
    free(nn->idx_weights);
    free(nn->weights);
    free(nn);
}
