#include "gradientdescend.cuh"

void calc_loss_weight_derivative(float *delta, float *input, float *gradient, int n_delta, int n_input)
{
    // Weights have shape (n_output x (n_input + 1)). Last column contains the bias-weights.
    for (int i = 0; i < n_delta; i++)
    {
        // Add bias-weight
        gradient[i * (n_input + 1) + n_input] = delta[i];
        for (int j = 0; j < n_input; j++)
        {
            gradient[i * (n_input + 1) + j] = delta[i] * input[j];
        }
    }
}

float *calc_gradient(NeuralNetwork *nn, float *input, float *ground_truth)
{
    float *current_weights, *output, *current_output, *current_input;
    int i, j, k;

    // Write input data x to the start of output
    // Idea: output = [x,z^1,h^1,z^2,...,h^N-1,z^N,h^N=o]
    output = (float *)malloc((nn->n_neurons * 2 - nn->layer_sizes[0]) * sizeof(float));
    for (i = 0; i < nn->layer_sizes[0]; i++)
    {
        output[i] = input[i];
    }

    // Forward pass with saving results in output
    current_input = output;
    current_output = &output[nn->layer_sizes[0]];

    for (i = 1; i < nn->n_layer; i++)
    {
        current_weights = &nn->weights[nn->idx_weights[i - 1]];
        calc_layer(current_input, current_weights, current_output, nn->layer_sizes[i - 1], nn->layer_sizes[i]);
        // Write values of z^i to memory of h^i
        for (j = 0; j < nn->layer_sizes[i]; j++)
        {
            current_output[j + nn->layer_sizes[i]] = current_output[j];
        }
        current_output = &current_output[nn->layer_sizes[i]];
        run_activation(current_output, nn->layer_sizes[i], nn->activation_func[i]);
        current_input = current_output;
        current_output = &current_output[nn->layer_sizes[i]];
    }
    current_input = NULL;

    // Array with values of delta starting from N to 2 [N, N-1,...]
    float *delta = (float *)calloc((nn->n_neurons - nn->layer_sizes[0]), sizeof(float));
    float *current_delta;
    // Array with same shape of weights
    float *gradient = (float *)calloc(nn->n_weights, sizeof(float));

    current_delta = &delta[nn->n_neurons - nn->layer_sizes[0] - nn->layer_sizes[nn->n_layer - 1]];

    // Backwards pass for output layer
    current_output -= nn->layer_sizes[nn->n_layer - 1];
    vector_substraction(current_output, ground_truth, current_delta, nn->layer_sizes[nn->n_layer - 1]);
    current_output -= nn->layer_sizes[nn->n_layer - 1];
    run_del_activation(current_output, nn->layer_sizes[nn->n_layer - 1], nn->activation_func[nn->n_layer - 1]);
    elementwise_multiplication(current_output, current_delta, current_delta, nn->layer_sizes[nn->n_layer - 1]);
    current_output -= nn->layer_sizes[nn->n_layer - 2];
    calc_loss_weight_derivative(current_delta, current_output, &gradient[nn->idx_weights[nn->n_layer - 2]], nn->layer_sizes[nn->n_layer - 1], nn->layer_sizes[nn->n_layer - 2]);

    // Backwards pass for remaining layers
    for (k = nn->n_layer - 2; k > 0; k--)
    {
        current_delta -= nn->layer_sizes[k];
        // (W^k+1)^T * delta^k+1
        transpose_matrix_vector_multiplication(&nn->weights[nn->idx_weights[k]], &current_delta[nn->layer_sizes[k]], current_delta, nn->layer_sizes[k + 1], nn->layer_sizes[k]);
        current_output -= nn->layer_sizes[k];
        // a'(z^k)
        run_del_activation(current_output, nn->layer_sizes[k], nn->activation_func[k]);
        // delta^k = a'(z^k) ° (W^k+1)^T * delta^k+1
        elementwise_multiplication(current_output, current_delta, current_delta, nn->layer_sizes[k]);
        current_output -= nn->layer_sizes[k - 1];
        // del_loss/del_weight_ij^k = d_i^k * h_j^k-1
        calc_loss_weight_derivative(current_delta, current_output, &gradient[nn->idx_weights[k - 1]], nn->layer_sizes[k], nn->layer_sizes[k - 1]);
    }

    current_weights = NULL;
    current_output = NULL;
    current_delta = NULL;
    free(delta);
    free(output);
    return gradient;
}

void update(NeuralNetwork *nn, float **batch_data_x, float **batch_data_y, float learning_rate, int batch_size)
{
    float *gradient, *temp;
    int i, j;
    gradient = (float *)calloc(nn->n_weights, sizeof(float));

    // Calculate all gradients of batch and sum up
    for (i = 0; i < batch_size; i++)
    {
        temp = calc_gradient(nn, batch_data_x[i], batch_data_y[i]);
        for (j = 0; j < nn->n_weights; j++)
        {
            gradient[j] += temp[j];
        }
        free(temp);
    }

    // Update weights with gradient
    for (j = 0; j < nn->n_weights; j++)
    {
        nn->weights[j] -= learning_rate * 2 * gradient[j] / batch_size;
        gradient[j] = 0;
    }
    free(gradient);
}

void sequential_training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos)
{
    int i, j, b, k;
    // initialize indices for uniform random shuffle of trainings data
    int *idx_samples = (int *)malloc(n_samples * sizeof(int));
    for (i = 0; i < n_samples; i++)
    {
        idx_samples[i] = i;
    }

    // Calculate variables concerning the batch
    int n_batches, n_samples_last_batch;
    n_batches = n_samples / batch_size;
    n_samples_last_batch = n_samples % batch_size;

    // Allocate batch data x/y as 2d Arrays
    float **batch_data_x, **batch_data_y;
    float *helper_2d_array_data_x, *helper_2d_array_data_y;
    batch_data_x = (float **)malloc(batch_size * sizeof(float *));
    batch_data_y = (float **)malloc(batch_size * sizeof(float *));

    helper_2d_array_data_x = (float *)malloc(batch_size * nn->layer_sizes[0] * sizeof(float));
    helper_2d_array_data_y = (float *)malloc(batch_size * nn->layer_sizes[nn->n_layer - 1] * sizeof(float));
    for (i = 0; i < batch_size; i++)
    {
        batch_data_x[i] = helper_2d_array_data_x + i * nn->layer_sizes[0];
        batch_data_y[i] = helper_2d_array_data_y + i * nn->layer_sizes[nn->n_layer - 1];
    }

    // Initialize variables and clock for printing the progress bar
    int print_n_total_batches = n_batches;
    int size_progress_bar = 40;
    clock_t before, after;
    if (n_samples_last_batch > 0)
    {
        print_n_total_batches++;
    }
    float print_next_update = 0;
    before = clock();

    for (k = 0; k < n_epos; k++)
    {
        // Progress bar print
        printf("Epoche %d/%d\n", k + 1, n_epos);
        print_progress(0, print_n_total_batches, 0);
        fflush(stdout);
        print_next_update = print_n_total_batches / size_progress_bar;

        // Implicit shuffle of data for new epos
        shuffle(idx_samples, n_samples);

        for (b = 0; b < n_batches; b++)
        {
            // Copy batch data from global data
            for (i = 0; i < batch_size; i++)
            {
                for (j = 0; j < nn->layer_sizes[0]; j++)
                {
                    batch_data_x[i][j] = training_data_x[idx_samples[b * batch_size + i]][j];
                }
                for (j = 0; j < nn->layer_sizes[nn->n_layer - 1]; j++)
                {
                    batch_data_y[i][j] = training_data_y[idx_samples[b * batch_size + i]][j];
                }
            }

            update(nn, batch_data_x, batch_data_y, learning_rate, batch_size);

            // Progress bar print
            if ((int)print_next_update <= (b + 1))
            {
                int n_signs = size_progress_bar * (b + 1) / print_n_total_batches;
                print_progress((b + 1), print_n_total_batches, n_signs);
                print_next_update += print_n_total_batches / size_progress_bar;
                fflush(stdout);
            }
        }

        // Calculate last batch
        if (n_samples_last_batch > 0)
        {
            // Copy batch data from global data
            for (i = 0; i < n_samples_last_batch; i++)
            {
                for (j = 0; j < nn->layer_sizes[0]; j++)
                {
                    batch_data_x[i][j] = training_data_x[idx_samples[n_batches * batch_size + i]][j];
                }
                for (j = 0; j < nn->layer_sizes[nn->n_layer - 1]; j++)
                {
                    batch_data_y[i][j] = training_data_y[idx_samples[n_batches * batch_size + i]][j];
                }
            }
            update(nn, batch_data_x, batch_data_y, learning_rate, n_samples_last_batch);
        }

        // Progress bar print
        print_progress(print_n_total_batches, print_n_total_batches, size_progress_bar);
        print_evaluation(nn, test_data_x, test_data_y, n_test_data);
        after = clock();
        print_time(before, after, print_n_total_batches);
        before = after;
        printf("\n");
        fflush(stdout);
    }

    free(helper_2d_array_data_x);
    free(helper_2d_array_data_y);
    free(batch_data_x);
    free(batch_data_y);
    free(idx_samples);
}
