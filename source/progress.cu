#include "progress.cuh"

void print_progress(int current_batch, int total_n_batches, int n_signs)
{
    char progress_str[43] = "[........................................]";
    for (int i = 0; i < n_signs; i++)
    {
        progress_str[i + 1] = '='; // Exchange '.' with '=' to indicate progress
    }
    printf("\r%d/%d %s", current_batch, total_n_batches, progress_str);
}

void print_time(clock_t before, clock_t after, int total_n_batches)
{
    int msec = (after - before) * 1000 / CLOCKS_PER_SEC;

    printf(" - time: %d.%d sec", msec / 1000, msec % 1000);
    msec = msec / total_n_batches;
    printf(" - %d.%d sec/batch", msec / 1000, msec % 1000);
}

void print_evaluation(NeuralNetwork *nn, float **test_data_x, float **test_data_y, int n_test_data)
{
    // Evaluation of NeuralNetwork
    float **predictions = multiple_predict(nn, test_data_x, n_test_data);
    float acc = calc_accuracy(predictions, test_data_y, n_test_data, nn->layer_sizes[nn->n_layer - 1]);
    float loss = loss_mse(predictions, test_data_y, n_test_data, nn->layer_sizes[nn->n_layer - 1]);
    free(predictions);

    printf(" - loss: %f - accuracy: %f", loss, acc);
}