#include "evaluation.cuh"

float *single_predict(NeuralNetwork *nn, float *x)
{
    float *input;
    float *current_weights;
    float *output;

    // Copy data x into input
    input = (float *)malloc(nn->layer_sizes[0] * sizeof(float));
    for (int i = 0; i < nn->layer_sizes[0]; i++)
    {
        input[i] = x[i];
    }

    // Calculate output with activation for each layer
    for (int i = 1; i < nn->n_layer; i++)
    {
        output = (float *)malloc(nn->layer_sizes[i] * sizeof(float));
        current_weights = &nn->weights[nn->idx_weights[i - 1]];
        calc_layer(input, current_weights, output, nn->layer_sizes[i - 1], nn->layer_sizes[i]);
        run_activation(output, nn->layer_sizes[i], nn->activation_func[i]);

        // Define current output as new input
        free(input);
        input = output;
    }
    return output;
}

float **multiple_predict(NeuralNetwork *nn, float **x, int n_data_samples)
{
    float **predictions;

    // Call single_predict for each data sample
    predictions = (float **)malloc(n_data_samples * sizeof(float *));
    for (int i = 0; i < n_data_samples; i++)
    {
        predictions[i] = single_predict(nn, x[i]);
    }

    return predictions;
}

float loss_mse(float **predictions, float **expected, int n_data_samples, int n_output_classes)
{
    float loss = 0;

    for (int i = 0; i < n_data_samples; i++)
    {
        for (int j = 0; j < n_output_classes; j++)
            loss += pow(expected[i][j] - predictions[i][j], 2);
    }
    return loss / n_data_samples;
}

float calc_accuracy(float **predictions, float **expected, int n_data_samples, int n_output_classes)
{
    // Convert predictions into labels
    int *labels;
    labels = get_labels(predictions, n_data_samples, n_output_classes);

    // Calculate the accuracy
    float accuracy = 0;
    for (int i = 0; i < n_data_samples; i++)
    {
        accuracy += expected[i][labels[i]];
    }

    free(labels);
    return accuracy / n_data_samples;
}

void calc_conf_matrix(float **predictions, float **expected, int n_data_samples)
{
    // Define class 0 as positive, change to 1 if needed
    int class_positive = 0;
    // Convert predictions into labels
    int *labels;
    labels = get_labels(predictions, n_data_samples, 2);

    // Calculate the 4 types of predictions needed for a confusion matrix
    int true_positive = 0, false_positive = 0, true_negative = 0, false_negative = 0;
    for (int i = 0; i < n_data_samples; i++)
    {
        if (labels[i] == class_positive)
        {
            if (1 == expected[i][labels[i]])
            {
                true_positive++;
            }
            else
            {
                false_positive++;
            }
        }
        else
        {
            if (1 == expected[i][labels[i]])
            {
                true_negative++;
            }
            else
            {
                false_negative++;
            }
        }
    }

    printf("\nPerformance:\n");
    printf("====================================\n");
    printf("\nConfusion Matrix:\n");
    printf("==================\n");
    printf("              Actual Class\n");
    printf("           |  Positive  | Negative |\n");
    printf("====================================\n");
    printf("Predicted  |            |          |\n");
    printf("====================================\n");
    printf("Positive   |   %8d | %8d |\n", true_positive, false_positive);
    printf("------------------------------------\n");
    printf("Negative   |   %8d | %8d |\n", false_negative, true_negative);
    printf("====================================\n");

    // Additional metrics
    float accuracy = (float)(true_positive + true_negative) / n_data_samples;
    float precision = (float)true_positive / (true_positive + false_positive);
    float recall = (float)true_positive / (true_positive + false_negative);
    float f1Score = 2 * ((precision * recall) / (precision + recall));

    printf("\nAccuracy: %.2f\n", accuracy);
    printf("Precision: %.2f\n", precision);
    printf("Recall: %.2f\n", recall);
    printf("F1 Score: %.2f\n", f1Score);

    free(labels);
}

__host__ __device__ int *get_labels(float **predictions, int n_data_samples, int n_output_classes)
{
    int label, *labels;
    float max_value;
    labels = (int *)malloc(n_data_samples * sizeof(int));

    // Find indice with largest value for each data sample
    for (int i = 0; i < n_data_samples; i++)
    {
        max_value = predictions[i][0];
        label = 0;
        for (int j = 1; j < n_output_classes; j++)
            if (max_value < predictions[i][j])
            {
                max_value = predictions[i][j];
                label = j;
            }
        labels[i] = label;
    }
    return labels;
}
