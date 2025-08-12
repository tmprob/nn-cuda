#include "kernels.cuh"

__global__ void kernel_calc_layer(float **output, int current_idx_input, int current_idx_output, NeuralNetwork *dev_nn, int idx_weights, int n_input, int n_output, int batch_size)
{
    // TODO: Implement forward pass calculation for a neural network layer
    // 
    // Your implementation should:
    // 1. Calculate the thread's data sample index using blockIdx.x, blockDim.x, and threadIdx.x
    // 2. Check if the thread's data index is within the batch size (return early if not)
    // 3. For each output neuron (row):
    //    a. Initialize sum with the bias weight (last weight in the row)
    //    b. Compute dot product of weights and input activations
    //    c. Store the result in output[idx_data][current_idx_output + row]
    //    d. Also store a copy for the activation function at output[idx_data][current_idx_output + n_output + row]
    // 4. Use __syncthreads() at the end to synchronize threads
    //
    // Formula: output = W * input + bias
    // Weight indexing: weights[idx_weights + row * n_input + col]
    // Bias is at: weights[idx_weights + row * n_input + (n_input - 1)]

    // Step 1: Calculate thread's data sample index
    int idx_data = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 2: Check bounds
    if (idx_data >= batch_size){
        return;
    }

    // Pointer to the flattened output vector for the current data sample
    float *data_output = output[idx_data];

    // Step 3: Compute output for each output neuron
    for (int row = 0; row < n_output; ++row) {
        float sum = 0.0f;

        // Step 3b: Compute dot product of weights and inputs
        for (int col = 0; col < n_input; ++col) {
            float input_val = data_output[current_idx_input + col];
            float weight = dev_nn->weights[idx_weights + row * n_input + col];
            sum += weight * input_val;
        }

        // Step 3a: Add bias (which comes after all n_input weights for this row)
        float bias = dev_nn->weights[idx_weights + row * n_input + (n_input-1)];
        sum += bias;

        // Step 3c: Store raw output before activation
        data_output[current_idx_output + row] = sum;

        // Step 3d: Duplicate for activation kernel
        data_output[current_idx_output + n_output + row] = sum;
    }

    // Step 4: Synchronize threads
    __syncthreads();
}

__global__ void kernel_run_activation(float **dev_output, int current_idx_output, int n, int activation_func, int batch_size)
{
    // Calculate index of data sample to be handled by this thread
    int idx_data = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if data of this thread is in the batch
    if (idx_data >= batch_size){
        return;
    }

    identity(&dev_output[idx_data][current_idx_output], n);

    __syncthreads();
}

__global__ void kernel_run_del_activation(float **dev_output, int current_idx_output, int n, int activation_func, int batch_size)
{
    // TODO: Implement derivative of activation function
    // 
    // Your implementation should:
    // 1. Calculate the thread's data sample index
    // 2. Check if the thread's data index is within the batch size
    // 3. Apply the derivative of the activation function to n elements
    //    starting at dev_output[idx_data][current_idx_output]
    // 4. Use __syncthreads() at the end
    //

    // Step 1: 
    int idx_data=blockIdx.x*blockDim.x+threadIdx.x;
    // Step 2:
    if (idx_data >= batch_size){
        return;
    }
    del_identity(&dev_output[idx_data][current_idx_output], n);


    // 4. Synchronisation
    __syncthreads();
}


__global__ void kernel_elementwise_multiplication(float **output, int idx_output, int n, int batch_size)
{
    // TODO: Implement element-wise multiplication between two arrays
    //
    // Your implementation should:
    // 1. Calculate the thread's data sample index
    // 2. Check if the thread's data index is within the batch size
    // 3. For each element i from 0 to n-1:
    //    Multiply output[idx_data][idx_output + i] by output[idx_data][idx_output + n + i]
    // 4. Use __syncthreads() at the end
    //
    // This is used in backpropagation to multiply deltas with activation derivatives
    // Step 1: 
    int idx_data=blockIdx.x*blockDim.x+threadIdx.x;
    // Step 2:
    if (idx_data >= batch_size){
        return;
    }
    // Step 3: 
    for(int i=0;i<n;i++){
        output[idx_data][idx_output + i]=output[idx_data][idx_output + i]*output[idx_data][idx_output + n + i];
    }
    // Use syncthreads at the end:
    __syncthreads(); 

}

__global__ void kernel_vector_substraction(float **output, int idx_output, float *ground_truth, int n, int batch_size)
{
    // TODO: Implement element-wise vector subtraction
    //
    // Your implementation should:
    // 1. Calculate the thread's data sample index
    // 2. Check if the thread's data index is within the batch size
    // 3. For each element i from 0 to n-1:
    //    Subtract ground_truth[idx_data * n + i] from output[idx_data][idx_output + i]
    // 4. Use __syncthreads() at the end
    //
    // This computes the error: prediction - ground_truth
    int idx_data=blockIdx.x*blockDim.x+threadIdx.x;
    // Step 2:
    if (idx_data >= batch_size){
        return;
    }
    // Step 3: 
    for(int i=0;i<n;i++){
        output[idx_data][idx_output + i]= output[idx_data][idx_output + i] - ground_truth[idx_data * n + i];
    }

    // Step 4: use syncthreads at the end:
    __syncthreads();
}

__global__ void kernel_transpose_matrix_vector_multiplication(float **output, int idx_delta_k, int idx_delta_k_plus_1, NeuralNetwork *dev_nn, int idx_weights, int n, int m, int batch_size)
{
    // TODO: Implement transposed matrix-vector multiplication
    //
    // Your implementation should:
    // 1. Calculate the thread's data sample index using blockIdx.x and blockDim.x
    // 2. Calculate the row index using threadIdx.y  
    // 3. Check if the thread's data index is within the batch size
    // 4. For each row i from idx_row to m (stepping by blockDim.y):
    //    a. Initialize output[idx_data][idx_delta_k + i] to 0
    //    b. For each column j from 0 to n-1:
    //       Add weights[idx_weights + i + j * (m + 1)] * output[idx_data][idx_delta_k_plus_1 + j]
    // 5. Use __syncthreads() at the end
    //
    // Note: This performs W^T * delta, SKIPPING bias weights
    // The weight matrix is implicitly transposed during access (work with indices)
     // This computes the error: prediction - ground_truth
    int idx_data=blockIdx.x*blockDim.x+threadIdx.x;
    int idx_row=threadIdx.y;


    // Step 3:
    if (idx_data >= batch_size){
        return;
    }
    // Step 4:
    for(int i=idx_row; i<m; i+=blockDim.y){
        // Step 4b
        output[idx_data][idx_delta_k + i] = 0;
        for(int j=0; j<n; j++){
            // Step 4c
            output[idx_data][idx_delta_k + i] += dev_nn->weights[idx_weights + i + j * (m + 1)] * output[idx_data][idx_delta_k_plus_1 + j];
        }
        
    }

    // 5. Sync threads:
    __syncthreads();
}

__global__ void kernel_calc_loss_weight_derivative(float **output, int idx_h, int idx_delta, float *dev_gradient, int idx_del, int n_delta, int n_input, int n_weights, int batch_size)
{
    // Calculate index of data sample to be handled by this thread
    int idx_data = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate index of first row to calculate by this thread
    int idx_row = threadIdx.y;
    // Offset for gradient vector since the gradients of all data samples are saved in one array
    int offset_weights = idx_data * n_weights + idx_del;

    // Check if data of this thread is in the batch
    if (idx_data >= batch_size)
        return;

    for (int i = idx_row; i < n_delta; i += blockDim.y)
    {
        // Derivative with respect to bias-weight
        dev_gradient[offset_weights + (i * (n_input + 1) + n_input)] = output[idx_data][idx_delta + i];
        for (int j = 0; j < n_input; j++)
        {
            dev_gradient[offset_weights + i * (n_input + 1) + j] = output[idx_data][idx_delta + i] * output[idx_data][idx_h + j];
        }
    }

    __syncthreads();
}

__global__ void kernel_reduce(float *dev_gradient, int n_weights, int batch_size)
{
    // TODO: Implement reduction to sum gradients across batch samples
    //r
    // Your implementation should:
    // 1. Calculate the weight index using blockIdx.x, blockDim.x, and threadIdx.x
    // 2. Use a while loop to handle multiple weights per thread (stride by gridDim.x * blockDim.x)
    // 3. For each weight:
    //    a. Initialize result to 0
    //    b. Sum dev_gradient[idx + i * n_weights] for all batch samples i
    //    c. Store the sum back to dev_gradient[idx]
    // 4. Use __syncthreads() at the end
    //
    // This sums gradients from all batch samples for each weight
    // 1. Index of the actual weight
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 2. Loop over all weights
    while (idx < n_weights)
    {
    // 3. For each weight:
        //a) Initialize result to 0:
        float sum = 0.0f;

        //b) Sum
        for (int i = 0; i < batch_size; i++)
        {
            sum += dev_gradient[i * n_weights + idx];
        }

        //c) store:
        dev_gradient[idx] = sum;

        idx += stride;  // Further to the next weight

    }
    
    // 4. Synchronise of all threads
    __syncthreads();

}

__global__ void kernel_update_weights(NeuralNetwork *dev_nn, float *dev_gradient, float learning_rate, int batch_size)
{
    // TODO: Implement weight update using gradients
    //
    // Your implementation should:
    // 1. Calculate the weight index using blockIdx.x, blockDim.x, and threadIdx.x  
    // 2. Use a while loop to handle multiple weights per thread (stride by gridDim.x * blockDim.x)
    // 3. For each weight, update using gradient descent:
    //    dev_nn->weights[idx] -= learning_rate * 2 * dev_gradient[idx] / batch_size
    // 4. Use __syncthreads() at the end
    //
    // The factor of 2 comes from the derivative of the MSE loss function
    // 1. Index of the weight
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Stride for the network
    int stride = blockDim.x * gridDim.x;

    // 3. Loop over all weights
    while (idx < dev_nn->n_weights) {
        // 3a. Compute the update
        dev_nn->weights[idx] -= learning_rate * 2.0f * dev_gradient[idx] / batch_size;

        // Further to the weight
        idx += stride;
        
    }
    // 4. Synchronise of all threads.
    __syncthreads();
   
}

__global__ void kernel_copy_x(float **dev_output, float *dev_batch_data_x, int n_input_features, int batch_size)
{
    // TODO: Implement copying input data to output array
    //
    // Your implementation should:
    // 1. Calculate the thread's data sample index
    // 2. Check if the thread's data index is within the batch size
    // 3. Copy n_input_features elements from dev_batch_data_x to dev_output:
    //    dev_output[idx_data][j] = dev_batch_data_x[idx_data * n_input_features + j]
    // 4. Use __syncthreads() at the end
    //
    // This initializes the input layer of the neural network
        // 1. Compute the index
    int idx_data = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Check if the thread´s data sample is correct
    if (idx_data >= batch_size)
        return;

    // 3. Ponter on the row
    float *output_row = dev_output[idx_data];

    //Copy everything
    for (int j = 0; j < n_input_features; ++j) {
        output_row[j] = dev_batch_data_x[idx_data * n_input_features + j];
    }

    // 4. Synchronisiere Threads
    __syncthreads();
}

__global__ void kernel_loss_mse(float **predictions, float *expected, int idx_output, int n_data_samples, int n_output_classes, float *loss)
{
    // TODO: Implement Mean Squared Error loss calculation
    //
    // Your implementation should:
    // 1. Initialize loss[0] to 0
    // 2. For each data sample i from 0 to n_data_samples-1:
    //    For each output class j from 0 to n_output_classes-1:
    //       Add pow(expected[i * n_output_classes + j] - predictions[i][idx_output + j], 2) to loss[0]
    // 3. Use __syncthreads() at the end
    //
    // Note: This function is intended for monitoring training progress
    // 1. Initialize loss[0] to 0.
    loss[0] = 0; 
    // 2. Enter the for-loop.
    for(int i=0;i<n_data_samples;i++){
        for(int j=0;j<n_output_classes;j++){
            loss[0]+=pow(expected[i*n_output_classes + j]-predictions[i][idx_output + j], 2);
        }
    
    }
    // 3. Sync threads:
    __syncthreads();
}

__device__ int get_predicted_label(float *prediction, int n_output_classes) {
    int max_idx = 0;
    float max_val = prediction[0];

    for (int i = 1; i < n_output_classes; i++) {
        if (prediction[i] > max_val) {
            max_val = prediction[i];
            max_idx = i;
        }
    }
    return max_idx;
}

__global__ void kernel_calc_accuracy(float **predictions, float *expected, int idx_output, int n_data_samples, int n_output_classes, float *accuracy)
{
    // TODO: Implement accuracy calculation
    //
    // Your implementation should:
    // 1. Convert predictions to labels (find index of maximum value for each sample)
    // 2. Calculate accuracy by comparing predicted labels with expected labels
    // 3. Store result in accuracy[1]
    // 4. Use __syncthreads() at the end
    //
    // You'll need to implement a helper function get_labels() or equivalent logic
    // Note: This function is intended for monitoring training progress
    
    int idx_data = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_data >= n_data_samples)
        return;

    // Get the prediction 
    float *prediction = &predictions[idx_data][idx_output];

    // 2. Determine the class
    int predicted_label = get_predicted_label(prediction, n_output_classes);

    // 3. compare with the label
    int true_label = (int)expected[idx_data];

        // increase if correct
    if (predicted_label == true_label) {
        atomicAdd(&accuracy[1], 1.0f);
    }
    accuracy[1]=accuracy[1]/n_data_samples;

    // 4. use snycthreads at the end:
    __syncthreads();

}