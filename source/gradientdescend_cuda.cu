#include "gradientdescend_cuda.cuh"

void cuda_calc_gradient(NeuralNetwork *dev_nn, NeuralNetwork *nn, float *dev_batch_data_y, float *dev_gradients, int batchsize, float **dev_output)
{
    int i, k;
    // Configure 1D blocks on GPU with BLOCKSIZE_X threads each
    int nblocks = (batchsize + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    dim3 gridDim(nblocks);
    dim3 blockDim(BLOCKSIZE_X);
    dim3 blockDim2d(BLOCKSIZE_X, BLOCKSIZE_Y);
    int batch_size = batchsize;


    // Forward pass with saving results in dev_output
    // Memory layout: dev_output[idx_data] = [x, z^1, h^1, z^2, ..., h^(N-1), z^N, h^N=output]
  
    int current_idx_output, current_idx_input;
    current_idx_input = 0;                   // Pointer to input of current layer (e.g., for layer k -> h^(k-1))
    current_idx_output = nn->layer_sizes[0]; // Pointer for calculations, starts at z^1

    for (i = 1; i < nn->n_layer; i++)
    {
        // TODO: Implement forward pass for layer i
        // You should complete the following kernel calls:
        // 1. Calculate layer output: z^i = W^i * h^(i-1) + b^i
        // 2. Apply activation function: h^i = activation(z^i)
        // 3. Update pointers for next layer
        //Step 1:
        kernel_calc_layer<<<gridDim,blockDim>>>(dev_output, current_idx_input, current_idx_output, dev_nn, nn->idx_weights[i-1], nn->layer_sizes[i-1]+1, nn->layer_sizes[i], batch_size);
        // Step 2: 
        kernel_run_activation<<<gridDim,blockDim>>>(dev_output, current_idx_output+nn->layer_sizes[i], nn->layer_sizes[i], nn->activation_func[i], batch_size);
        // Step 3:
        
        current_idx_input = current_idx_output + nn->layer_sizes[i];
        current_idx_output = current_idx_input + nn->layer_sizes[i];
        


    }

    // Backward pass for output layer
    // Override dev_output with delta calculations. Deltas for layer k stored at z^k position

    // TODO: Implement backward pass for output layer
    // You should complete the following operations:
    // 1. Calculate prediction error: delta^N = (prediction - ground_truth)
 
    // Length of the output vector

    // Move to h^N=o
    current_idx_output-=nn->layer_sizes[nn->n_layer-1];


    kernel_vector_substraction<<<gridDim,blockDim>>>(dev_output, current_idx_output, dev_batch_data_y, nn->layer_sizes[nn->n_layer-1], batch_size);
    // 2. Apply activation derivative: delta^N = delta^N * activation'(z^N)
    // move to z^N: 
    current_idx_output-=nn->layer_sizes[nn->n_layer-1];
    kernel_run_del_activation<<<gridDim,blockDim>>>(dev_output, current_idx_output, nn->layer_sizes[nn->n_layer-1], 1, batch_size);

    kernel_elementwise_multiplication<<<gridDim,blockDim>>>(dev_output, current_idx_output, nn->layer_sizes[nn->n_layer-1], batch_size);
    

    //move to h^N-1
    current_idx_output-=nn->layer_sizes[nn->n_layer-2];
    // 3. Calculate weight gradients for output layer
    kernel_calc_loss_weight_derivative<<<gridDim,blockDim2d>>>(dev_output, current_idx_output, current_idx_output + nn->layer_sizes[nn->n_layer-2], dev_gradients, nn->idx_weights[nn->n_layer-2], nn->layer_sizes[nn->n_layer-1], nn->layer_sizes[nn->n_layer-2], nn->n_weights, batch_size);
    
    //int current_idx_z_k_p_1;

    for (k = nn->n_layer - 2; k > 0; k--)
    {
        
        // TODO: Implement backward pass for hidden layer k
        // You should complete the following operations:
        // 1. Backpropagate error: delta^k = (W^(k+1))^T * delta^(k+1)
        // 2. Apply activation derivative: delta^k = delta^k * activation'(z^k)
        // 3. Calculate weight gradients for layer k

        // output on h^k
        kernel_transpose_matrix_vector_multiplication<<<gridDim,blockDim2d>>>(dev_output, current_idx_output, current_idx_output+nn->layer_sizes[k], dev_nn, nn->idx_weights[k], nn->layer_sizes[k+1], nn->layer_sizes[k], batch_size);
        
        //move to z^k:
        current_idx_output-=nn->layer_sizes[k];
        
        kernel_run_del_activation<<<gridDim,blockDim>>>(dev_output, current_idx_output, nn->layer_sizes[k], 1, batch_size);
        kernel_elementwise_multiplication<<<gridDim,blockDim>>>(dev_output, current_idx_output, nn->layer_sizes[k], batch_size); 
        //move to h^k-1:
        current_idx_output-=nn->layer_sizes[k-1];
        kernel_calc_loss_weight_derivative<<<gridDim, blockDim2d>>>(dev_output, current_idx_output, current_idx_output+nn->layer_sizes[k-1], dev_gradients, nn->idx_weights[k-1], nn->layer_sizes[k], nn->layer_sizes[k - 1], nn->n_weights, batchsize);
        
        cudaDeviceSynchronize();
    }
}

void cuda_update(NeuralNetwork *dev_nn, NeuralNetwork *nn, float *dev_batch_data_x, float *dev_batch_data_y, float *dev_gradients, float learning_rate, int batch_size, float **dev_output)
{
    // Configure 1D blocks on GPU with BLOCKSIZE_X threads each
    int nblocks = (batch_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    dim3 gridDim(nblocks);
    dim3 blockDim(BLOCKSIZE_X);

    // Copy input data from dev_batch_data_x into dev_output structure
    kernel_copy_x<<<gridDim, blockDim>>>(dev_output, dev_batch_data_x, nn->layer_sizes[0], batch_size);
    cudaDeviceSynchronize();

    // Calculate gradients using forward and backward passes
    // TODO: Students should uncomment this line after implementing cuda_calc_gradient
    cuda_calc_gradient(dev_nn, nn, dev_batch_data_y, dev_gradients, batch_size, dev_output);
    
    cudaDeviceSynchronize();

    // Reconfigure grid for weight operations (limit to 60 blocks for efficiency)
    nblocks = (nn->n_weights + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    if (nblocks > 60)
    {
        nblocks = 60;
    }
    gridDim.x = nblocks;

    // Sum gradients across all batch samples (reduction operation)
    if (batch_size > 1)
    {
        kernel_reduce<<<gridDim, blockDim>>>(dev_gradients, nn->n_weights, batch_size);
        cudaDeviceSynchronize();
    }
    
    // Update neural network weights using computed gradients
    kernel_update_weights<<<gridDim, blockDim>>>(dev_nn, dev_gradients, learning_rate, batch_size);
    cudaDeviceSynchronize();
}

void cuda_training(NeuralNetwork *nn, float **training_data_x, float **training_data_y, float **test_data_x, float **test_data_y, float learning_rate, int n_samples, int n_test_data, int batch_size, int n_epos)
{
    int i, j, b, k;

    // Initialize sample indices for uniform random shuffle of training data
    int *idx_samples = (int *)malloc(n_samples * sizeof(int));
    for (i = 0; i < n_samples; i++)
    {
        idx_samples[i] = i;
    }

    // Calculate batch-related variables
    int n_batches, n_samples_last_batch;
    n_batches = n_samples / batch_size;
    n_samples_last_batch = n_samples % batch_size;

    // Allocate batch data on CPU
    // Input data shape: (batch_size * n_input_features)
    // Memory layout needed because GPU memory doesn't allow direct memcpy of 2D structure
    float *batch_data_x;
    batch_data_x = (float *)malloc(batch_size * nn->layer_sizes[0] * sizeof(float));
    
    // Output data shape: (batch_size * n_output_classes)
    // Contiguous memory layout allows efficient memcpy to GPU
    float *data_y_memory_allocator, **batch_data_y;
    batch_data_y = (float **)malloc(batch_size * sizeof(float *));
    data_y_memory_allocator = (float *)malloc(batch_size * nn->layer_sizes[nn->n_layer - 1] * sizeof(float));
    for (i = 0; i < batch_size; i++)
    {
        batch_data_y[i] = data_y_memory_allocator + i * nn->layer_sizes[nn->n_layer - 1];
    }

    // Allocate GPU memory for intermediate outputs
    // Shape: (batch_size * total_intermediate_values_per_sample)
    // Stores all layer outputs: [x, z^1, h^1, z^2, ..., h^(N-1), z^N, h^N]
    float **dev_output, *dev_output_memory_allocator, **row_ptr_dev_output;
    cudaMalloc((void **)&dev_output_memory_allocator, batch_size * (nn->n_neurons * 2 - nn->layer_sizes[0]) * sizeof(float));
    cudaMalloc((void **)&dev_output, batch_size * sizeof(float *));
    row_ptr_dev_output = (float **)malloc(batch_size * sizeof(float *));
    for (i = 0; i < batch_size; i++)
    {
        row_ptr_dev_output[i] = dev_output_memory_allocator + i * (nn->n_neurons * 2 - nn->layer_sizes[0]);
    }
    cudaMemcpy(dev_output, row_ptr_dev_output, batch_size * sizeof(float *), cudaMemcpyHostToDevice);

    // Allocate GPU memory for batch labels
    float *dev_batch_data_y;
    cudaMalloc((void **)&dev_batch_data_y, batch_size * nn->layer_sizes[nn->n_layer - 1] * sizeof(float));

    // Allocate GPU memory for batch input data
    float *dev_batch_data_x;
    cudaMalloc((void **)&dev_batch_data_x, batch_size * nn->layer_sizes[0] * sizeof(float));

    // Allocate GPU memory for gradients (includes all batch samples)
    float *dev_gradients;
    cudaMalloc((void **)&dev_gradients, batch_size * nn->n_weights * sizeof(float));

    // Allocate and initialize neural network structure on GPU
    NeuralNetwork *dev_nn;
    cudaMalloc((void **)&dev_nn, sizeof(NeuralNetwork));
    cudaMemcpy(dev_nn, nn, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);

    // Allocate and copy neural network arrays to GPU
    float *dev_weights;
    int *dev_idx_weights, *dev_layer_sizes;
    cudaMalloc((void **)&dev_weights, nn->n_weights * sizeof(float));
    cudaMalloc((void **)&dev_idx_weights, nn->n_layer * sizeof(int));
    cudaMalloc((void **)&dev_layer_sizes, nn->n_layer * sizeof(int));

    // Copy data and update device pointers
    cudaMemcpy(dev_weights, nn->weights, nn->n_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_nn->weights), &dev_weights, sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_idx_weights, nn->idx_weights, nn->n_layer * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_nn->idx_weights), &dev_idx_weights, sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_layer_sizes, nn->layer_sizes, nn->n_layer * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(dev_nn->layer_sizes), &dev_layer_sizes, sizeof(int *), cudaMemcpyHostToDevice);

    // Initialize progress tracking variables
    int print_n_total_batches = n_batches;
    int size_progress_bar = 40;
    clock_t before, after;
    if (n_samples_last_batch > 0)
    {
        print_n_total_batches++;
    }
    float print_next_update = 0;
    before = clock();

    // Training loop over epochs
    for (k = 0; k < n_epos; k++)
    {
        // Display epoch progress header
        printf("Epoche %d/%d\n", k + 1, n_epos);
        print_progress(0, print_n_total_batches, 0);
        print_next_update = print_n_total_batches / size_progress_bar;
        fflush(stdout);

        // Shuffle training data for current epoch
        shuffle(idx_samples, n_samples);

        // Process all complete batches
        for (b = 0; b < n_batches; b++)
        {
            // Copy batch data from CPU to GPU memory layout
            for (i = 0; i < batch_size; i++)
            {
                // Copy input features for sample i
                for (j = 0; j < nn->layer_sizes[0]; j++)
                {
                    batch_data_x[i * nn->layer_sizes[0] + j] = training_data_x[idx_samples[b * batch_size + i]][j];
                }

                // Copy output labels for sample i
                for (j = 0; j < nn->layer_sizes[nn->n_layer - 1]; j++)
                {
                    batch_data_y[i][j] = training_data_y[idx_samples[b * batch_size + i]][j];
                }
            }
            
            // Transfer batch data to GPU
            cudaMemcpy(dev_batch_data_x, batch_data_x, batch_size * nn->layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_batch_data_y, batch_data_y[0], batch_size * nn->layer_sizes[nn->n_layer - 1] * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            // Perform one training step (forward pass, backward pass, weight update)
            cuda_update(dev_nn, nn, dev_batch_data_x, dev_batch_data_y, dev_gradients, learning_rate, batch_size, dev_output);
            cudaDeviceSynchronize();

            // Update progress bar display
            if ((int)print_next_update <= (b + 1))
            {
                int n_signs = size_progress_bar * (b + 1) / print_n_total_batches;
                print_progress((b + 1), print_n_total_batches, n_signs);
                print_next_update += print_n_total_batches / size_progress_bar;
                fflush(stdout);
            }
        }

        // Process last incomplete batch if it exists
        if (n_samples_last_batch > 0)
        {
            // Copy remaining samples to batch
            for (i = 0; i < n_samples_last_batch; i++)
            {
                for (j = 0; j < nn->layer_sizes[0]; j++)
                {
                    batch_data_x[i * nn->layer_sizes[0] + j] = training_data_x[idx_samples[b * batch_size + i]][j];
                }

                for (j = 0; j < nn->layer_sizes[nn->n_layer - 1]; j++)
                {
                    batch_data_y[i][j] = training_data_y[idx_samples[b * batch_size + i]][j];
                }
            }
            
            // Transfer partial batch to GPU
            cudaMemcpy(dev_batch_data_x, batch_data_x, n_samples_last_batch * nn->layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_batch_data_y, batch_data_y[0], n_samples_last_batch * nn->layer_sizes[nn->n_layer - 1] * sizeof(float), cudaMemcpyHostToDevice);

            // Process partial batch
            cuda_update(dev_nn, nn, dev_batch_data_x, dev_batch_data_y, dev_gradients, learning_rate, n_samples_last_batch, dev_output);
        }

        // Complete progress bar and display timing information
        print_progress(print_n_total_batches, print_n_total_batches, size_progress_bar);
        after = clock();
        print_time(before, after, print_n_total_batches);
        before = after;
        printf("\n");
        fflush(stdout);
    }

    // Copy updated weights back from GPU to CPU
    float *weights;
    weights = (float *)malloc(nn->n_weights * sizeof(float));
    cudaMemcpy(weights, dev_weights, nn->n_weights * sizeof(float), cudaMemcpyDeviceToHost);
    free(nn->weights);
    nn->weights = weights;

    // Clean up GPU memory allocations
    cudaFree(dev_gradients);
    cudaFree(dev_output_memory_allocator);
    cudaFree(dev_batch_data_x);
    cudaFree(dev_batch_data_y);
    cudaFree(dev_weights);
    cudaFree(dev_idx_weights);
    cudaFree(dev_layer_sizes);
    cudaFree(dev_nn);
    
    // Clean up CPU memory allocations
    free(data_y_memory_allocator);
    free(idx_samples);
    free(row_ptr_dev_output);
    free(batch_data_x);
    free(batch_data_y);
}