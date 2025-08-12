#include "data_handling.cuh"

float **read_data_csv(const char *path_prefix, const char *path_suffix, int n_data_samples, int n_values)
{
    // Open file using path_prefix + path_suffix. Used to get an absolute path (see examples).
    FILE *fptr;
    char *path = (char *)malloc((strlen(path_prefix) + strlen(path_suffix)) * sizeof(char));
    strcpy(path, path_prefix);
    strcat(path, path_suffix);
    fptr = fopen(path, "r");
    free(path);

    if (fptr == NULL)
    {
        printf("Can not open file\n");
        exit(0);
    }

    // Assume: Number of chars per line is <10000.
    char line[10000];
    char *sp;
    float **data;
    int row = 0;

    // Allocate data array
    data = (float **)malloc(n_data_samples * sizeof(float *));
    for (int i = 0; i < n_data_samples; i++)
    {
        data[i] = (float *)malloc(n_values * sizeof(float));
    }

    // Reading lines of file until end of file is reached.
    // Assume: Delimiter is ','.
    while (fgets(line, 10000, fptr) != NULL)
    {
        sp = strtok(line, ",");
        data[row][0] = atof(sp);

        for (int i = 1; i < n_values; i++)
        {
            sp = strtok(NULL, ",");
            data[row][i] = atof(sp);
        }
        row++;
    }
    fclose(fptr);

    return data;
}

float randfrom(float min, float max)
{
    float range = (max - min);
    float div = RAND_MAX / range;

    // Seeding dependent on time of day
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    return min + (rand() / div);
}

void shuffle(int *array, int n)
{
    int i, j, t;

    // Seeding dependent on time of day
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    for (i = 0; i < n - 1; i++)
    {
        j = i + rand() / (RAND_MAX / (n - i) + 1);
        t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

void shuffle_data(float **data_x, float **data_y, int n_data_samples)
{
    int i, j;
    float *temp;

    // Seeding dependent on time of day
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    for (i = 0; i < n_data_samples - 1; i++)
    {
        j = i + rand() / (RAND_MAX / (n_data_samples - i) + 1);
        temp = data_x[j];
        data_x[j] = data_x[i];
        data_x[i] = temp;

        temp = data_y[j];
        data_y[j] = data_y[i];
        data_y[i] = temp;
    }
}

void train_test_split(float **data_x, float **data_y, float ***test_data_x, float ***test_data_y, int n_data_samples, int n_training_data)
{
    shuffle_data(data_x, data_y, n_data_samples);

    // By overriding pointer to 2d array, they aren't lost when leaving the scope. Thus, float*** as parameter type for function.
    test_data_x[0] = &data_x[n_training_data];
    test_data_y[0] = &data_y[n_training_data];
}

int train_test_split_percentage(float **data_x, float **data_y, float ***test_data_x, float ***test_data_y, int n_data_samples, float training_split)
{
    int n_training_data;
    n_training_data = n_data_samples * training_split;

    train_test_split(data_x, data_y, test_data_x, test_data_y, n_data_samples, n_training_data);
    return n_training_data;
}
