#include <stdio.h>
#include <math.h>
#include "util.h"
#include <string.h>
#include <time.h>

int mpirank, p;
int pN, d, C, N;

static int compare(const void *a, const void *b) {
    int *da = (int *) a;
    int *db = (int *) b;

    if (*da > *db)
        return 1;
    else if (*da < *db)
        return -1;
    else
        return 0;
}


double **alloc_2d_double(int rows, int cols) {
    int i;
    double *data = (double *) malloc(rows * cols * sizeof(double));
    double **array = (double **) malloc(rows * sizeof(double *));
    for (i = 0; i < rows; i++)
        array[i] = &(data[cols * i]);
    return array;
}


void printData(double **data, int row, int col) {
    int k, j;
    for (k = 0; k < row; k++) {
        for (j = 0; j < col; j++) {
            printf("%f ", data[k][j]);
        }
        printf("\n");
    }
}


void readDataPartial(char *file_name, double **data) {
    int start = mpirank * pN;
    int end = (mpirank + 1) * pN;
    FILE *fp = fopen(file_name, "r");
    const char s[1] = ",";
    char *token;
    int i;
    int line_count = 0;
    int read_count = 0;
    if (fp != NULL) {
        char line[100];
        while (fgets(line, sizeof line, fp) != NULL) {
            if (line_count >= start && line_count < end) {
                token = strtok(line, s);
                for (i = 0; i < d; i++) {
                    data[read_count][i] = atof(token);
                    if (i < d - 1) {
                        token = strtok(NULL, s);
                    }
                }
                read_count++;
            }
            line_count++;
        }
        fclose(fp);
    } else {
        perror(file_name);
    }
}

void getInitialClustersFromData(char *file_name, double **centers) {
    int indices[C];
    FILE *fp = fopen(file_name, "r");
    const char s[1] = ",";
    char *token;
    int i;
    int line_count = 0;
    int c_i = 0;
    for (i = 0; i < C; i++) {
        indices[i] = rand() % N;
    }
    qsort(indices, C, sizeof(int), compare);
    if (fp != NULL) {
        char line[100];
        while (fgets(line, sizeof line, fp) != NULL) {
            if (line_count == indices[c_i]) {
                printf("Center %d from line %d\n", c_i, line_count);
                token = strtok(line, s);
                for (i = 0; i < d; i++) {
                    centers[c_i][i] = atof(token);
                    if (i < d - 1) {
                        token = strtok(NULL, s);
                    }
                }
                c_i++;

                if (c_i == C) {
                    break;
                } else if (indices[c_i] == indices[c_i - 1]) {
                    indices[c_i] += 1;
                }

            }
            line_count++;
        }
        fclose(fp);
    } else {
        perror(file_name);
    }
}


int main(int argc, char *argv[]) {
    int i, j, k, l;
    char f_name[100];
    int N_ITER;

    if (argc != 6) {
        fprintf(stderr, "Usage: ./k_means dim N C data n_iter\n");
        fprintf(stderr, "dim: data dimension\n");
        fprintf(stderr, "N: # data points\n");
        fprintf(stderr, "C: number of clusters\n");
        fprintf(stderr, "data: data file path\n");
        fprintf(stderr, "n_iter: number of iterations\n");
    }

    //READ arguments
    sscanf(argv[1], "%d", &d);
    sscanf(argv[2], "%d", &N);
    sscanf(argv[3], "%d", &C);
    sscanf(argv[5], "%d", &N_ITER);

    if (((N % p) != 0)) {
        printf("N: %d\n", N);
        printf("p: %d\n", p);
        printf("Exiting. N must be a power of p. Accepting this is trivial, but not implemented/tested.\n");
    }
    pN = N / p;
    strcpy(f_name, argv[4]);

    double **centers;
    double **class_data_sum;
    int local_class_residual_count[C];
    int global_class_residual_count[C];
    centers = alloc_2d_double(C, d);
    class_data_sum = alloc_2d_double(C, d);

    getInitialClustersFromData(f_name, centers);
    double min_res, c_res;
    int min_i;


    double **data;
    data = alloc_2d_double(pN, d);
    readDataPartial(f_name, data);
    /* timing */
    clock_t start, end;
    start = clock();

    for (i = 0; i < N_ITER; i++) {
        //For each turn zero the sums

        for (k = 0; k < C; k++) {
            for (j = 0; j < d; j++) {
                class_data_sum[k][j] = 0;
            }
            local_class_residual_count[k] = 0;
        }

        //For each data point get the clusters cluster center and add the residual
        for (j = 0; j < pN; j++) {
            min_res = INFINITY;
            min_i = -1;

            for (k = 0; k < C; k++) {
                c_res = 0.0;

                //norm
                for (l = 0; l < d; l++) {
                    c_res += pow(abs(data[j][l] - centers[k][l]), 2);
                }
                c_res = sqrt(c_res);

                if (c_res < min_res) {
                    min_i = k;
                    min_res = c_res;
                }

            }

            for (l = 0; l < d; l++) {
                class_data_sum[min_i][l] += data[j][l];
            }
            local_class_residual_count[min_i] += 1;
        }

        for (k = 0; k < C; k++) {
            for (l = 0; l < d; l++) {
                centers[k][l] /= global_class_residual_count[k];
            }
        }
    }
    if (mpirank == 0) {
        printf("centers:\n");
        printData(centers, C, d);
    }

    free(data[0]);
    free(data);
    free(centers[0]);
    free(centers);
    end = clock();

    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", end - start);
    }
    return 0;
}
