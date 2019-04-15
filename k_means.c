#include <stdio.h>
#include <math.h>
#include "util.h"
#include <string.h>
#include <mpi.h>

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
    //printf("Rank %d range, %d-%d\n",mpirank,start,end);
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
        // printf("C%d is %d\n",i,indices[i]);
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
    // 并行初始化函数 argc为变量数目，argv为变量数组，两个参数均来自main函数的参数
    MPI_Init(&argc, &argv);
    // 获得当前进程标识函数 comm为该进程所在的通信域句柄，rank为调用这一函数返回的进程在通信域中的标识号
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    // 获取通信域包含的进程总数函数 comm为通信域句柄，size为函数返回的通信域comm内包含的进程总数
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    // 获得本进程的机器名函数 name为返回的机器名字符串，resultlen为返回的机器名长度
    MPI_Get_processor_name(processor_name, &name_len);

    // 验证参数
    if (argc != 6 && mpirank == 0) {
        fprintf(stderr, "Usage: ./k_means dim N C data n_iter\n");
        fprintf(stderr, "dim: data dimension\n");
        fprintf(stderr, "N: # data points\n");
        fprintf(stderr, "C: number of clusters\n");
        fprintf(stderr, "data: data file path\n");
        fprintf(stderr, "n_iter: number of iterations\n");
        // 终止MPI执行环境
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    // READ arguments
    sscanf(argv[1], "%d", &d);
    sscanf(argv[2], "%d", &N);
    sscanf(argv[3], "%d", &C);
    sscanf(argv[5], "%d", &N_ITER);

    // 验证所给进程数是否可以均分数据
    if (((N % p) != 0) && mpirank == 0) {
        printf("N: %d\n", N);
        printf("p: %d\n", p);
        printf("Exiting. N must be a power of p. Accepting this is trivial, but not implemented/tested.\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    // 批数据大小 按顺序每个进程读取一个批次的数据
    pN = N / p;
    strcpy(f_name, argv[4]);

    // 进程0执行数据读取工作
    if (mpirank == 0) printf("%dd data,Total: %d points,%d points per process from file %s\n", d, N, pN, f_name);

    double **centers;
    double **class_data_sum;
    // 残差数组
    int local_class_residual_count[C];
    int global_class_residual_count[C];

    // 动态申请三维数组
    centers = alloc_2d_double(C, d);
    // 类别总和数组
    class_data_sum = alloc_2d_double(C, d);

    if (mpirank == 0) {
        // 读取所有数据并选定9个质心
        getInitialClustersFromData(f_name, centers);
    }

    // 广播函数 MPI_Bcast()，将某个进程的某个变量的值广播到该通信子中所有进程的同名变量中
    MPI_Bcast(&(centers[0][0]), C * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double min_res, c_res;
    int min_i;

    double **data;
    data = alloc_2d_double(pN, d);
    // 部分读取数据
    readDataPartial(f_name, data);
    /* timing */
    double start, end;
    // 表示阻止调用直到communicator中所有进程完成调用,进程将一直等待，直到其他并行进程执行结束
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

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

        //Share the sums of points and the countrs
        MPI_Allreduce(&(class_data_sum[0][0]), &(centers[0][0]), C * d, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(local_class_residual_count, global_class_residual_count, C, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);

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
    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", end - start);
    }

    // 并行结束函数
    MPI_Finalize();
    return 0;
}
