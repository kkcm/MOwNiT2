#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

void write_result_to_file(FILE *file, int size, char* version, double time);

double substruct_measurement_time(struct timeval start, struct timeval end);

double **create_matrix(int size);

double **fill_matrix(double **D, int size);

void clear_matrix(double **D, int size);

void free_matrix(double **D, int size);

void fill_gsl_matrix(gsl_matrix *matrix, int size);

void naive_multiplication(double **A, double **B, double **result, int size);

void better_multiplication(double **A, double **B, double **result, int size);

double generate_random(double min, double max);

int main() {

    struct timeval start_measurement, end_measurement;

    int trials = 10;
    int iterations = 10;
    int step = 100;

    FILE* result_file = fopen("c_martix_multiplication_results.csv", "w+");
    fprintf(result_file, "size,version,time\n");

    for (int i = 0; i < trials; i++){
        for(int j = 0; j < iterations; j++){
            int size = (j + 1) * step;
            double** A = create_matrix(size);
            double** B = create_matrix(size);
            double** C = create_matrix(size);
            A = fill_matrix(A, size);
            B = fill_matrix(B, size);

            gettimeofday(&start_measurement, NULL);
            naive_multiplication(A, B, C, size);
            gettimeofday(&end_measurement, NULL);

            write_result_to_file(result_file, size, "naive", substruct_measurement_time(start_measurement, end_measurement));
            clear_matrix(C, size);

            gettimeofday(&start_measurement, NULL);
            better_multiplication(A, B, C, size);
            gettimeofday(&end_measurement, NULL);

            write_result_to_file(result_file, size, "better", substruct_measurement_time(start_measurement, end_measurement));

            free_matrix(A, size);
            free_matrix(B, size);
            free_matrix(C, size);
        }
    }

    for (int i = 0; i < trials; i++){
        for(int j = 0; j < iterations; j++){
            int size = (j + 1) * step;
            gsl_matrix* A = gsl_matrix_alloc(size,size);
            gsl_matrix* B = gsl_matrix_alloc(size,size);
            gsl_matrix* C = gsl_matrix_alloc(size,size);

            fill_gsl_matrix(A, size);
            fill_gsl_matrix(A, size);

            gettimeofday(&start_measurement, NULL);
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
            gettimeofday(&end_measurement, NULL);

            write_result_to_file(result_file, size, "gsl", substruct_measurement_time(start_measurement, end_measurement));

            gsl_matrix_free(C);
            gsl_matrix_free(A);
            gsl_matrix_free(B);
        }
    }
    return 0;
}


void better_multiplication(double **A, double **B, double **result, int size) {
    for (int i = 0; i < size; i++){
        for (int k = 0; k < size; k++){
            for(int j = 0; j < size; j++){
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void naive_multiplication(double **A, double **B, double **result, int size) {
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for(int k = 0; k < size; k++){
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void fill_gsl_matrix(gsl_matrix *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        double value = generate_random(1.0, 100.0);
        gsl_matrix_set(matrix, i / size, i % size, value);
    }
}

double **create_matrix(int size) {
    double **D = calloc((size_t)size, sizeof(double*));
    for(int i=0; i < size; i++){
        D[i] = calloc((size_t) size, sizeof(double));
    }
    return D;
}

double **fill_matrix(double **D, int size) {
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++) {
            D[i][j] = generate_random(1.0, 100.0);
        }
    }
    return D;
}

double generate_random(double min, double max) {
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void clear_matrix(double **D, int size) {
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++) {
            D[i][j] = 0.0;
        }
    }
}

void free_matrix(double **D, int size) {
    for(int i = 0; i < size; i++){
        free(D[i]);
    }
    free(D);
}

double substruct_measurement_time(struct timeval start, struct timeval end) {
    double start_sec = ((double) start.tv_sec)  + (((double) start.tv_usec) / 1000000);
    double end_sec = ((double) end.tv_sec)  + (((double) end.tv_usec) / 1000000);
    return end_sec - start_sec;
}

void write_result_to_file(FILE *file, int size, char* version, double time) {
    fprintf(file, "%d, %s, %lf\n", size, version, time);
}