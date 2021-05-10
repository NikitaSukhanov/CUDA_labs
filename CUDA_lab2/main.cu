#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix_utils.h"

#include <chrono>
#include <iostream>

typedef void (*multiplicator)(size_t, const float*, const float*, float*);

using namespace std;

const size_t BLOCK_SIZE = 64;

__global__ void multiplication_kernel_transp(size_t n, const float* a, const float* b_t, float* c)
{
    const size_t linear_index_c = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (linear_index_c >= n * n)
    {
        return;
    }
    const size_t j = linear_index_c % n;
    const size_t linear_index_a = linear_index_c - j;
    const size_t linear_index_b_t = j * n;
    float result = 0.0f;

    for (size_t k = 0; k < n; k++)
    {
        result += a[linear_index_a + k] * b_t[linear_index_b_t + k];
    }
    c[linear_index_c] = result;
}

__global__ void multiplication_kernel_def(size_t n, const float* a, const float* b_t, float* c)
{
    const size_t linear_index_c = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (linear_index_c >= n * n)
    {
        return;
    }
    const size_t j = linear_index_c % n;
    const size_t i = linear_index_c / n;
    float result = 0.0f;

    for (size_t k = 0; k < n; k++)
    {
        result += a[i * n + k] * b_t[k * n + j];
    }
    c[linear_index_c] = result;
}

void multiply_by_def_gpu(size_t n, const float* a, const float* b, float* c)
{
    const size_t n_elements = n * n;
    const size_t array_lgh = sizeof(float) * n_elements;
    float *a_gpu, *b_gpu, *c_gpu;

    cudaMalloc(&a_gpu, array_lgh);
    cudaMalloc(&b_gpu, array_lgh);
    cudaMalloc(&c_gpu, array_lgh);

    cudaMemcpy(a_gpu, a, array_lgh, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, array_lgh, cudaMemcpyHostToDevice);

    const dim3  grid_dim((unsigned)((n_elements - 1) / BLOCK_SIZE + 1), 1, 1);
    const dim3  block_dim((unsigned)BLOCK_SIZE, 1, 1);
    multiplication_kernel_def <<< grid_dim, block_dim >>> (n, a_gpu, b_gpu, c_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_gpu, array_lgh, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}

void multiply_by_transpose_gpu(size_t n, const float* a, const float* b, float* c)
{
    const size_t n_elements = n * n;
    const size_t array_lgh = sizeof(float) * n_elements;
    const size_t block_size = 64;
    float *a_gpu, *b_t_gpu, *c_gpu;
    float *b_t = new float[n_elements];
    transpose(n, b, b_t);

    cudaMalloc(&a_gpu, array_lgh);
    cudaMalloc(&b_t_gpu, array_lgh);
    cudaMalloc(&c_gpu, array_lgh);

    cudaMemcpy(a_gpu, a, array_lgh, cudaMemcpyHostToDevice);
    cudaMemcpy(b_t_gpu, b_t, array_lgh, cudaMemcpyHostToDevice);

    const dim3  grid_dim((unsigned)((n_elements - 1) / BLOCK_SIZE + 1), 1, 1);
    const dim3  block_dim((unsigned)BLOCK_SIZE, 1, 1);
    multiplication_kernel_transp <<< grid_dim, block_dim >>> (n, a_gpu, b_t_gpu, c_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_gpu, array_lgh, cudaMemcpyDeviceToHost);

    delete[] b_t;
    cudaFree(a_gpu);
    cudaFree(b_t_gpu);
    cudaFree(c_gpu);
}

double time_profile(multiplicator f, size_t n, const float* a, const float* b, float* c)
{
    auto start = chrono::high_resolution_clock::now();
    f(n, a, b, c);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> time = stop - start;
    return time.count();
}

void deviation_check(size_t n, const float* a, const float* b)
{
    float deviation = deviation_norm(n, a, b);
    if (deviation > 1e-2)
    {
        cout << " large deviation: " << deviation << " ";
    }
    return;
}

int main(void)
{
    size_t n = 10;
    float *a, *b, *c, *c1;
    for (size_t i = 0; i < 8; i++)
    {
        a = new float[n * n];
        b = new float[n * n];
        c = new float[n * n];
        c1 = new float[n * n];
        random_matrix_generate(n, a);
        random_matrix_generate(n, b);

        cout << "n = " << n << ":" << endl;

        cout << "By definition: ";
        double cpu_time_def = time_profile(multiply_by_def, n, a, b, c);
        cout << "CPU time = " << cpu_time_def;
        double gpu_time_def = time_profile(multiply_by_def_gpu, n, a, b, c1);
        deviation_check(n, c, c1);
        cout << ", GPU time = " << gpu_time_def << endl;

        cout << "By transpose:  ";
        double cpu_time_transp = time_profile(multiply_by_transpose, n, a, b, c1);
        deviation_check(n, c, c1);
        cout << "CPU time = " << cpu_time_transp;
        double gpu_time_transp = time_profile(multiply_by_transpose_gpu, n, a, b, c1);
        deviation_check(n, c, c1);
        cout << ", GPU time = " << gpu_time_transp << endl;
        cout << endl;

        delete[] a;
        delete[] b;
        delete[] c;
        delete[] c1;
        n *= 2;
    }
}


