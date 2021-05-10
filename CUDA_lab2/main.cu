#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix_utils.h"

#include <assert.h>
#include <chrono>
#include <iostream>

typedef void (*multiplicator)(size_t, const float*, const float*, float*);

using namespace std;

const size_t BLOCK_SIZE_1D = 64;
const size_t BLOCK_SIZE_2D = 32;

__global__ void multiplication_kernel_def(size_t n, const float* a, const float* b, float* c)
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
        result += a[i * n + k] * b[k * n + j];
    }
    c[linear_index_c] = result;
}

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

__global__ void multiplication_kernel_shared(size_t n, const float* a, const float* b, float* c)
{
    assert(blockDim.x == BLOCK_SIZE_2D);
    assert(blockDim.y == BLOCK_SIZE_2D);
    const size_t j = blockIdx.x * BLOCK_SIZE_2D + threadIdx.x;
    const size_t i = blockIdx.y * BLOCK_SIZE_2D + threadIdx.y;
    float result = 0.0f;
    
    for (size_t k = 0; k < n; k += BLOCK_SIZE_2D)
    {
        __shared__ float a_block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
        __shared__ float b_block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
        a_block[threadIdx.y][threadIdx.x] = 0.0f;
        b_block[threadIdx.y][threadIdx.x] = 0.0f;
        const size_t a_x = k + threadIdx.x;
        const size_t b_y = k + threadIdx.y;
        if (i < n && a_x < n)
        {
            a_block[threadIdx.y][threadIdx.x] = a[i * n + k + threadIdx.x];
        }
        if (j < n && b_y < n)
        {
            b_block[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + j];
        }
        __syncthreads();

        for (size_t l = 0; l < BLOCK_SIZE_2D; l++)
        {
            result += a_block[threadIdx.y][l] * b_block[l][threadIdx.x];
        }
        __syncthreads();
    }

    if (i < n && j < n)
    {
        c[i * n + j] = result;
    }
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

    const dim3  grid_dim((unsigned)((n_elements - 1) / BLOCK_SIZE_1D + 1), 1, 1);
    const dim3  block_dim((unsigned)BLOCK_SIZE_1D, 1, 1);
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
    float *a_gpu, *b_t_gpu, *c_gpu;
    float *b_t = new float[n_elements];
    transpose(n, b, b_t);

    cudaMalloc(&a_gpu, array_lgh);
    cudaMalloc(&b_t_gpu, array_lgh);
    cudaMalloc(&c_gpu, array_lgh);

    cudaMemcpy(a_gpu, a, array_lgh, cudaMemcpyHostToDevice);
    cudaMemcpy(b_t_gpu, b_t, array_lgh, cudaMemcpyHostToDevice);

    const dim3  grid_dim((unsigned)((n_elements - 1) / BLOCK_SIZE_1D + 1), 1, 1);
    const dim3  block_dim((unsigned)BLOCK_SIZE_1D, 1, 1);
    multiplication_kernel_transp <<< grid_dim, block_dim >>> (n, a_gpu, b_t_gpu, c_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_gpu, array_lgh, cudaMemcpyDeviceToHost);

    delete[] b_t;
    cudaFree(a_gpu);
    cudaFree(b_t_gpu);
    cudaFree(c_gpu);
}

void multiply_shared_gpu(size_t n, const float* a, const float* b, float* c)
{
    const size_t n_elements = n * n;
    const size_t array_lgh = sizeof(float) * n_elements;
    const size_t blocks_per_dim = (n - 1) / BLOCK_SIZE_2D + 1;
    float* a_gpu, * b_gpu, * c_gpu;

    cudaMalloc(&a_gpu, array_lgh);
    cudaMalloc(&b_gpu, array_lgh);
    cudaMalloc(&c_gpu, array_lgh);

    cudaMemcpy(a_gpu, a, array_lgh, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, array_lgh, cudaMemcpyHostToDevice);

    const dim3  grid_dim((unsigned)blocks_per_dim, (unsigned)blocks_per_dim, 1);
    const dim3  block_dim((unsigned)BLOCK_SIZE_2D, (unsigned)BLOCK_SIZE_2D, 1);
    multiplication_kernel_shared <<< grid_dim, block_dim >>> (n, a_gpu, b_gpu, c_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_gpu, array_lgh, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
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

void deviation_check(size_t n, const float* a, const float* b, float treshold = 1e-2f)
{
    float deviation = deviation_norm(n, a, b);
    if (deviation > treshold)
    {
        cout << " large deviation: " << deviation << " ";
    }
    return;
}

int main(void)
{
    size_t n = 10;
    float *a, *b, *c, *c1;
    for (size_t i = 0; i < 9; i++)
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
        double gpu_time_def = time_profile(multiply_by_def_gpu, n, a, b, c1);
        deviation_check(n, c, c1);
        cout << "GPU time = " << gpu_time_def;
        cout << ", CPU time = " << cpu_time_def << endl;

        //cout << "By transpose:  ";
        //double cpu_time_transp = time_profile(multiply_by_transpose, n, a, b, c1);
        //deviation_check(n, c, c1);
        //double gpu_time_transp = time_profile(multiply_by_transpose_gpu, n, a, b, c1);
        //deviation_check(n, c, c1);
        //cout << "GPU time = " << gpu_time_transp;
        //cout << ", CPU time = " << cpu_time_transp << endl;

        cout << "Shared memory: ";
        double gpu_time_shared = time_profile(multiply_shared_gpu, n, a, b, c1);
        deviation_check(n, c, c1);
        cout << "GPU time = " << gpu_time_shared << endl;
        cout << endl;

        delete[] a;
        delete[] b;
        delete[] c;
        delete[] c1;
        n *= 2;
    }
}


