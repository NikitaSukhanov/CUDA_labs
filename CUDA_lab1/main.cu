#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matrix_utils.h"

#include <chrono>
#include <iostream>

using namespace std;

__global__ void multiplication_kernel(size_t n, const float* a, const float* b_t, float* c)
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

void multiply_gpu(size_t n, const float* a, const float* b, float* c, size_t block_size = 64)
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

    const dim3  grid_dim((unsigned)((n_elements - 1) / block_size + 1), 1, 1);
    const dim3  block_dim((unsigned)block_size, 1, 1);
    multiplication_kernel <<< grid_dim, block_dim >>> (n, a_gpu, b_t_gpu, c_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(c, c_gpu, array_lgh, cudaMemcpyDeviceToHost);

    delete[] b_t;
    cudaFree(a_gpu);
    cudaFree(b_t_gpu);
    cudaFree(c_gpu);
}

int main(void)
{
    size_t n = 10;
    float *a, *b, *c_cpu, *c_gpu;
    const float abs_max = 10.0f;
    for (size_t i = 0; i < 8; i++)
    {
        a = new float[n * n];
        b = new float[n * n];
        c_cpu = new float[n * n];
        c_gpu = new float[n * n];
        random_matrix_generate(n, a, abs_max);
        random_matrix_generate(n, b, abs_max);

        auto start = chrono::high_resolution_clock::now();
        multiply_simple(n, a, b, c_cpu);
        auto stop = chrono::high_resolution_clock::now();
        chrono::duration<double> cpu_time = stop - start;
       
        start = chrono::high_resolution_clock::now();
        multiply_gpu(n, a, b, c_gpu);
        stop = chrono::high_resolution_clock::now();
        chrono::duration<double> gpu_time = stop - start;

        cout << "n = " << n << " CPU time = " << cpu_time.count();
        cout << " GPU time = " << gpu_time.count() << endl;
        float deviation = deviation_norm(n, c_cpu, c_gpu);
        if (deviation / abs_max > 1e-2)
        {
            cout << "large deviation: " << deviation << endl;
        }

        delete[] a;
        delete[] b;
        delete[] c_cpu;
        delete[] c_gpu;
        n *= 2;
    }
}


