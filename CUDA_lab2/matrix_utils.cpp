#include "matrix_utils.h"

#include <random>
#include <iostream>

using namespace std;

void random_matrix_generate(size_t n, float* a, float abs_max)
{
    abs_max = fabs(abs_max);
    random_device rd_seed;
    mt19937 generator(rd_seed());
    uniform_real_distribution<float> const ur_distr(-abs_max, abs_max);
    size_t i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i * n + j] = ur_distr(generator);
        }
    }
}

void transpose_inplace(size_t n, float* a)
{
    size_t i, j;
    float tmp;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            tmp = a[i * n + j];
            a[i * n + j] = a[j * n + i];
            a[j * n + i] = tmp;
        }
    }
}

void transpose(size_t n, const float* a, float* a_t)
{
    size_t i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a_t[i * n + j] = a[j * n + i];
        }
    }
}

void multiply_by_def(size_t n, const float* a, const float* b, float* c)
{
    size_t i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            c[i * n + j] = 0.0f;
            for (k = 0; k < n; k++)
            {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void multiply_by_transpose(size_t n, const float* a, const float* b, float* c)
{
    size_t i, j, k;
    float* b_t = new float[n * n];
    transpose(n, b, b_t);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            c[i * n + j] = 0.0f;
            for (k = 0; k < n; k++)
            {
                c[i * n + j] += a[i * n + k] * b_t[j * n + k];
            }
        }
    }
    delete[] b_t;
}


float deviation_norm(size_t n, const float* a, const float* b)
{
    size_t i, j;
    float max_deviation = 0.0f;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            max_deviation = max(max_deviation, fabs(a[i * n + j] - b[i * n + j]));
        }
    }
    return max_deviation;
}

void matrix_print(size_t n, const float* a)
{
    size_t i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            cout << a[i * n + j] << " ";
        }
        cout << endl;
    }
}

