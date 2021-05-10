#ifndef __MATRIX_UTILS__
#define __MATRIX_UTILS__
#pragma once

void random_matrix_generate(size_t n, float* a, float abs_max = 1e1f);

void transpose_inplace(size_t n, float* a);

void transpose(size_t n, const float* a, float* a_t);

void multiply_by_def(size_t n, const float* a, const float* b, float* c);

void multiply_by_transpose(size_t n, const float* a, const float* b, float* c);

float deviation_norm(size_t n, const float* a, const float* b);

void matrix_print(size_t n, const float* a);


#endif // !__MATRIX_UTILS__