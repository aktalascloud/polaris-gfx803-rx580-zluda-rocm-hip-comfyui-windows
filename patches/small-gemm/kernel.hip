// SPDX-License-Identifier: GPL-3.0-only
extern "C" __attribute__((global)) void gfx803_small_sgemm(
    const float* a, const float* b, const float* c, float* d,
    int m, int n, int k, int lda, int ldb, int ldc, int ldd,
    long long stride_a, long long stride_b, long long stride_c, long long stride_d,
    int transpose_a, int transpose_b, float alpha, float beta)
{
    unsigned item = __builtin_amdgcn_workitem_id_x();
    int row = (int)(item & 7);
    int column = (int)(item >> 3);
    if (row >= m || column >= n) return;
    long long batch = __builtin_amdgcn_workgroup_id_x();
    float sum = 0.0f;
    if (alpha != 0.0f) {
        for (int inner = 0; inner < k; ++inner) {
            long long ai = batch * stride_a + (transpose_a ? inner + row * lda : row + inner * lda);
            long long bi = batch * stride_b + (transpose_b ? column + inner * ldb : inner + column * ldb);
            sum += a[ai] * b[bi];
        }
    }
    float previous = beta == 0.0f ? 0.0f : beta * c[batch * stride_c + row + column * ldc];
    d[batch * stride_d + row + column * ldd] = alpha * sum + previous;
}
