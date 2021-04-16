#include "basic_types.h"
#include "tblis.h"
 
void dTensorTranspose( const int *perm, const int dim,
        const double alpha, const double *A, const int *sizeA, const int *outerSizeA,
        const double beta,        double *B,                   const int *outerSizeB,
        const int numThreads, const int useRowMajor);

void get_row_major_stride(long int *shape, long int *stride, int dim)
{
    long int acc = 1;
    long int d;
    for (int i = dim - 1; i >= 0; i--)
    {
        stride[i] = acc;
        d = shape[i];
        acc *= d;
    }
}

void tensor_contract(double *data_A, long int *shape_A, int *idx_A, int dim_A,
                     double *data_B, long int *shape_B, int *idx_B, int dim_B,
                     double *data_C, long int *shape_C, int *idx_C, int dim_C)
{
    //calculate strides
    long int *stride_A;
    stride_A = (long int *)malloc(dim_A * sizeof(long int));
    get_row_major_stride(shape_A, stride_A, dim_A);

    long int *stride_B;
    stride_B = (long int *)malloc(dim_B * sizeof(long int));
    get_row_major_stride(shape_B, stride_B, dim_B);

    long int *stride_C;
    stride_C = (long int *)malloc(dim_C * sizeof(long int));
    get_row_major_stride(shape_C, stride_C, dim_C);

    //Initialize Tensors
    tblis_tensor A;
    tblis_init_tensor_d(&A, dim_A, shape_A, data_A, stride_A);
    tblis_tensor B;
    tblis_init_tensor_d(&B, dim_B, shape_B, data_B, stride_B);
    tblis_tensor C;
    tblis_init_tensor_d(&C, dim_C, shape_C, data_C, stride_C);
    //Contract Tensors
    tblis_tensor_mult(NULL, NULL, &A, idx_A, &B, idx_B, &C, idx_C);
    free(stride_A);
    free(stride_B);
    free(stride_C);
}

void tensor_transpose(const int *perm, const int dim,
                      const double alpha, const double *A, const int *sizeA,
                      const double beta, double *B,
                      const int numThreads)
{
    dTensorTranspose(perm, dim,
                     alpha, A, sizeA, NULL,
                     beta, B, NULL,
                     numThreads, 1);
}

