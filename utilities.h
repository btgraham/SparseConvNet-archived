#pragma once
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <iostream>
// return vector 0,1,...,n-1
std::vector<int> range(int n);

// Integer powers
// http://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
// (sykora)
int ipow(int base, int exp);

// Calculate the number of points in a d-dimensional triangle/tetrahedro with
// given side length.
// Binomial coefficient \binom{linearSize+dimension-1}{dimension}
int triangleSize(int linearSize, int dimension);

// Assume test.size() is at least k.
template <typename t>
std::vector<int> vectorTopIndices(std::vector<t> &test, int k);

// Utilities to
//// check for CUDA errors,
//// use cublasSgemm for row-major matrix multiplication,
////  ...

// https://gist.github.com/ashwin/2652488#file-cudaerrorcheck-cu
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError()                                                       \
  { __cudaCheckError(__FILE__, __LINE__); }

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    std::cout << "cudaSafeCall() failed at " << file << ":" << line << " "
              << cudaGetErrorString(err) << "\n" << std::flush;
    exit(-1);
  }
#endif
  return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cout << "cudaCheckError() failed at " << file << ":" << line << " "
              << cudaGetErrorString(err) << "\n" << std::flush;
    exit(-1);
  }

// More careful checking. However, this will affect performance. Comment away
// if needed.
// err = cudaDeviceSynchronize();
// if (cudaSuccess != err) {
//   std::cout << "cudaCheckError() failed at " << file << ":" << line << " "
//             << cudaGetErrorString(err) << "\n" << std::flush;
//   exit(-1);
// }
#endif

  return;
}

void cublasError(cublasStatus_t error, const char *file = 0,
                 int linenumber = 0);

#define NTHREADS 512
#define KERNELBLOCKSIZE 32

int intRound(int a, int d);
int intRoundUp(int a, int d);

int initializeGPU(int pciBusID); // pciBusID, or -1 for the first device

//////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM for matrices in row major form.
// ///////////////////////////////////////////////////////////
// A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC(cublasHandle_t handle, float *A, float *B,
                                   float *C, int l, int m, int r, float alpha,
                                   float beta, const char *file = 0,
                                   int linenumber = 0);
// A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle_t handle, float *A, float *B,
                                    float *C, int l, int m, int r, float alpha,
                                    float beta, const char *file = 0,
                                    int linenumber = 0);
// A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC(cublasHandle_t handle, float *A, float *B,
                                    float *C, int l, int m, int r, float alpha,
                                    float beta, const char *file = 0,
                                    int linenumber = 0);
// A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC(cublasHandle_t handle, float *A, float *B,
                                     float *C, int l, int m, int r, float alpha,
                                     float beta, const char *file = 0,
                                     int linenumber = 0);
///////////////////////////////////////////////////////////////////////////////////////////////////

class cudaMemStream {
public:
  int pinnedMemorySize;
  void *pinnedMemory;
  cudaStream_t stream;
  cudaMemStream();
  ~cudaMemStream();
};
