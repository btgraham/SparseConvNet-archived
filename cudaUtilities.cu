#define CUDAUTILITIES_CU
#include "cudaUtilities.h"
#include <iostream>
#include <cassert>

static void cublasError(cublasStatus_t error, const char* file, int linenumber)
{
  switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
      break;

    case CUBLAS_STATUS_NOT_INITIALIZED:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_NOT_INITIALIZED\n";
      break;

    case CUBLAS_STATUS_ALLOC_FAILED:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_ALLOC_FAILED\n";
      break;

    case CUBLAS_STATUS_INVALID_VALUE:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_INVALID_VALUE\n";
      break;

    case CUBLAS_STATUS_ARCH_MISMATCH:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_ARCH_MISMATCH\n";
      break;

    case CUBLAS_STATUS_MAPPING_ERROR:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_MAPPING_ERROR\n";
      break;

    case CUBLAS_STATUS_EXECUTION_FAILED:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_EXECUTION_FAILED\n";
      break;

    case CUBLAS_STATUS_INTERNAL_ERROR:
      std::cout << file << " " << linenumber<<std::endl;
      std::cout <<  "CUBLAS_STATUS_INTERNAL_ERROR\n";
      break;
    }
}

int intRoundUp(int a, int d) {
  return ((a+d-1)/d)*d;
}
int intRound(int a, int d) {
  return round(a*1.0/d)*d;
}

int initializeGPU(int pciBusID) { //pciBusID, or -1 for the first device
  int nGPU;
  int deviceID=-1;
  cudaSafeCall(cudaGetDeviceCount(&nGPU));
  for (int i=0;i<nGPU;i++) {
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, i));
    if (i==0 and pciBusID==-1)
      pciBusID=prop.pciBusID;
    if (prop.pciBusID==pciBusID) {
      std::cout << "*";
      cudaSafeCall(cudaSetDevice(i));
      deviceID=i;
    } else {
      std::cout << " ";
    }
    std::cout << prop.pciBusID << " " << prop.name<< " " << (prop.totalGlobalMem>>20) << "MB Compute capability: " << prop.major << "." << prop.minor << std::endl;
  }
  assert(deviceID>=0);
  cublasError(cublasCreate(&cublasHandle),__FILE__,__LINE__);
  cnnMemStream = new cudaMemStream();
  return deviceID;
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. ///////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
                                    float* A, float* B, float* C,
                                    int l, int m, int r,
                                    float alpha, float beta, const char* file, int linenumber)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N,r,l,m,&alpha,B,r,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file, int linenumber)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T,r,l,m,&alpha,B,r,A,l,&beta,C,r), file, linenumber);
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file, int linenumber)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N,r,l,m,&alpha,B,m,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
                                      float* A, float* B, float* C,
                                      int l, int m, int r,
                                      float alpha, float beta, const char* file, int linenumber)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T,r,l,m,&alpha,B,m,A,l,&beta,C,r), file, linenumber);
}


cudaMemStream::cudaMemStream() : pinnedMemorySize(1<<24) {
  cudaSafeCall(cudaMallocHost(&pinnedMemory,pinnedMemorySize));
  cudaSafeCall(cudaStreamCreate(&stream));
}
cudaMemStream::~cudaMemStream() {
  cudaSafeCall(cudaStreamDestroy(stream));
  cudaFreeHost(pinnedMemory);
}
