//Utilities to
//// check for CUDA errors,
//// use cublasSgemm for row-major matrix multiplication,
////  ...



// https://gist.github.com/ashwin/2652488#file-cudaerrorcheck-cu
// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError() { __cudaCheckError( __FILE__, __LINE__ ); }

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
#endif
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }

  // More careful checking. However, this will affect performance. Comment away if needed.
  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
#endif

  return;
}


static void cublasError(cublasStatus_t error,const char* file = 0, int linenumber = 0)
{
  switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
      break;

    case CUBLAS_STATUS_NOT_INITIALIZED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_NOT_INITIALIZED\n";
      break;

    case CUBLAS_STATUS_NOT_SUPPORTED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_NOT_SUPPORTED\n";
      break;

    case CUBLAS_STATUS_ALLOC_FAILED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_ALLOC_FAILED\n";
      break;

    case CUBLAS_STATUS_INVALID_VALUE:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_INVALID_VALUE\n";
      break;

    case CUBLAS_STATUS_ARCH_MISMATCH:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_ARCH_MISMATCH\n";
      break;

    case CUBLAS_STATUS_MAPPING_ERROR:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_MAPPING_ERROR\n";
      break;

    case CUBLAS_STATUS_EXECUTION_FAILED:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_EXECUTION_FAILED\n";
      break;

    case CUBLAS_STATUS_INTERNAL_ERROR:
      cout << file << " " << linenumber<<endl;
      cout <<  "CUBLAS_STATUS_INTERNAL_ERROR\n";
      break;
    }
}

const int NTHREADS=512;
cublasHandle_t cublasHandle;
#define KERNELBLOCKSIZE 32

int kernelBlockSizeRoundUp(int a) {
  return ((a+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE)*KERNELBLOCKSIZE;
}
int kernelBlockSizeRound(int a) {
  return round(a*1.0/KERNELBLOCKSIZE)*KERNELBLOCKSIZE;
}

void initializeGPU(int cudaDevice) { //pciBusID, or -1 for the first device
  int nGPU;
  bool setGPU=false;
  cudaSafeCall(cudaGetDeviceCount(&nGPU));
  for (int i=0;i<nGPU;i++) {
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, i));
    if (i==0 and cudaDevice==-1)
      cudaDevice=prop.pciBusID;
    if (prop.pciBusID==cudaDevice) {
      cout << "*";
      cudaSafeCall(cudaSetDevice(i));
      setGPU=true;
    } else {
      cout << " ";
    }
    cout << prop.pciBusID << " " << prop.name<< endl;
  }
  assert(setGPU);
  cublasError(cublasCreate(&cublasHandle),__FILE__,__LINE__);
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//GEMM for matrices in row major form. ///////////////////////////////////////////////////////////
//A is l*m, B is m*r, C is l*r. Set C to alpha A B + beta C.
void d_rowMajorSGEMM_alphaAB_betaC (cublasHandle_t handle,
                                    float* A, float* B, float* C,
                                    int l, int m, int r,
                                    float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N,r,l,m,&alpha,B,r,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtB_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_T,r,l,m,&alpha,B,r,A,l,&beta,C,r), file, linenumber);
}
//A is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaABt_betaC (cublasHandle_t handle,
                                     float* A, float* B, float* C,
                                     int l, int m, int r,
                                     float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_N,r,l,m,&alpha,B,m,A,m,&beta,C,r), file, linenumber);
}
//A^t is l*m, B^t is m*r, C is l*r
void d_rowMajorSGEMM_alphaAtBt_betaC (cublasHandle_t handle,
                                      float* A, float* B, float* C,
                                      int l, int m, int r,
                                      float alpha, float beta, const char* file = 0, int linenumber = 0)
{
  cublasError(cublasSgemm (handle, CUBLAS_OP_T, CUBLAS_OP_T,r,l,m,&alpha,B,m,A,l,&beta,C,r), file, linenumber);
}
///////////////////////////////////////////////////////////////////////////////////////////////////


vector<int> range(int n) {
  vector<int> ret(n);
  for (int i=0; i<n; i++)
    ret[i]=i;
  return ret;
}



//Adapted from http://stackoverflow.com/questions/14902876/indices-of-the-k-largest-elements-in-an-unsorted-length-n-array
template <typename t> std::vector<int> vectorTopIndices(vector<t> &test, int k) {
  std::priority_queue<std::pair<t, int> > q;
  for (int i = 0; i < test.size(); ++i)
    q.push(std::pair<t, int>(test[i], i));
  vector<int> indices;
  for (int i = 0; i < k; ++i) {
    indices.push_back(q.top().second);
    q.pop();
  }
  return indices;
}



void matrixMul2x2inPlace(float& c00, float& c01, float& c10, float& c11, float a00, float a01, float a10, float a11) { //c<-c*a
  float t00=c00*a00+c01*a10;  float t01=c00*a01+c01*a11;
  float t10=c10*a00+c11*a10;  float t11=c10*a01+c11*a11;
  c00=t00;c01=t01;
  c10=t10;c11=t11;
}
