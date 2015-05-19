#include "vectorCUDA.h"
#include <iostream>
#include "Rng.h"
#include "cudaUtilities.h"
#include <cstring>

template <typename t> void vectorCUDA<t>::copyToCPU() {
  if (onGPU) {
    onGPU=false;
    if (dsize>0)  {
      vec.resize(dsize);
      cudaSafeCall(cudaMemcpy(&vec[0],d_vec,sizeof(t)*dsize,cudaMemcpyDeviceToHost));
    }
  }
}
template <typename t> void vectorCUDA<t>::copyToGPU() {
  if (!onGPU) {
    onGPU=true;
    dsize=vec.size();
    if (dsize>dAllocated) {
      if (dAllocated>0) {
        cudaSafeCall(cudaFree(d_vec));
      }
      dAllocated=dsize;
      cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dAllocated));
    }
    if (dsize>0)  {
      cudaSafeCall(cudaMemcpy(d_vec,&vec[0],sizeof(t)*dsize,cudaMemcpyHostToDevice));
    }
    vec.clear();
  }
}

template <typename t> void vectorCUDA<t>::copyToGPUAsync(cudaMemStream &memStream) {
  if (!onGPU) {
    onGPU=true;
    dsize=vec.size();
    if (dsize>dAllocated) {
      if (dAllocated>0) {
        cudaSafeCall(cudaFree(d_vec));
      }
      dAllocated=dsize;
      cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dAllocated));
    }
    if (memStream.pinnedMemorySize<sizeof(t)*dsize) {
      memStream.pinnedMemorySize=2*sizeof(t)*dsize;
      cudaSafeCall(cudaFreeHost(memStream.pinnedMemory));
      cudaSafeCall(cudaMallocHost(&memStream.pinnedMemory,
                                  memStream.pinnedMemorySize));
    }
    if (dsize>0)  {
      std::memcpy(memStream.pinnedMemory,&vec[0],sizeof(t)*dsize);
      cudaSafeCall(cudaMemcpyAsync(d_vec,memStream.pinnedMemory,sizeof(t)*dsize,cudaMemcpyHostToDevice,memStream.stream));
      cudaStreamSynchronize(memStream.stream);
    }
    vec.clear();
  }
}
template <typename t> void vectorCUDA<t>::copyToCPUAsync(cudaMemStream &memStream) {
  if (onGPU) {
    onGPU=false;
    if (memStream.pinnedMemorySize<sizeof(t)*dsize) {
      memStream.pinnedMemorySize=2*sizeof(t)*dsize;
      cudaSafeCall(cudaFreeHost(memStream.pinnedMemory));
      cudaSafeCall(cudaMallocHost(&memStream.pinnedMemory,
                                  memStream.pinnedMemorySize));
    }
    if (dsize>0)  {
      cudaSafeCall(cudaMemcpyAsync(memStream.pinnedMemory,d_vec,sizeof(t)*dsize,cudaMemcpyDeviceToHost,memStream.stream));
      vec.resize(dsize);
      cudaStreamSynchronize(memStream.stream);
      std::memcpy(&vec[0],memStream.pinnedMemory,sizeof(t)*dsize);
    }
  }
}

template <typename t> t*& vectorCUDA<t>::dPtr() {
  copyToGPU();
  return d_vec;
}
template <typename t> std::vector<t>& vectorCUDA<t>::hVector() {
  copyToCPU();
  return vec;
}
template <typename t> int vectorCUDA<t>::size() {
  if (onGPU) return dsize;
  return vec.size();
}
template <typename t> float vectorCUDA<t>::meanAbs() {
  float total=0;
  for (int i=0;i<size();i++)
    total+=fabs(hVector()[i]);
  if (total!=total) {
    std::cout << "NaN in vectorCUDA<t>::meanAbs()\n";
    exit(1);
  }
  return total/size();
}
template <typename t> void vectorCUDA<t>::multiplicativeRescale(float multiplier) {
  for (int i=0;i<size();i++)
    hVector()[i]*=multiplier;
}
template <typename t> void vectorCUDA<t>::setZero() {
  if (onGPU) {
    cudaSafeCall(cudaMemset(d_vec,  0,sizeof(t)*dsize));
  } else {
    memset(&vec[0],0,sizeof(t)*vec.size());
  }
}
template <typename t> void vectorCUDA<t>::setConstant(float a) {
  copyToCPU();
  for (int i=0;i<vec.size();i++)
    vec[i]=a;
}
template <typename t> void vectorCUDA<t>::setUniform(float a,float b) {
  RNG rng;
  copyToCPU();
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.uniform(a,b);
}
template <typename t> void vectorCUDA<t>::setBernoulli(float p) {
  RNG rng;
  copyToCPU();
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.bernoulli(p);
}
template <typename t> void vectorCUDA<t>::setNormal(float mean, float sd) {
  RNG rng;
  copyToCPU();
  for (int i=0;i<vec.size();i++)
    vec[i]=rng.normal(mean,sd);
}
template <typename t> void vectorCUDA<t>::resize(int n) {
  if (onGPU) {
    if (dsize!=n) {
      dsize=n;
      if (dsize>dAllocated) {
        if (dAllocated>0) {
          cudaSafeCall(cudaFree(d_vec));
        }
        dAllocated=dsize;
        cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
      }
    }
  } else {
    vec.resize(n);
  }
}
template <typename t> vectorCUDA<t>::vectorCUDA(bool onGPU, int dsize) : onGPU(onGPU), dsize(dsize), dAllocated(0) {
  if (onGPU && dsize>0) {
    dAllocated=dsize;
    cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));

  } else {
    vec.resize(dsize);
  }
}
template <typename t> vectorCUDA<t>::~vectorCUDA() {
  if (onGPU && dAllocated>0) {
    cudaSafeCall(cudaFree(d_vec));
  }
}
template <typename t> void vectorCUDA<t>::printSubset(const char *name, int nCol,int maxPrint) {
  RNG rng;
  copyToCPU();
  int nRow=vec.size()/nCol;
  std::cout << name << " " << nRow << " " << nCol << std::endl;
  std::vector<int> rr=rng.NchooseM(nRow,min(maxPrint,nRow));
  std::vector<int> rc=rng.NchooseM(nCol,min(maxPrint,nCol));
  for (int i=0;i<rr.size(); i++) {
    for (int j=0;j<rc.size(); j++) {
      std::cout.precision(3);
      std::cout <<std::scientific<< vec[rr[i]*nCol+rc[j]] << "\t";
      if (abs(vec[rr[i]*nCol+rc[j]])>1000000) exit(1);
    }
    std::cout << std::endl;
  }
  std::cout << "---------------------------------------"<<std::endl;
}

//Borrowing from http://stackoverflow.com/questions/22734067/checking-if-a-matrix-contains-nans-or-infinite-values-in-cuda
__global__ void dNanCheck(float* d, int n, bool* result) {
  for (int i=threadIdx.x; i<n; i+=128)
    if (isnan(d[i])) *result=false;
}
template <typename t> void vectorCUDA<t>::check(const char* file, int linenumber) {}
template <> void vectorCUDA<float>::check(const char* file, int linenumber) {
  if (onGPU and dsize>0) {
    if (dsize==0) return;
    bool *d_result, h_result=true;
    cudaMalloc((void **)&d_result, sizeof (bool));
    cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
    dNanCheck<<<1,128>>>(d_vec, dsize, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    if (!h_result) {
      std::cout <<"NaN " << file <<" " << linenumber << "\n";
      exit(1);
    }
  }
  if (!onGPU) {
    bool result=true;
    for (int i=0; i<vec.size();++i)
      if (isnan(vec[i])) result=false;
    if (!result) {
      std::cout <<"NaN " << file <<" " << linenumber << "\n";
      exit(1);
    }
  }
}

template class vectorCUDA<float>;
template class vectorCUDA<int>;
