#pragma once

//General vector-y object, transferable between CPU and GPU memory
//Replace with CUDA 6.0+ Unified memory??

template <typename t> class vectorCUDA {
private:
  t* d_vec;
  int dsize; //When on GPU
  std::vector<t> vec;
public:
  vectorCUDA(bool onGPU=true, int dsize=0);
  ~vectorCUDA();
  bool onGPU;
  void copyToCPU();
  void copyToGPU();
  void copyToGPU(cudaStream_t stream);
  t*& dPtr();
  vector<t>& hVector();
  int size();
  float meanAbs();
  void multiplicativeRescale(float multiplier);
  void setZero();
  void setConstant(float a=0);
  void setUniform(float a=0,float b=1);
  void setBernoulli(float p=0.5);
  void setNormal(float mean=0, float sd=1);
  void resize(int n);
  void printSubset(const char *name, int nCol,int maxPrint=10);
};
