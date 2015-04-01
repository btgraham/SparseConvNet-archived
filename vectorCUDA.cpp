template <typename t> void vectorCUDA<t>::copyToCPU() {
  if (onGPU) {
    onGPU=false;
    if (dsize>0)  {
      vec.resize(dsize);
      cudaSafeCall(cudaMemcpy(&vec[0],d_vec,sizeof(t)*dsize,cudaMemcpyDeviceToHost));
      cudaSafeCall(cudaFree(d_vec));
    }
  }
}
template <typename t> void vectorCUDA<t>::copyToGPU() {
  if (!onGPU) {
    onGPU=true;
    if (vec.size()>0)  {
      dsize=vec.size();
      cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
      if (d_vec==NULL)
        cout <<"d_vec: " << d_vec<<endl;
      cudaSafeCall(cudaMemcpy(d_vec,&vec[0],sizeof(t)*dsize,cudaMemcpyHostToDevice));
      vec.clear();
    } else {
      dsize=0;
    }
  }
}
template <typename t> void vectorCUDA<t>::copyToGPU(cudaStream_t stream) {
  if (!onGPU) {
    onGPU=true;
    if (vec.size()>0)  {
      dsize=vec.size();
      cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
      cudaSafeCall(cudaMemcpyAsync(d_vec,&vec[0],sizeof(t)*dsize,cudaMemcpyHostToDevice,stream));
      vec.clear();
    }
  }
}
template <typename t> t*& vectorCUDA<t>::dPtr() {
  copyToGPU();
  return d_vec;
}
template <typename t> vector<t>& vectorCUDA<t>::hVector() {
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
  if (total!=total) exit(1);
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
      if (dsize>0)
        cudaSafeCall(cudaFree(d_vec));
      if (n>0) {
        cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*n));
        if (d_vec==NULL)
          cout <<"d_vec: " << d_vec<<endl;
      }
      dsize=n;
    }
  } else {
    vec.resize(n);
  }
}
template <typename t> vectorCUDA<t>::vectorCUDA(bool onGPU, int dsize) : onGPU(onGPU), dsize(dsize) {
  if (onGPU && dsize>0) {
    cudaSafeCall(cudaMalloc((void**) &d_vec, sizeof(t)*dsize));
    if (d_vec==NULL)
      cout <<"d_vec: " << d_vec<<endl;
  } else {
    vec.resize(dsize);
  }
}
template <typename t> vectorCUDA<t>::~vectorCUDA() {
  if (onGPU && dsize>0)
    cudaSafeCall(cudaFree(d_vec));
}
template <typename t> void vectorCUDA<t>::printSubset(const char *name, int nCol,int maxPrint) {
  RNG rng;
  copyToCPU();
  int nRow=vec.size()/nCol;
  cout << name << " " << nRow << " " << nCol << endl;
  vector<int> rr=rng.NchooseM(nRow,min(maxPrint,nRow));
  vector<int> rc=rng.NchooseM(nCol,min(maxPrint,nCol));
  for (int i=0;i<rr.size(); i++) {
    for (int j=0;j<rc.size(); j++) {
      cout.precision(3);
      cout <<scientific<< vec[rr[i]*nCol+rc[j]] << "\t";
      if (abs(vec[rr[i]*nCol+rc[j]])>1000000) exit(1);
    }
    cout << endl;
  }
  cout << "---------------------------------------"<<endl;
}
