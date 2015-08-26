#include "SparseConvNet.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=50;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  float
    c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
    c10=0, c11=1;
  if (type==TRAINBATCH) {
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    if (rng.randint(2)==0) c00*=-1; //Horizontal flip
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
  }
  matrixMul2x2inPlace(c00,         c01,
                      c10,         c11,
                      1,           0,
                      -powf(3,-0.5),powf(0.75,-0.5)); //to triangular lattice

  transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
  if (type==TRAINBATCH)
    pic->jiggle(rng,16);
  return pic;
}

class DeepC2Triangular : public SparseConvTraingLeNet {
public:
  DeepC2Triangular (int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=-1, int nTop=1);
};
DeepC2Triangular::DeepC2Triangular
(int dimension, int l, int k, ActivationFunction fn,
 int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop)
  : SparseConvTriangLeNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<=l;i++) {
    addLeNetLayerMP((i+1)*k,2,1,1,1,fn,p*i*1.0f/l);
    addLeNetLayerMP((i+1)*k,2,1,(i<l)?3:1,(i<l)?2:1,fn,p*i*1.0f/l);
  }
  addSoftmaxLayer();
}


int main() {
  std::string baseName="weights/cifar10";

  SpatiallySparseDataset trainSet=Cifar10TrainSet();
  SpatiallySparseDataset testSet=Cifar10TestSet();

  trainSet.summary();
  testSet.summary();
  DeepC2Triangular cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.1f,cudaDevice);

  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
    cnn.processDatasetRepeatTest(testSet,  batchSize,12);
  }
  for (epoch++;epoch<=800;epoch++) {
    std::cout <<"epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.005 * epoch)); //reduce annealing rate for better results ...
    cnn.processDatasetRepeatTest(testSet,  batchSize,1);
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
    }
  }
  cnn.processDatasetRepeatTest(testSet,  batchSize,12);
}
