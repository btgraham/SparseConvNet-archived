#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=50;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  //writeImage(pic->mat,epoch++);std::cout<<"!\n";
  if (type==TRAINBATCH)
    pic->colorDistortion(rng, 0.1*255, 0.15*255, 0.8, 0.8);
  if (type==TRAINBATCH and epoch<=500) {
    float
      c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
      c10=0, c11=1;
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    if (rng.randint(2)==0) c00*=-1; //Horizontal flip
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
    transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
    //writeImage(pic->mat,epoch++);std::cout<<"!\n";
    pic->jiggle(rng,16);
  }
  return pic;
}

int main() {
  std::string baseName="weights/cifar10";

  SpatiallySparseDataset trainSet=Cifar10TrainSet();
  SpatiallySparseDataset testSet=Cifar10TestSet();

  trainSet.summary();
  testSet.summary();
  POFMPSparseConvNet cnn(2,11,96,powf(2,0.3333),VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.5,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;epoch<=510;epoch++) {
    std::cout <<"epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.005 * epoch),0.99);
    if (epoch%5==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize/2);
    }
  }
  cnn.processDatasetRepeatTest(testSet, batchSize/2,12);
}
