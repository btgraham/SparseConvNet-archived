#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetMnist.h"

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=100;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  return pic; //No data augmentation
  // pic->loadData();
  // if (type==TRAINBATCH)
  //   pic->jiggle(rng,2);
  //return pic;
}

class CNN : public SparseConvNet {
public:
  CNN (int dimension, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=-1, int nTop=1);
};
CNN::CNN
(int dimension, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  int l=0;
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.1);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.2);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.3);
  addLeNetLayerMP   (32*(++l),2,1,1,1,          VLEAKYRELU,0.4);
  addLeNetLayerMP   (32*(++l),1,1,1,1,          VLEAKYRELU,0.5);
  addSoftmaxLayer();
}

int main() {
  std::string baseName="weights/mnist";

  SpatiallySparseDataset trainSet=MnistTrainSet();
  SpatiallySparseDataset testSet=MnistTestSet();

  trainSet.summary();
  testSet.summary();
  CNN cnn(2,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);
  //DeepCNet cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.01 * epoch));
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
    }
  }
}
