#include "SparseConvNet.h"
#include "OpenCVPicture.h"
#include "OpenCVPicture_AffineTransform.h"
#include "readCIFAR100.h"

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU

float dropoutProbabilityMultiplier=0;// Set to 0.5 say to use dropout
int nFeaturesPerLevel(int i) {
  return 32*(i+1); //This can be increased
}

class SparseConvNet : public SpatiallySparseCNN {
public:
  SparseConvNet(int nInputFeatures, int nClasses, int cudaDevice,int nTop=1) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    const float fmpShrink=powf(2,0.3333); //Fractional Max-Pooling ratio.
    int l=12; //Number of levels of FMP
    cnn.push_back(new ColorShiftLayer(0.2,0.3,0.6,0.6));
    for (int i=0;i<l;i++) {
      addLeNetLayerPOFMP(nFeaturesPerLevel(i),2,1,fmpShrink,VLEAKYRELU,dropoutProbabilityMultiplier*i/(l+1));
    }
    addLeNetLayerMP(nFeaturesPerLevel(l),2,1,1,1,VLEAKYRELU,dropoutProbabilityMultiplier*l/(l+1));
    addNetworkInNetworkLayer(nFeaturesPerLevel(l+1),VLEAKYRELU,dropoutProbabilityMultiplier);
    addSoftmaxLayer();
  }
};

int main() {
  string baseName="weights/cifar100";

  SpatialDataset trainSet=Cifar100TrainSet();
  SpatialDataset testSet=Cifar100TestSet();

  int batchSize=50;
  trainSet.summary();
  testSet.summary();
  SparseConvNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    cout <<"epoch: " << epoch << flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.02 * epoch)); //reduce annealing rate for better results ...
    if (epoch%50==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
      cnn.processDatasetRepeatTest(testSet, batchSize, 12);
    }
  }
}
