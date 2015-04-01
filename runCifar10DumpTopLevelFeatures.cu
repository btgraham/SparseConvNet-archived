#include "SparseConvNet.h"
#include "OpenCVPicture.h"
#include "OpenCVPicture_AffineTransform.h"
#include "readCIFAR10.h"

int epoch=500;
int cudaDevice=1;

class SparseConvNet : public SpatiallySparseCNN {
public:
  SparseConvNet(int nInputFeatures, int nClasses, int cudaDevice) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    cnn.push_back(new ColorShiftLayer(0.2,0.2,0.6,0.6));
    int i=1;
    float alpha=powf(2,0.5);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerMP(32*i++,2,1,1,1,VLEAKYRELU,0);
    addSoftmaxLayer();
  }
};


int main() {
  string baseName="/tmp/Cifar10";

  SpatialDataset trainSet=Cifar10TrainSet();   //For training
  SpatialDataset trainSet2=Cifar10TrainSet();trainSet2.type=TESTBATCH; //For calculating the top hidden layer for the training set (without dropout or data augmentation)
  SpatialDataset testSet=Cifar10TestSet(); //For calculating the top hidden layer for the test set

  int batchSize=100;
  trainSet.summary();
  trainSet2.summary();
  testSet.summary();
  SparseConvNet cnn(trainSet.nFeatures,trainSet.pictures.size(),cudaDevice);
  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
    //cnn.processDatasetDumpTopLevelFeatures(trainSet2, batchSize);
    //cnn.processDatasetDumpTopLevelFeatures(testSet, batchSize);
  }
  for (epoch++;epoch<=100000;epoch++) {
    cout <<"epoch: " << epoch << endl;
    cnn.processDataset(trainSet, batchSize,0.0003);
    if (epoch%10==0) {
      cnn.processDataset(testSet, batchSize,0.0003);
      cnn.saveWeights(baseName,epoch);
    }
  }
}
