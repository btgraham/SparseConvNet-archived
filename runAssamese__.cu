#include "SparseConvNet.h"
const int onlineHandwritingCharacterScale = 64;
#include "OnlineHandwritingSimple.h"
#include "readAssamese.h"

Picture* OnlinePicture::distort(RNG& rng, batchType type) {
  OnlinePicture* pic=new OnlinePicture(*this);
  if (type==TRAINBATCH) {
    jiggleStrokes(pic->ops,rng,1);
    stretchXY(pic->ops,rng,0.3);
    int r=rng.randint(3);
    if (r==0) rotate(pic->ops,rng,0.3);
    if (r==1) slant_x(pic->ops,rng,0.3);
    if (r==2) slant_y(pic->ops,rng,0.3);
    jiggleCharacter(pic->ops,rng,12);
  }
  return pic;
}

class FractionalMPSparseConvNet : public SpatiallySparseCNN {
public:
  FractionalMPSparseConvNet(int nInputFeatures, int nClasses, int cudaDevice) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    int l=1;
    for (;l<=10;l++) {
      addLeNetLayerROFMP(32*l,2,1,powf(2,0.5),VLEAKYRELU);
    }
    addLeNetLayerMP         (32*(l++),2,1,1,1,VLEAKYRELU);
    addNetworkInNetworkLayer(32*(l++),        VLEAKYRELU);
    addSoftmaxLayer();
  }
};

int epoch=0;
int cudaDevice=-1;

int main() {
  string baseName="weights/assamese";

  SpatialDataset trainSet=AssameseTrainSet();
  SpatialDataset testSet=AssameseTestSet();

  int batchSize=100;
  trainSet.summary();
  testSet.summary();
  FractionalMPSparseConvNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);
  if (epoch++>0)
    cnn.loadWeights(baseName,epoch-1);
  for (;epoch<=600;epoch++) {
    cout <<"epoch: " << epoch << flush;
    trainSet.shuffle();
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.005 * epoch));
    if (epoch%10==0)
      cnn.saveWeights(baseName,epoch);
    if (epoch%100==0)
      cnn.processDatasetRepeatTest(testSet,batchSize,12);
  }
}
