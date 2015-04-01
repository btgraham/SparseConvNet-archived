#include "SparseConvNet.h"
const int onlineHandwritingCharacterScale = 64;
const float delta=onlineHandwritingCharacterScale/5;
const int logSigDepth=3;
#include "OnlineHandwritingLogSignature.h"
#include "readCasiaOLHWDB.h"

Picture* OnlinePicture::distort(RNG& rng) {
  OnlinePicture* pic=new OnlinePicture(*this);
  jiggleStrokes(pic->ops,rng,1);
  stretchXY(pic->ops,rng,0.3);
  int r=rng.randint(3);
  if (r==0) rotate(pic->ops,rng,0.3);
  if (r==1) slant_x(pic->ops,rng,0.3);
  if (r==2) slant_y(pic->ops,rng,0.3);
  jiggleCharacter(pic->ops,rng,12);
  return pic;
}


int epoch=0;
int cudaDevice=-1;

int main() {
  string baseName="/tmp/casia";

  SpatialDataset trainSet=Casia11TrainSet();
  SpatialDataset testSet=Casia11TestSet();

  int batchSize=100;
  trainSet.summary();
  testSet.summary();
  DeepCNiN cnn(6,160,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);
  if (epoch++>0)
    cnn.loadWeights(baseName,epoch-1);
  for (;;epoch++) {
    cout <<"epoch: " << epoch << flush;
    trainSet.shuffle();
    iterate(cnn, trainSet, batchSize,0.003*exp(-0.005 * epoch));
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
      iterate(cnn, testSet,  batchSize);
    }
  }
}
