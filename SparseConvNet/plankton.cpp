// epoch: 50
// Data/kagglePlankton/train Mistakes:20.2894% NLL:0.593941
// MegaMultiplyAdds/sample:760 time:86s GigaMultiplyAdds/s:266 rate:351/s
// Data/kagglePlankton/testPrivate Mistakes:20.4956% NLL:0.599171
// MegaMultiplyAdds/sample:751 time:60s GigaMultiplyAdds/s:264 rate:352/s
// ....
// Loading network parameters from weights/plankton_epoch-50.cnn
// Data/kagglePlankton/testPublic rep 1/24 Mistakes: 21.1591% NLL 0.679194
// MegaMultiplyAdds/sample:300 time:14s GigaMultiplyAdds/s:188 rate:628/s
// Data/kagglePlankton/testPublic rep 2/24 Mistakes: 20.2683% NLL 0.635085
// MegaMultiplyAdds/sample:601 time:28s GigaMultiplyAdds/s:191 rate:318/s
// Data/kagglePlankton/testPublic rep 3/24 Mistakes: 20.0484% NLL 0.623887
// MegaMultiplyAdds/sample:902 time:42s GigaMultiplyAdds/s:193 rate:214/s
// Data/kagglePlankton/testPublic rep 4/24 Mistakes: 19.8064% NLL 0.620122
// MegaMultiplyAdds/sample:1203 time:56s GigaMultiplyAdds/s:194 rate:162/s
// Data/kagglePlankton/testPublic rep 5/24 Mistakes: 19.8064% NLL 0.616039
// MegaMultiplyAdds/sample:1504 time:70s GigaMultiplyAdds/s:195 rate:130/s
// Data/kagglePlankton/testPublic rep 6/24 Mistakes: 19.7625% NLL 0.613289
// MegaMultiplyAdds/sample:1804 time:83s GigaMultiplyAdds/s:195 rate:108/s
// Data/kagglePlankton/testPublic rep 7/24 Mistakes: 19.6525% NLL 0.611321
// MegaMultiplyAdds/sample:2105 time:97s GigaMultiplyAdds/s:195 rate:93/s
// Data/kagglePlankton/testPublic rep 8/24 Mistakes: 19.7295% NLL 0.609608
// MegaMultiplyAdds/sample:2406 time:111s GigaMultiplyAdds/s:195 rate:81/s
// Data/kagglePlankton/testPublic rep 9/24 Mistakes: 19.7295% NLL 0.608343
// MegaMultiplyAdds/sample:2707 time:125s GigaMultiplyAdds/s:195 rate:72/s
// Data/kagglePlankton/testPublic rep 10/24 Mistakes: 19.5975% NLL 0.607367
// MegaMultiplyAdds/sample:3008 time:139s GigaMultiplyAdds/s:196 rate:65/s
// Data/kagglePlankton/testPublic rep 11/24 Mistakes: 19.5865% NLL 0.606214
// MegaMultiplyAdds/sample:3309 time:153s GigaMultiplyAdds/s:196 rate:59/s
// Data/kagglePlankton/testPublic rep 12/24 Mistakes: 19.6085% NLL 0.605922
// MegaMultiplyAdds/sample:3609 time:167s GigaMultiplyAdds/s:196 rate:54/s
// Data/kagglePlankton/testPublic rep 13/24 Mistakes: 19.5755% NLL 0.605546
// MegaMultiplyAdds/sample:3910 time:180s GigaMultiplyAdds/s:196 rate:50/s
// Data/kagglePlankton/testPublic rep 14/24 Mistakes: 19.5645% NLL 0.60499
// MegaMultiplyAdds/sample:4211 time:194s GigaMultiplyAdds/s:196 rate:47/s
// Data/kagglePlankton/testPublic rep 15/24 Mistakes: 19.5645% NLL 0.604357
// MegaMultiplyAdds/sample:4511 time:208s GigaMultiplyAdds/s:196 rate:44/s
// Data/kagglePlankton/testPublic rep 16/24 Mistakes: 19.6085% NLL 0.603568
// MegaMultiplyAdds/sample:4812 time:222s GigaMultiplyAdds/s:196 rate:41/s
// Data/kagglePlankton/testPublic rep 17/24 Mistakes: 19.5535% NLL 0.603117
// MegaMultiplyAdds/sample:5113 time:236s GigaMultiplyAdds/s:196 rate:38/s
// Data/kagglePlankton/testPublic rep 18/24 Mistakes: 19.6305% NLL 0.602926
// MegaMultiplyAdds/sample:5413 time:250s GigaMultiplyAdds/s:196 rate:36/s
// Data/kagglePlankton/testPublic rep 19/24 Mistakes: 19.6195% NLL 0.602641
// MegaMultiplyAdds/sample:5714 time:264s GigaMultiplyAdds/s:196 rate:34/s
// Data/kagglePlankton/testPublic rep 20/24 Mistakes: 19.5755% NLL 0.60174
// MegaMultiplyAdds/sample:6015 time:277s GigaMultiplyAdds/s:196 rate:33/s
// Data/kagglePlankton/testPublic rep 21/24 Mistakes: 19.7185% NLL 0.601709
// MegaMultiplyAdds/sample:6316 time:291s GigaMultiplyAdds/s:197 rate:31/s
// Data/kagglePlankton/testPublic rep 22/24 Mistakes: 19.6965% NLL 0.601427
// MegaMultiplyAdds/sample:6617 time:305s GigaMultiplyAdds/s:197 rate:30/s
// Data/kagglePlankton/testPublic rep 23/24 Mistakes: 19.6415% NLL 0.600754
// MegaMultiplyAdds/sample:6918 time:319s GigaMultiplyAdds/s:197 rate:28/s
// Data/kagglePlankton/testPublic rep 24/24 Mistakes: 19.5865% NLL 0.600615
// MegaMultiplyAdds/sample:7218 time:333s GigaMultiplyAdds/s:197 rate:27/s

#include "SparseConvNet.h"
#include "SpatiallySparseDatasetOpenCV.h"

int epoch = 0;
int cudaDevice = -1; // PCI bus ID: -1 for default GPU
int batchSize = 50;  // Increase/decrease according to GPU memory

Picture *OpenCVPicture::distort(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);
  pic->loadDataWithoutScaling(0);
  float c00 = 1, c01 = 0, // 2x2 identity matrix---starting point for
                          // calculating affine distortion matrix
      c10 = 0, c11 = 1;
  float r, alpha, beta, s = 1;
  if (type == TRAINBATCH) {
    r = rng.uniform(-0.1, 0.1);
    alpha = rng.uniform(0, 2 * 3.1415926535);
    beta = rng.uniform(-0.2, 0.2) + alpha;
  } else {
    r = 0;
    alpha = rng.uniform(0, 2 * 3.1415926535);
    beta = alpha;
  }
  c00 = (1 + r) * cos(alpha);
  c01 = (1 + r) * sin(alpha);
  c10 = -(1 - r) * sin(beta);
  c11 = (1 - r) * cos(beta);
  if (rng.randint(2) == 0) {
    c00 *= -1;
    c01 *= -1;
  } // Horizontal flip
  pic->affineTransform(c00, c01, c10, c11);
  pic->jiggle(rng, 300);
  return pic;
}

int f(int i) { return 32 * (i + 1); }

class FractionalSparseConvNet : public SparseConvNet {
public:
  FractionalSparseConvNet(int nInputFeatures, int nClasses, int cudaDevice)
      : SparseConvNet(2, nInputFeatures, nClasses, cudaDevice) {
    int l = 12;
    float p = 0.25f;
    const float fmpShrink = 1.414;
    for (int i = 0; i < l; i++) {
      addLeNetLayerPOFMP(f(i), 2, 1, 2, fmpShrink, VLEAKYRELU,
                         p * std::max(i - 4, 0) / (l - 3));
    }
    addLeNetLayerPOFMP(f(l), 2, 1, 1, 1, VLEAKYRELU, p * (l - 4) / (l - 3));
    addTerminalPoolingLayer(32);
    addLeNetLayerPOFMP(f(l + 1), 1, 1, 1, 1, VLEAKYRELU, p);
    addSoftmaxLayer();
  }
};

int main() {
  std::string baseName = "weights/plankton";

  OpenCVLabeledDataSet trainSet("Data/kagglePlankton/classList",
                                "Data/kagglePlankton/train", "*.jpg",
                                TRAINBATCH, 255, true, 0);
  trainSet.summary();
  std::cout << "\n ** Use the private test set as as extra source of "
               "training data ! ** \n\n";
  OpenCVLabeledDataSet cheekyExtraTrainSet("Data/kagglePlankton/classList",
                                           "Data/kagglePlankton/testPrivate",
                                           "*.jpg", TRAINBATCH, 255, true, 0);
  cheekyExtraTrainSet.summary();
  std::cout << "\n ** Use the public test set for validation ** \n\n";
  OpenCVLabeledDataSet valSet("Data/kagglePlankton/classList",
                              "Data/kagglePlankton/testPublic", "*.jpg",
                              TESTBATCH, 255, true, 0);
  valSet.summary();

  FractionalSparseConvNet cnn(trainSet.nFeatures, trainSet.nClasses,
                              cudaDevice);

  if (epoch > 0) {
    cnn.loadWeights(baseName, epoch);
    cnn.processDatasetRepeatTest(valSet, batchSize / 2, 24);
  }
  for (epoch++;; epoch++) {
    std::cout << "epoch: " << epoch << std::endl;
    float lr = 0.003 * exp(-0.1 * epoch);
    for (int i = 0; i < 10; ++i) {
      cnn.processDataset(trainSet, batchSize, lr, 0.999);
      cnn.processDataset(cheekyExtraTrainSet, batchSize, lr, 0.999);
    }
    cnn.saveWeights(baseName, epoch);
    cnn.processDatasetRepeatTest(valSet, batchSize, 6);
  }

  // For unlabelled data
  // OpenCVUnlabeledDataSet
  // testSet("Data/kagglePlankton/classList"," ... ","*.jpg",255,true,0);
  // testSet.summary();
  // cnn.processDatasetRepeatTest(testSet, batchSize/2,
  // 24,"plankton.predictions",testSet.header);
}
