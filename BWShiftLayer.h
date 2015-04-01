class BWShiftLayer : public SpatiallySparseLayer {
public:
  float sigma1;
  float sigma2;
  float sigma3;
  float sigma4;
  RNG rng;
  BWShiftLayer(float sigma1=0, float sigma2=0, float sigma3=0, float sigma4=0) : sigma1(sigma1),  sigma2(sigma2), sigma3(sigma3),  sigma4(sigma4) {
    cout << "BW shift sigma = " << sigma1 << " " << sigma2 << " " << sigma3 <<" " << sigma4<<endl;
  };
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    assert(input.nFeatures=input.featuresPresent.size());
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=input.nSpatialSites;
    output.grids=input.grids;
    output.backgroundNullVectorNumbers=input.backgroundNullVectorNumbers;
    output.features.hVector()=input.features.hVector();
    output.backpropErrors=input.backpropErrors; //false
    if (input.type==TRAINBATCH) {
      for (int i=0;i<input.batchSize;i++) {
        int bg=input.backgroundNullVectorNumbers[i];
        vector<float> delta1(input.nFeatures);
        vector<float> delta2(input.nFeatures);
        // vector<float> delta3(input.nFeatures);
        // vector<float> delta4(input.nFeatures);
        for (int j=0;j<input.nFeatures;j++) {
          delta1[j]=rng.normal(0,sigma1);
          delta2[j]=rng.normal(0,sigma2);
          // delta3[j]=rng.normal(0,sigma3);
          // delta4[j]=rng.normal(0,sigma4);
        }
        for (int x=0;x<input.spatialSize;x++) {
          for (int y=0;y<input.spatialSize;y++) {
            for (int j=0;j<input.nFeatures;j++) {
              int k=y*input.spatialSize+x;
              if (output.grids[i][k]!=bg) {
                output.features.hVector()[output.grids[i][k]*output.nFeatures+j]+=
                  delta1[j]*input.features.hVector()[output.grids[i][k]*output.nFeatures+j]+
                  delta2[j]*sin(input.features.hVector()[output.grids[i][k]*output.nFeatures+j]*3.1415926535);
              }
            }
          }
        }
      }
    }
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    return outputSpatialSize;
  }
};
