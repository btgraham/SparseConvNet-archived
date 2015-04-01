#include "signature.h"
int logSigDimension=3;
int nInputFeatures=1+logsigdim(logSigDimension,logSigDepth);
#include "OnlineHandwriting.h"

vector<vector<float> > signatureWindows (OnlinePenStroke &path, int windowSize) {
  vector<vector<float> > sW(path.size());
  vector<float> repPath;
  for (int i=0;i<path.size();++i) {
    repPath.push_back(i);
    repPath.push_back(path[i].x);
    repPath.push_back(path[i].y);
  }
  for (int i=0;i<path.size();++i) {
    sW[i].resize(nInputFeatures);
    int first=max(0,i-windowSize);
    int last=min((int)path.size()-1,i+windowSize);
    logSignature(&repPath[3*first],last-first+1,logSigDimension,logSigDepth,&sW[i][0]);
  }
  return sW;
}

void OnlinePicture::codifyInputData(SpatiallySparseBatch &batch) {
  for(int i=0;i<nInputFeatures;i++)
    batch.interfaces[0].features.hVector().push_back(0);  //Background feature
  int backgroundNullVectorNumber=batch.interfaces[0].nSpatialSites++;
  batch.interfaces[0].backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,backgroundNullVectorNumber);

  int windowSize=4;
  float scale=delta/windowSize;
  int multiplier=max((int)(2*scale+0.5),1);
  for (int i=0; i<ops.size(); i++) {
    OnlinePenStroke CSP=constantSpeed(ops[i],scale,1);
    OnlinePenStroke csp=constantSpeed(ops[i],scale,multiplier);
    vector<vector<float> > sW=
      signatureWindows(CSP,windowSize);
    for (int j=0; j<csp.size(); j++) {
      int J=j/multiplier;
      int x=mapToGrid(csp[j].x,batch.interfaces[0].spatialSize);
      int y=mapToGrid(csp[j].y,batch.interfaces[0].spatialSize);
      int n=x*batch.interfaces[0].spatialSize+y;
      if (grid[n]==backgroundNullVectorNumber) {
        grid[n]=batch.interfaces[0].nSpatialSites++;
        batch.interfaces[0].features.hVector().push_back(1);
        for (int k=0; k<nInputFeatures-1; k++)
          batch.interfaces[0].features.hVector().push_back
            (sW[J][k]/onlineHandwritingRegularizingConstants[k]);
      }
    }
  }
  batch.interfaces[0].grids.push_back(grid);
  batch.interfaces[0].batchSize++;
  batch.labels.hVector().push_back(label);
}
