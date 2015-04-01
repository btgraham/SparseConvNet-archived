int nInputFeatures=1;
#include "OnlineHandwriting.h"

void OnlinePicture::codifyInputData(SpatiallySparseBatch &batch) {
  for(int i=0;i<nInputFeatures;i++)
    batch.interfaces[0].features.hVector().push_back(0);  //Background feature
  int backgroundNullVectorNumber=batch.interfaces[0].nSpatialSites++;
  batch.interfaces[0].backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,backgroundNullVectorNumber);
  vector<float> w(batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,0);
  for (int i=0; i<ops.size(); i++) {
    OnlinePenStroke csp=constantSpeed(ops[i],0.1);
    for (int j=0;j<csp.size()-1;j++) {
      int n=mapToGrid(csp[j].x,batch.interfaces[0].spatialSize)*batch.interfaces[0].spatialSize+mapToGrid(csp[j].y,batch.interfaces[0].spatialSize);
      w[n]=1;
    }
  }
  for (int n=0;n<batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize;n++)
    if (w[n]==1) {
      grid[n]=batch.interfaces[0].nSpatialSites++;
      batch.interfaces[0].features.hVector().push_back(1);
    }
  batch.interfaces[0].grids.push_back(grid);
  batch.interfaces[0].batchSize++;
  batch.labels.hVector().push_back(label);
}
