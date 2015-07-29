#pragma once
#include "SpatiallySparseDataset.h"
#include "OpenCVPicture.h"

SpatiallySparseDataset ImageNet2012TrainSet(int scale=256);
SpatiallySparseDataset ImageNet2012ValidationSet(int scale=256);
SpatiallySparseDataset ImageNet2012TestSet(int scale=256);
