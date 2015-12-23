#pragma once
#include "SpatiallySparseDataset.h"
#include <iostream>

SpatiallySparseDataset SHREC2015TrainSet(int renderSize, int kFold, int fold);
SpatiallySparseDataset SHREC2015TestSet(int renderSize, int kFold, int fold);
