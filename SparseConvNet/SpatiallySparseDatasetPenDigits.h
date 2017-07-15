#pragma once
#include "SpatiallySparseDataset.h"

SpatiallySparseDataset PenDigitsTrainSet(int renderSize,
                                         OnlineHandwritingEncoding enc);
SpatiallySparseDataset PenDigitsTestSet(int renderSize,
                                        OnlineHandwritingEncoding enc);
