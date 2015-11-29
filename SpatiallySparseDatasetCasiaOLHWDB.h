#pragma once
#include "SpatiallySparseDataset.h"

SpatiallySparseDataset CasiaOLHWDB11TrainSet(int renderSize,
                                             OnlineHandwritingEncoding enc);
SpatiallySparseDataset CasiaOLHWDB101112TrainSet(int renderSize,
                                                 OnlineHandwritingEncoding enc);
SpatiallySparseDataset CasiaOLHWDB11TestSet(int renderSize,
                                            OnlineHandwritingEncoding enc);
