#include <stdint.h>

#pragma once
enum batchType { TRAINBATCH, TESTBATCH, UNLABELEDBATCH, RESCALEBATCH };
#ifdef TYPES_CPP
const char *batchTypeNames[] = {"TRAINBATCH", "TESTBATCH", "UNLABELEDBATCH",
                                "RESCALEBATCH"};
#else
extern const char *batchTypeNames[];
#endif

enum ActivationFunction {
  NOSIGMOID,
  RELU,
  VLEAKYRELU,
  LEAKYRELU,
  TANH,
  SOFTMAX,
  PRELU,
  SIGMOID
};
#ifdef TYPES_CPP
const char *sigmoidNames[] = {"", "ReLU", "VeryLeakyReLU", "LeakyReLU", "tanh",
                              "Softmax Classification", "PReLU", "Sigmoid"};
#else
extern const char *sigmoidNames[];
#endif

enum OnlineHandwritingEncoding {
  Simple,
  Octogram,
  UndirectedOctogram,
  LogSignature1,
  LogSignature2,
  LogSignature3,
  LogSignature4,
  SpaceTime3d,
  VectorSpaceTime3d
};
#ifdef TYPES_CPP
int OnlineHandwritingEncodingSize[] = {1, 8, 4, 3, 6, 14, 32, 1, 1 + 2};
#else
extern int OnlineHandwritingEncodingSize[];
#endif

// Accumulator to Keep track of the number of multiply-accumulare operaions
// needed to to process a batch/dataset
#ifdef TYPES_CPP
__int128_t multiplyAddCount;
#else
extern __int128_t multiplyAddCount;
#endif
