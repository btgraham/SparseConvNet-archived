//Ben Graham, University of Warwick, 2015 b.graham@warwick.ac.uk
//SparseConvNet is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//SparseConvNet is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with SparseConvNet.  If not, see <http://www.gnu.org/licenses/>.

// All hidden layers should have size that is a multiple of KERNELBLOCKSIZE == 32
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <queue>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <boost/assign/list_of.hpp>
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/thread.hpp>
#include <boost/timer/timer.hpp>
#include "cuda.h"
#include <cublas_v2.h>
using namespace std;
using namespace boost::assign;

//1 to include bias terms, 0 to exclude them
#define BIAS 1

#include "Rng.h"
#include "utilities.h"
#include "vectorCUDA.h"
#include "Batches.h"
#include "SpatiallySparseLayer.h"
#include "ColorShiftLayer.h"
#include "SigmoidLayer.h"
#include "MaxPoolingLayer.h"
#include "AveragePoolingLayer.h"
#include "NetworkInNetworkLayer.h"
#include "ConvolutionalLayer.h"
#include "XConvLayer.h"
#include "X4ConvLayer.h"
#include "YConvLayer.h"
#include "SoftmaxClassifier.h"
#include "IndexLearnerLayer.h"
#include "SpatialDataset.h"
#include "BatchProducer.h"
#include "SpatiallySparseCNN.h"
#include "NetworkArchitectures.h"

#include "Rng.cpp"
#include "vectorCUDA.cpp"
#include "BatchProducer.cpp"
#include "SpatiallySparseCNN.cpp"
