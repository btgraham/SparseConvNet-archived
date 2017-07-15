# SparseConvNet
## A spatially-sparse convolutional neural network
### [Ben Graham](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/), [University of Warwick](http://www2.warwick.ac.uk/fac/sci/statistics/), 2013-2015, GPLv3

SparseConvNet is a convolutional neural network for processing sparse data on a variety of lattices, i.e.
(i) the square lattice,
(ii) the triangular lattice,
(iii) the cubic lattice,
(iv) the tetrahedral lattice, ...  
![lattice](/figures/lattices.png)  
... and of course the hyper-cubic and hyper-tetrahedral 4D lattices as well.

Data is sparse if most sites take the value zero. For example, if a loop of string has a knot in it, and you trace the shape of the string in a 3D lattice, most sites will not form part of the knot (left). Applying a 2x2x2 convolution (middle), and a pooling operation (right), the set of non-zero sites stays fairly small:
![lattice](/figures/trefoil.png)

This can be used to analyse 3D models, or space-time paths.
Here are some examples from a [3D object dataset](http://www.itl.nist.gov/iad/vug/sharp/contest/2014/Generic3D/index.html). The insides are hollow, so the data is fairly sparse. The computational complexity of processing the models is related to the [fractal dimension](http://en.wikipedia.org/wiki/Fractal_dimension) of the underlying objects.

![lattice](/figures/shrec.png)
Top row: four exemplars of snakes. Bottom row: an ant, an elephant, a robot and a tortoise.

## [Wiki](https://github.com/btgraham/SparseConvNet/wiki)
## [Dependencies and Installation](https://github.com/btgraham/SparseConvNet/wiki/Installation)

**************************************************************************
SparseConvNet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SparseConvNet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](http://www.gnu.org/licenses/) for more details.
**************************************************************************
