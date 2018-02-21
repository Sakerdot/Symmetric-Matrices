# Symmetric Matrix

## Summary
Header only library of Symmetric Matrices.  

Supports addition, subtraction, multiplication using operators `+`, `-`, `*`, and initialization from an Eigen Matrix.  
It also suports usage of the operator `()` to accesss a coefficient of the matrix, such as `A(i, j)`.

Requires the Eigen library and C++11. 

## Setup and usage
To install, it needs the Eigen library to be installed (check out https://eigen.tuxfamily.org/dox/GettingStarted.html), then it can be
included and that's it.
```cpp
#include "SymmetricMatrix.h"
```

An example of usage can be seen on file `tests.cc`
