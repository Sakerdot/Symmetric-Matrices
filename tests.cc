#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include "SymmetricMatrix.h"

using Eigen::MatrixXd;

int main()
{
  std::clog << "Starting tests..." << std::endl << std::endl;

  const int ORDER = 10;
  
  MatrixXd m1 = MatrixXd::Random(ORDER, ORDER);

  SymMat<double> sym1(m1);

  for (int i = 0; i < ORDER; ++i)
  {
    for (int j = i; j < ORDER; ++j)
    {
      if (sym1(i, j) != m1(i, j))
      {
        std::clog << "Test failed: SymMat and upper triangular part of Eigen::Matrix are not equal." << std::endl;
        i = ORDER;
        j = ORDER;
      }
    }
  }

  for (int i = 0; i < ORDER; ++i)
  {
    for (int j = i; j < ORDER; ++j)
    {
      if (sym1(i, j) != sym1(j, i))
      {
        std::clog << "Test failed: SymMat(i, j) != SymMat(j, i)." << std::endl;
        i = ORDER;
        j = ORDER;
      }
    }
  }

  MatrixXd ident = MatrixXd::Identity(ORDER, ORDER);

  SymMat<double> symIdent(ident);

  if ((symIdent + symIdent) * ident != ident + ident ||
      (symIdent + ident) != ident + ident)
  {
    std::clog << "Test failed: Addition or multiplication failed." << std::endl;
  }

  MatrixXd mzero(ORDER, ORDER);
  mzero.setZero(ORDER, ORDER);

  if ((sym1 - sym1) * ident != mzero ||
      (symIdent - ident) != mzero)
  {
    std::clog << "Test failed: Subtraction or multiplication failed." << std::endl;    
  }

  if (sym1 * mzero != mzero)
  {
    std:: clog << "Test failed: SymMat * 0 != 0." << std::endl;
  }

  MatrixXd m2 = MatrixXd::Random(ORDER, 5);

  try
  {
    SymMat<double> sym2(m2);
    std::clog << "Test failed: SymMat constructor didn't throw an exception with non square matrix." << std::endl;
  }
  catch (std::exception &e) {}

  std::cout << std::endl << "Tests finished." << std::endl;
}
