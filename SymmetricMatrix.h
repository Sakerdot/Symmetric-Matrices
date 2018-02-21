#pragma once

#include <stdexcept>
#include <Eigen/Dense>

/**
 * The SymMat class represents a symmetric matrix. It only stores the necessary coefficients, using the
 * Packed Storage method (http://www.netlib.org/lapack/lug/node123.html)
 * 
 * It's initialized by a square Eigen::Matrix, storing only its upper triangular part, or manually by
 * using the access operator() once created.
 * 
 * It supports addition, subtraction and multiplication of matrices with either SymMat or Eigen::Matrix.
 */
template <typename Scalar>
class SymMat
{
public:
  /**
   * Copy constructor.
   */
  SymMat(const SymMat &other) : m_order(other.m_order)
  {
    initArray();

    for (int i = arraySize() - 1; i >= 0; --i)
    {
      m_data[i] = other.m_data[i];
    }
  }

  /**
   * Move constructor.
   */
  SymMat(SymMat &&other) : m_order(other.m_order), m_data(other.m_data)
  {
    other.m_data = nullptr;
  }

  /**
   * Constructor to initialize an order X order symmetric matrix.
   */
  SymMat(int order) : m_order(order)
  {
    initArray();
  }

  /**
   * Constructor to copy the upper triangular part of a Eigen::Matrix into a symmetric matrix.
   * 
   * Throws an exception if other is not a square matrix.
   */
  template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  SymMat(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) : m_order(other.rows())
  {
    if (m_order == other.cols())
    {
      initArray();

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = i; j < m_order; ++j)
        {
          (*this)(i, j) = other(i, j);
        }
      }
    }
    else
    {
      m_order = 0;
      m_data = nullptr;

      throw std::invalid_argument("Expected a square matrix");
    }
  }

  ~SymMat()
  {
    if (m_data)
    {
      delete[] m_data;
    }
  }

  /**
   * Returns SymMat as the addition of two symmetric matrices.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  inline SymMat operator+(const SymMat &other) const
  {
    if (m_order == other.m_order && m_data && other.m_data)
    {
      SymMat addition(m_order);

      for (int i = arraySize() - 1; i >= 0; --i)
      {
        addition.m_data[i] = m_data[i] + other.m_data[i];
      }

      return addition;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to add or two matrices with same order");
    }
  }

  /**
   * Returns Eigen::Matrix as the addition of a symmetric matrix and a Eigen::Matrix.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  inline Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
  operator+(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) const
  {
    if (other.rows() == m_order && other.cols() == m_order && m_data)
    {
      Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> addition(other.rows(), other.cols());

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < m_order; ++j)
        {
          addition(i, j) = (*this)(i, j) + other(i, j);
        }
      }

      return addition;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to add or two matrices with same order");
    }
  }

  /**
   * Returns SymMat as the subtraction of two symmetric matrices.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  inline SymMat operator-(const SymMat &other) const
  {
    if (m_order == other.m_order && m_data && other.m_data)
    {
      SymMat subtraction(m_order);

      for (int i = arraySize() - 1; i >= 0; --i)
      {
        subtraction.m_data[i] = m_data[i] - other.m_data[i];
      }

      return subtraction;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to subtract or two matrices with same order");
    }
  }

  /**
   * Returns Eigen::Matrix as the subtraction of a symmetric matrix and a Eigen::Matrix.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  inline Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>
  operator-(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) const
  {
    if (other.rows() == m_order && other.cols() == m_order && m_data)
    {
      Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> subtraction(other.rows(), other.cols());

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < m_order; ++j)
        {
          subtraction(i, j) = (*this)(i, j) - other(i, j);
        }
      }

      return subtraction;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to subtract or two matrices with same order");
    }
  }

  /**
   * Returns SymMat as the multiplication of two symmetric matrices.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> operator*(const SymMat &other) const
  {
    if (m_order == other.m_order && m_data && other.m_data)
    {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> product(m_order, m_order);

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < m_order; ++j)
        {
          Scalar sum = 0;

          for (int k = 0; k < m_order; ++k)
          {
            sum += (*this)(i, k) * other(k, j);
          }

          product(i, j) = sum;
        }
      }

      return product;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to multiply or two compatible ones");
    }
  }

  /**
   * Returns Eigen::Matrix as the multiplication of a symmetric matrix and a Eigen::Matrix, in
   * this order.
   * 
   * Throws an exception if their data is invalid or if they don't have compatible orders based on 
   * matrix multiplication rules.
   */
  inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  operator*(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &other) const
  {
    if (m_order == other.rows() && m_data)
    {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> product(m_order, other.cols());

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < other.cols(); ++j)
        {
          Scalar sum = 0;

          for (int k = 0; k < m_order; ++k)
          {
            sum += (*this)(i, k) * other(k, j);
          }

          product(i, j) = sum;
        }
      }

      return product;
    }
    else
    {
      throw std::invalid_argument("Expected two valid matrices to multiply or two compatible ones");
    }
  }

  /**
   * Returns a reference to the coefficient at the given indices. The resulting value can be modified.
   * 
   * Throws an exception if the given indices are invalid.
   */
  inline Scalar &operator()(Eigen::Index row, Eigen::Index col)
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      row++;
      col++;

      if (row > col)
      {
        return m_data[(col + row * (row - 1) / 2) - 1];
      }

      return m_data[(row + col * (col - 1) / 2) - 1];
    }
    else
    {
      throw std::range_error("Matrix indices out of range");
    }
  }

  /**
   * Returns the coefficient at the given indices. The resulting value cannot be modified.
   * 
   * Throws an exception if the given indices are invalid.
   */
  inline const Scalar &operator()(Eigen::Index row, Eigen::Index col) const
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      row++;
      col++;

      if (row > col)
      {
        return m_data[(col + row * (row - 1) / 2) - 1];
      }

      return m_data[(row + col * (col - 1) / 2) - 1];
    }
    else
    {
      throw std::range_error("Matrix indices out of range");
    }
  }

  /**
   * Returns true if all coefficients of the two symmetric matrices are equal by the
   * operator== of the Scalar type and both matrices are of the same order.
   */
  inline bool operator==(const SymMat &other) const
  {
    bool areEqual = m_order == other.m_order;

    for (int i = arraySize() - 1; areEqual && i >= 0; --i)
    {
      areEqual = m_data[i] == other.m_data[i];
    }

    return areEqual;
  }

  /**
   * Returns the order of the symmetric matrix.
   */
  int order() const
  {
    return m_order;
  }

private:
  /**
   * Reserves memory for the coefficients of the symmetric matrix.
   * 
   * m_order needs to be initialized before usage.
   */
  void initArray()
  {
    if (m_order > 0)
    {
      m_data = new Scalar[arraySize()];
    }
  }

  /**
   * Returns the number of values actually being stored by the symmetric matrix.
   */
  int arraySize() const
  {
    return m_order * (m_order + 1) / 2;
  }

private:
  // The values of the symmetric matrix are stored by the Packed Storage method (http://www.netlib.org/lapack/lug/node123.html)
  Scalar *m_data;
  int m_order;
};
