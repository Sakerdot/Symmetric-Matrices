#include <stdexcept>
#include <Eigen/Dense>

template <typename _Scalar>
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
  SymMat(const Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) : m_order(other.rows())
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
   * Operator to add two symmetric matrices.
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
   * Operator to add a symmetric matrix and a Eigen::Matrix.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  inline Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols>
  operator+(const Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) const
  {
    if (other.rows() == m_order && other.cols() == m_order && m_data)
    {
      Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols> addition(other.rows(), other.cols());

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
   * Operator to subtract two symmetric matrices.
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
   * Operator to subtract a symmetric matrix and a Eigen::Matrix.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  template <int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  inline Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols>
  operator-(const Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other) const
  {
    if (other.rows() == m_order && other.cols() == m_order && m_data)
    {
      Eigen::Matrix<_Scalar, Rows, Cols, Options, MaxRows, MaxCols> subtraction(other.rows(), other.cols());

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
   * Operator to multiply two symmetric matrices.
   * 
   * Throws an exception if their data is invalid or if they don't have the same order.
   */
  inline Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> operator*(const SymMat &other) const
  {
    if (m_order == other.m_order && m_data && other.m_data)
    {
      Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> product(m_order, m_order);

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < m_order; ++j)
        {
          _Scalar sum = 0;

          for (int k = 0; k < m_order; ++k)
          {
            auto a = (*this)(i, k);
            auto b = other(k, j);
            sum += a * b;
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
   * Operator to multiply a symmetric matrix and a Eigen::Matrix.
   * 
   * Throws an exception if their data is invalid or if they don't have compatible orders based on 
   * matrix multiplication rules.
   */
  inline Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>
  operator*(const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> &other) const
  {
    if (m_order == other.rows() && m_data)
    {
      Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> product(m_order, other.cols());

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = 0; j < other.cols(); ++j)
        {
          _Scalar sum = 0;

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
   * Operator to access a value of a symmetric matrix. The resulting value can be modified.
   * 
   * Throws an exception if the given indices are invalid.
   */
  inline _Scalar &operator()(Eigen::Index row, Eigen::Index col)
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      if (row > col)
      {
        return m_data[col * (m_order - 1) + row];
      }

      return m_data[row * (m_order - 1) + col];
    }
    else
    {
      throw std::range_error("Matrix indices out of range");
    }
  }

  /**
   * Operator to access a value of a symmetric matrix. The resulting value cannot be modified.
   * 
   * Throws an exception if the given indices are invalid.
   */
  inline _Scalar &operator()(Eigen::Index row, Eigen::Index col) const
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      if (row > col)
      {
        return m_data[col * (m_order - 1) + row];
      }

      return m_data[row * (m_order - 1) + col];
    }
    else
    {
      throw std::range_error("Matrix indices out of range");
    }
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
   * Reserves memory for the values of the symmetric matrix.
   * 
   * m_order needs to be initialized before usage.
   */
  void initArray()
  {
    if (m_order > 0)
    {
      m_data = new _Scalar[arraySize()];
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
  // The values of the symmetric matrix are stored sequentially.
  _Scalar *m_data;
  int m_order;
};
