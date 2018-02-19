#include <iostream>
#include <type_traits>
#include <Eigen/Dense>

template <typename _Scalar>
class SymMat
{
public:
  SymMat() : m_data(nullptr), m_order(0) {}

  SymMat(const SymMat &other) : m_order(other.m_order)
  {
    if (m_order != other.m_order)
    {
      m_order = 0;
      m_data = nullptr;

      // throw exception
    }
    else
    {
      initArray();
      copyArray(other);
    }
  }

  SymMat(SymMat &&other) : m_order(other.m_order), m_data(other.m_data)
  {
    if (m_order != other.m_order)
    {
      m_order = 0;
      m_data = nullptr;

      // throw exception
    }
    else
    {
      other.m_data = nullptr;
    }
  }

  SymMat(int order) : m_order(order)
  {
    initArray();
  }

  template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  SymMat(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &other)
  {
    static_assert(std::is_same<_Scalar, Scalar>::value, "SymMat and Eigen::Matrix don't have the same Scalar type.");

    m_order = other.rows();

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
      // throw exception
    }
  }

  ~SymMat()
  {
    delete[] m_data;
  }

  inline SymMat operator+(const SymMat &other) const
  {
    if (m_order == other.m_order && m_data && other.m_data)
    {
      SymMat addition(m_order);

      for (int i = 0; i < m_order; ++i)
      {
        for (int j = i; j < m_order; ++j)
        {
          addition(i, j) = other(i, j) + (*this)(i, j);
        }
      }

      return addition;
    }
    else
    {
      // throw exception
    }
  }

  inline _Scalar& operator()(Eigen::Index row, Eigen::Index col)
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      if (row > col)
      {
        return m_data[(col + 1) * col / 2 + row];
      }

      return m_data[(row + 1) * row / 2 + col];
    }
    else
    {
      // throw exception
    }
  }

  inline _Scalar& operator()(Eigen::Index row, Eigen::Index col) const
  {
    if (row >= 0 && row < m_order && col >= 0 && col < m_order && m_data)
    {
      if (row > col)
      {
        return m_data[(col + 1) * col / 2 + row];
      }

      return m_data[(row + 1) * row / 2 + col];
    }
    else
    {
      // throw exception
    }
  }

private:
  void initArray()
  {
    m_data = new _Scalar[m_order * (m_order + 1) / 2];
  }

  void copyArray(const SymMat &other)
  {
    for (int i = 0; i < m_order; ++i)
    {
      for (int j = i; j < m_order; ++j)
      {
        (*this)(i, j) = other(i, j);
      }
    }
  }

private:
  _Scalar *m_data;
  int m_order;
};
