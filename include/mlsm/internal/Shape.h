/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/internal/LineIterator.h>
#include <mlsm/internal/Size.h>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace mls::internal
{

enum class ShapeType
{
  Empty = 0,
  Band,
  Dense,
  Sparse,
  Undefined // Not known or not computable
};

class ShapeBase;
using ShapePtr = std::unique_ptr<ShapeBase>;

template<>
struct LineIteratorTraits<int>
{
  static constexpr int End = -1;
};

/** Describe the size of a matrix and its non-zero pattern, i.e. where the (possibly) non-zero
 * elements can be found.
 */
class ShapeBase
{
public:
  template<bool Row>
  using Line = Line<int, ShapeBase, Row>;
  using Row = Line<true>;
  using Col = Line<false>;

  ShapeBase(int rows, int cols) : rows_(rows), cols_(cols) { assert(rows >= 0 && cols >= 0); }

  virtual ShapeType type() const = 0;
  virtual ShapePtr copy() const = 0;
  virtual ShapePtr transposed() const = 0;

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int minDim() const { return std::min(rows_, cols_); }
  int maxDim() const { return std::max(rows_, cols_); }

  bool checkRowIndex(int r) const { return r >= 0 && r < rows_; }
  bool checkColIndex(int c) const { return c >= 0 && c < cols_; }
  bool checkIndices(int r, int c) const { return checkRowIndex(r) && checkColIndex(c); }

  /** Number of non-zero on row \p r.*/
  int rowNNZ(int r) const
  {
    assert(checkRowIndex(r));
    return rowNNZ_(r);
  }
  /** Number of non-zero on col \p c.*/
  int colNNZ(int c) const
  {
    assert(checkColIndex(c));
    return colNNZ_(c);
  }

  Row row(int r) const
  {
    assert(checkRowIndex(r));
    return {*this, r};
  }
  Col col(int c) const
  {
    assert(checkColIndex(c));
    return {*this, c};
  }

  typename Row::UnderlyingItPtr rowIterator(int r) const
  {
    assert(checkRowIndex(r));
    return rowIterator_(r);
  }
  typename Col::UnderlyingItPtr colIterator(int c) const
  {
    assert(checkColIndex(c));
    return colIterator_(c);
  }

protected:
  virtual int rowNNZ_(int r) const = 0;
  virtual int colNNZ_(int c) const = 0;
  virtual typename Row::UnderlyingItPtr rowIterator_(int r) const = 0;
  virtual typename Col::UnderlyingItPtr colIterator_(int c) const = 0;

private:
  int rows_;
  int cols_;
};

/** The shape of a zero matrix*/
class EmptyShape : public ShapeBase
{
public:
  EmptyShape(int rows, int cols) : ShapeBase(rows, cols) {}
  ShapeType type() const override { return ShapeType::Empty; }

  ShapePtr copy() const override { return ShapePtr(new EmptyShape(rows(), cols())); }

  ShapePtr transposed() const override { return ShapePtr(new EmptyShape(cols(), rows())); }

protected:
  int rowNNZ_(int r) const override { return 0; }
  int colNNZ_(int c) const override { return 0; }
  typename Row::UnderlyingItPtr rowIterator_(int r) const override { return std::make_unique<Row::UnderlyingIt>(r); };
  typename Col::UnderlyingItPtr colIterator_(int c) const override { return std::make_unique<Col::UnderlyingIt>(c); };
};

/** Describe a matrix band structure, i.e. non-zero are the elements a_{i,j} such that
 * i - l <= j <= i + u where l and u are the lower and upper bandwith (said in another way
 * a_{i,j} = 0 if j < i - l or j > i + u).
 * Band matrix includes diagonal, tridiagonal, upper and lower triangular, upper and lower
 * Hessenberg matrices, ...
 *
 * See also https://en.wikipedia.org/wiki/Band_matrix
 */
class BandShape : public ShapeBase
{
public:
  BandShape(int rows, int cols, Size lowerBandwidth, Size upperBandwidth)
  : ShapeBase(rows, cols), lowerBandwidth_(lowerBandwidth), upperBandwidth_(upperBandwidth)
  {
    assert(lowerBandwidth + upperBandwidth >= 0);
  }

  ShapeType type() const override { return ShapeType::Band; }
  ShapePtr copy() const override { return ShapePtr(new BandShape(rows(), cols(), lowerBandwidth_, upperBandwidth_)); }

  ShapePtr transposed() const override
  {
    return ShapePtr(new BandShape(cols(), rows(), upperBandwidth_, lowerBandwidth_));
  }

  bool isNonZero(int r, int c) const
  {
    assert(checkIndices(r, c));
    return c >= r - lowerBandwidth_ && c <= r + upperBandwidth_;
  }

  Size lowerBandwidth() const { return lowerBandwidth_; }
  Size upperBandwidth() const { return upperBandwidth_; }
  Size bandwidth() const { return lowerBandwidth_ + upperBandwidth_ + 1; }

  bool isDiagonal() const { return lowerBandwidth_ == 0 && upperBandwidth_ == 0; }
  bool isLowerBidiagonal() const { return lowerBandwidth_ == 1 && upperBandwidth_ == 0; }
  bool isUpperBidiagonal() const { return lowerBandwidth_ == 0 && upperBandwidth_ == 1; }
  bool isTridiagonal() const { return lowerBandwidth_ == 1 && upperBandwidth_ == 1; }
  bool isLowerTriangular() const { return lowerBandwidth_ >= rows() - 1 && upperBandwidth_ == 0; }
  bool isUpperTriangular() const { return lowerBandwidth_ == 0 && upperBandwidth_ >= cols() - 1; }
  bool isLowerHessenberg() const { return lowerBandwidth_ >= rows() - 1 && upperBandwidth_ == 1; }
  bool isUpperHessenberg() const { return lowerBandwidth_ == 1 && upperBandwidth_ >= cols() - 1; }
  bool isEmpty() const { return -lowerBandwidth_ >= cols() || -upperBandwidth_ >= rows(); }
  bool isDense() const { return lowerBandwidth_ >= rows() - 1 && upperBandwidth_ >= cols() - 1; }

protected:
  int rBegin(int r) const { return std::clamp(r - lowerBandwidth_.toInt(cols()), 0, cols()); }
  int cBegin(int c) const { return std::clamp(c - upperBandwidth_.toInt(rows()), 0, rows()); }
  int rEnd(int r) const { return std::clamp(r + upperBandwidth_.toInt(cols()) + 1, 0, cols()); }
  int cEnd(int c) const { return std::clamp(c + lowerBandwidth_.toInt(rows()) + 1, 0, rows()); }
  int rowNNZ_(int r) const override { return rEnd(r) - rBegin(r); }
  int colNNZ_(int c) const override { return cEnd(c) - cBegin(c); }

  typename Row::UnderlyingItPtr rowIterator_(int r) const override
  {
    return Row::UnderlyingItPtr(new ContinuousLineIterator<int>(r, rBegin(r), rEnd(r)));
  }
  typename Col::UnderlyingItPtr colIterator_(int c) const override
  {
    return Row::UnderlyingItPtr(new ContinuousLineIterator<int>(c, cBegin(c), cEnd(c)));
  };

private:
  Size lowerBandwidth_;
  Size upperBandwidth_;
};

/** Shape for general dense matrices*/
class DenseShape : public ShapeBase
{
public:
  DenseShape(int rows, int cols) : ShapeBase(rows, cols) {}
  ShapeType type() const override { return ShapeType::Dense; }
  ShapePtr copy() const override { return ShapePtr(new DenseShape(rows(), cols())); }

  ShapePtr transposed() const override { return ShapePtr(new DenseShape(cols(), rows())); }

protected:
  int rowNNZ_(int r) const override { return cols(); }
  int colNNZ_(int c) const override { return rows(); }
  typename Row::UnderlyingItPtr rowIterator_(int r) const override
  {
    return Row::UnderlyingItPtr(new ContinuousLineIterator<int>(r, 0, cols()));
  }
  typename Col::UnderlyingItPtr colIterator_(int c) const override
  {
    return Col::UnderlyingItPtr(new ContinuousLineIterator<int>(c, 0, rows()));
  }
};

/** Shape for general sparse matrices*/
class SparseShape : public ShapeBase
{
public:
  // SparseShape(int rows, int cols) : ShapeBase(rows, cols) {}
  ShapeType type() const override { return ShapeType::Sparse; }
  // ShapePtr copy() const override { return ShapePtr(new SparseShape(rows(),
  // cols())); }

  // TODO
};

inline ShapeType mult(ShapeType lhs, ShapeType rhs)
{
  constexpr auto E = ShapeType::Empty;
  constexpr auto B = ShapeType::Band;
  constexpr auto D = ShapeType::Dense;
  constexpr auto S = ShapeType::Sparse;
  constexpr auto U = ShapeType::Undefined;
  constexpr int count = 5; // total number of enum types

  // clang-format off
  ShapeType table[] = {/*     | E | B | D | S | U */
                       /* E*/   E,  E,  E,  E,  E,
                       /* B*/   E,  B,  D,  U,  U,
                       /* D*/   E,  D,  D,  D,  D,
                       /* S*/   E,  U,  D,  S,  U,
                       /* U*/   E,  U,  D,  U,  U};
  // clang-format on

  return table[static_cast<int>(lhs) * count + static_cast<int>(rhs)];
}

inline ShapePtr mult(const ShapeBase & lhs, const ShapeBase & rhs)
{
  assert(lhs.type() != ShapeType::Undefined);
  assert(rhs.type() != ShapeType::Undefined);
  assert(lhs.cols() == rhs.rows());
  int rows = lhs.rows();
  int cols = rhs.cols();

  // Table of cases
  //    | E | B | D | S
  //  E | 1 | 1 | 1 | 1
  //  B | 1 | 3 | 2 | X
  //  D | 1 | 2 | 2 | 2
  //  S | 1 | X | 2 | X

  // Case 1
  if(lhs.type() == ShapeType::Empty || rhs.type() == ShapeType::Empty)
    return std::make_unique<EmptyShape>(rows, cols);

  // Case 2
  if(lhs.type() == ShapeType::Dense || rhs.type() == ShapeType::Dense)
    return std::make_unique<DenseShape>(rows, cols);

  // Case 3
  if(lhs.type() == ShapeType::Band && rhs.type() == ShapeType::Band)
  {
    const auto & l = static_cast<const BandShape &>(lhs);
    const auto & r = static_cast<const BandShape &>(rhs);
    return std::make_unique<BandShape>(rows, cols, l.lowerBandwidth() + r.lowerBandwidth(),
                                       l.upperBandwidth() + r.upperBandwidth());
  }

  throw std::runtime_error("[mult(ShapeBase, ShapeBase)] Non implemented cases.");
}

inline ShapeType add(ShapeType lhs, ShapeType rhs)
{
  constexpr auto E = ShapeType::Empty;
  constexpr auto B = ShapeType::Band;
  constexpr auto D = ShapeType::Dense;
  constexpr auto S = ShapeType::Sparse;
  constexpr auto U = ShapeType::Undefined;
  constexpr int count = 5; // total number of enum types

  // clang-format off
  ShapeType table[] = {/*     | E | B | D | S | U */
                       /* E*/   E,  B,  D,  S,  U,
                       /* B*/   B,  B,  D,  U,  U,
                       /* D*/   D,  D,  D,  D,  D,
                       /* S*/   S,  U,  D,  S,  U,
                       /* U*/   U,  U,  D,  U,  U};
  // clang-format on

  return table[static_cast<int>(lhs) * count + static_cast<int>(rhs)];
}

inline ShapePtr add(const ShapeBase & lhs, const ShapeBase & rhs)
{
  assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
  int rows = lhs.rows();
  int cols = lhs.cols();

  // Table of cases
  //    | E | B | D | S
  //  E | 1 | 1 | 1 | 1
  //  B | 1 | 3 | 2 | X
  //  D | 1 | 2 | 2 | 2
  //  S | 1 | X | 2 | X

  // Case 1
  if(lhs.type() == ShapeType::Empty)
    return rhs.copy();
  if(rhs.type() == ShapeType::Empty)
    return lhs.copy();

  // Case 2
  if(lhs.type() == ShapeType::Dense || rhs.type() == ShapeType::Dense)
    return std::make_unique<DenseShape>(rows, cols);

  // Case 3
  if(lhs.type() == ShapeType::Band && rhs.type() == ShapeType::Band)
  {
    const auto & l = static_cast<const BandShape &>(lhs);
    const auto & r = static_cast<const BandShape &>(rhs);
    return std::make_unique<BandShape>(rows, cols, std::max(l.lowerBandwidth(), r.lowerBandwidth()),
                                       std::max(l.upperBandwidth(), r.upperBandwidth()));
  }

  throw std::runtime_error("[add(ShapeBase, ShapeBase)] Non implemented cases.");
}

} // namespace mls::internal