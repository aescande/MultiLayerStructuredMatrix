/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/internal/Shape.h>

#include <assert.h>
#include <utility>
#include <vector>

namespace mls::internal
{
enum class SymmetricStorage
{
  None,
  Lower,
  Upper
};

class StorageScheme
{
public:
  /** A triplet used as value for a dereferenced iterator on a line*/
  struct LineIterValue
  {
    int i = -1;      // Place of the element in the line
    int idx = 0;     // Index of the element in the storage
    bool tr = false; // Need to transpose the element

    friend bool operator==(const LineIterValue & lhs, const LineIterValue & rhs)
    {
      return std::tie(lhs.i, lhs.idx, lhs.tr) == std::tie(rhs.i, rhs.idx, rhs.tr);
    }
  };

  /** A default implementation of LineIteratorBase that uses the line iterator of the shape for
   * getting the place of the element in the line and retrieves the index in the storage (and need
   * to transpose) by simply calling the \c index function of the storage scheme.
   * 
   * This might not be efficient for some storage scheme, in which case a dedicated implementation
   * should be used.
   */
  template<bool Row>
  class LineIterator : public Line<LineIterValue, StorageScheme, Row>::UnderlyingIt
  {
  public:
    using Base = typename Line<LineIterValue, StorageScheme, Row>::UnderlyingIt;
    LineIterator(int l, const StorageScheme & s)
    : Base(l), shapeIt_(shapeLine(l, s).begin()), shapeItEnd_(shapeLine(l, s).end()), scheme_(s)
    {
      assign();
    }
    LineIterValue val() const override { return v_; }
    void next() override
    {
      if(++shapeIt_ != shapeItEnd_)
        assign();
      else
        v_ = {};
    };

  private:
    const auto & shapeLine(int l, const StorageScheme & s)
    {
      if constexpr(Row)
        return s.shape().row(l);
      else
        return s.shape().col(l);
    }
    void assign()
    {
      v_.i = *shapeIt_;
      if constexpr(Row)
        std::tie(v_.idx, v_.tr) = scheme_.index(line(), v_.i);
      else
        std::tie(v_.idx, v_.tr) = scheme_.index(v_.i, line());
    }
    typename ShapeBase::Line<Row>::Iterator shapeIt_;
    typename ShapeBase::Line<Row>::Iterator shapeItEnd_;
    const StorageScheme & scheme_;
    LineIterValue v_;
  };

  using Row = Line<LineIterValue, StorageScheme, true>;
  using Col = Line<LineIterValue, StorageScheme, false>;
  using RowIterator = LineIterator<true>;
  using ColIterator = LineIterator<false>;

  StorageScheme(SymmetricStorage symmetric) : shape_(nullptr), symmetric_(symmetric) {}

  void setShape(const ShapeBase & shape)
  {
    assert(!isSymmetric() || shape.rows() == shape.cols());
    shape_ = &shape;
    processShape_();
  }
  const ShapeBase & shape() const { return *shape_; }

  /** Return true if stored.*/
  virtual bool isStored(int r, int c) const
  {
    assert(shape_->checkIndices(r, c));
    return isStored_(r, c);
  }
  /**Return a pair (index, transpose), where index is the place of the block (r,c) in the storage
   * vector and transpose indicates if the stored matrix needs to be transposed.
   * Return index = -1 if the block is a non-stored empty matrix.
   */
  virtual std::pair<int, bool> index(int r, int c) const
  {
    assert(shape_->checkIndices(r, c));
    return index_(r, c);
  }

  virtual int size() const = 0;

  Row row(int r) const { return {*this, r}; }
  Col col(int c) const { return {*this, c}; }

  virtual typename Row::UnderlyingItPtr rowIterator(int r) const { return std::make_unique<RowIterator>(r, *this); }
  virtual typename Col::UnderlyingItPtr colIterator(int c) const { return std::make_unique<ColIterator>(c, *this); }

protected:
  bool isSymmetric() const { return symmetric_ != SymmetricStorage::None; }
  virtual void processShape_() = 0; // Check the shape and make some storage-specific processes.
  virtual bool isStored_(int r, int c) const = 0;
  virtual std::pair<int, bool> index_(int r, int c) const = 0;

  const ShapeBase * shape_;
  SymmetricStorage symmetric_;
};

class DenseStorageScheme : public StorageScheme
{
public:
  DenseStorageScheme() : StorageScheme(SymmetricStorage::None) {}
  int size() const override { return shape_->cols() * shape_->rows(); }

protected:
  void processShape_() override { assert(dynamic_cast<const DenseShape *>(shape_)); }

  bool isStored_(int r, int c) const override { return true; };

  std::pair<int, bool> index_(int r, int c) const override { return {c * shape_->rows() + r, false}; }

private:
  const DenseShape & shape() const { return static_cast<const DenseShape &>(*shape_); }
};

/** Storage for band matrices
 *
 * Given a matrix (here 5 by 9 with lower- and upperbandwidth 1 and 2
 * | B11 B12 B13  0   0   0   0   0   0  |
 * | B21 B22 B23 B24  0   0   0   0   0  |
 * |  0  B32 B33 B34 B35  0   0   0   0  |
 * |  0   0  B43 B44 B45 B46  0   0   0  |
 * |  0   0   0  B54 B55 B56 B57  0   0  |
 * storage is done column by column in a bandwidth x n matrix, where n is the number of non-zero
 * columns, that is
 * |  0   0  B13 B24 B35 B46 B57 |
 * |  0  B12 B23 B34 B45 B56  0  |
 * | B11 B22 B33 B44 B55  0   0  |
 * | B21 B32 B43 B54  0   0   0  |
 * and represented in a column-major way i.e
 * (0, 0, B12, B21, 0, B12, B22, B32, B13, B23, B33, B43, B24, ...)
 *
 * For symmetric matrices only the upper or lower part is stored. Storage of a symmetric matrix S
 * is the same as the one of the non-symmetric S' obtained from S by setting the symmetric part to
 * 0.
 */
class BandStorageScheme : public StorageScheme
{
public:
  BandStorageScheme(SymmetricStorage s = SymmetricStorage::None) : StorageScheme(s) {}
  int size() const override { return (cn_ - c0_) * b_; }
  int effectiveBandwidth() const { return b_; }
  int effectiveLowerBandwidth() const { return le_; }
  int effectiveUpperBandwidth() const { return ue_; }

protected:
  void processShape_() override
  {
    assert(dynamic_cast<const BandShape *>(shape_));
    const auto & s = shape();
    assert((!isSymmetric() || s.lowerBandwidth() == s.upperBandwidth())
           && "Symmetric band matrix should have the same lower and upper bandwidth.");
    assert((!isSymmetric() || s.rows() == s.cols()));

    int r = s.rows();
    int c = s.cols();
    int lm = s.lowerBandwidth().toInt(s.maxDim());
    int um = s.upperBandwidth().toInt(s.maxDim());
    // Effective lower and upper bandwidth
    le_ = (symmetric_ == SymmetricStorage::Upper) ? 0 : std::clamp(lm, - c, r - 1);
    ue_ = (symmetric_ == SymmetricStorage::Lower) ? 0 : std::clamp(um, - r, c - 1);
    assert(!isSymmetric() || (le_ >= 0 && ue_ >= 0));

    c0_ = std::max(0, -le_);
    cn_ = std::min(s.cols(), s.rows() + ue_);
    b_ = le_ + ue_ + 1;
  }

  bool isStored_(int r, int c) const override
  {
    switch(symmetric_)
    {
      case SymmetricStorage::None:
        return shape().isNonZero(r, c);
      case SymmetricStorage::Lower:
        return r >= c && shape().isNonZero(r, c);
      case SymmetricStorage::Upper:
        return r <= c && shape().isNonZero(r, c);
      default:
        assert(false);
    }
  }

  std::pair<int, bool> index_(int r, int c) const override
  {
    if(!shape().isNonZero(r, c))
      return {-1, false};

    // We have that c-u <= r <= c+l. This is equivalent to 0 <= r-c+u <= l+u
    // We make the "coordinate change" (r, c) -> (r-c+u, c-c0)
    auto idx = [this](int r, int c) { return r - c + ue_ + (c - c0_) * b_; };
    switch(symmetric_)
    {
      case SymmetricStorage::None:
        return {idx(r, c), false};
      case SymmetricStorage::Lower:
        if(r >= c)
          return {idx(r, c), false};
        else
          return {idx(c, r), true};
      case SymmetricStorage::Upper:
        if(r <= c)
          return {idx(r, c), false};
        else
          return {idx(c, r), true};
      default:
        assert(false);
    }
  }

private:
  const BandShape & shape() const { return static_cast<const BandShape &>(*shape_); }

  int c0_ = -1; // first non-zero column
  int cn_ = -1; // first zero column after the non-zero ones
  int b_ = -1;  // effective bandwith from the storage viewpoint
  int le_ = 0;
  int ue_ = 0;
};

} // namespace mls::internal