/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/internal/Shape.h>

#include <assert.h>
#include <utility>

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
  class IteratorBase
  {
  public:
    IteratorBase(int l) : l_(l) {}
    virtual int val() const { return -1; }
    virtual void next(){};
    int line() const { return l_; }

  protected:
    int l_;
  };

  template<bool Row>
  class Line
  {
  public:
    class Iterator
    {
    public:
      Iterator & operator++()
      {
        it_->next();
        return *this;
      }
      int operator*() const { return it_->val(); }

      friend bool operator==(const Iterator & a, const Iterator & b)
      {
        return a.it_->line() == b.it_->line() && *a == *b;
      };
      friend bool operator!=(const Iterator & a, const Iterator & b) { return !(a == b); };

    private:
      Iterator(int l) : it_(new IteratorBase(l)) {}
      Iterator(const StorageScheme & s, int l)
      {
        if constexpr(Row)
          it_ = s.rowIterator(l);
        else
          it_ = s.colIterator(l);
      }
      std::unique_ptr<IteratorBase> it_;
      friend class Row;
      friend class Col;
    };

    Iterator begin() const { return {s_, l_}; }
    Iterator end() const { return {l_}; }

  private:
    Line(const StorageScheme & s, int l) : l_(l), s_(s) {}
    int l_;
    const StorageScheme & s_;
    friend class StorageScheme;
  };

  using Row = Line<true>;
  using Col = Line<false>;

  StorageScheme(SymmetricStorage symmetric) : shape_(nullptr), symmetric_(symmetric) {}

  void setShape(const ShapeBase & shape)
  {
    assert(!isSymmetric() || shape_->rows() == shape_->cols());
    shape_ = &shape;
    processShape_();
  }

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

  virtual std::unique_ptr<IteratorBase> rowIterator(int r) const = 0;
  virtual std::unique_ptr<IteratorBase> colIterator(int c) const = 0;

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
  class RowIterator : public StorageScheme::IteratorBase
  {
  public:
    RowIterator(int r, const DenseShape & s) : IteratorBase(r), v_(r), s_(s), last_((s_.cols() - 1) * s_.rows()) {}
    int val() const override { return v_; }
    void next() override
    {
      if(v_ != -1)
      {
        if(v_ < last_)
          v_ += s_.rows();
        else
          v_ = -1;
      }
    };

  private:
    int v_;
    const DenseShape & s_;
    int last_;
  };

  class ColIterator : public StorageScheme::IteratorBase
  {
  public:
    ColIterator(int c, const DenseShape & s) : IteratorBase(c), v_(c * s.rows()), s_(s) {}
    int val() const override { return v_; }
    void next() override
    {
      if(v_ != -1)
      {
        if(v_ < s_.cols() - 1)
          ++v_;
        else
          v_ = -1;
      }
    };

  private:
    int v_;
    const DenseShape & s_;
  };
  DenseStorageScheme() : StorageScheme(SymmetricStorage::None) {}
  int size() const override { return shape_->cols() * shape_->rows(); }
  std::unique_ptr<IteratorBase> rowIterator(int r) const override
  {
    return std::unique_ptr<IteratorBase>(new RowIterator(r, shape()));
  }
  std::unique_ptr<IteratorBase> colIterator(int c) const override
  {
    return std::unique_ptr<IteratorBase>(new ColIterator(c, shape()));
  }

protected:
  void processShape_() override { assert(dynamic_cast<const DenseShape *>(shape_)); }

  bool isStored_(int r, int c) const override { return true; };

  std::pair<int, bool> index_(int r, int c) const override { return {c * shape_->cols() + r, false}; }

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
 * | B21 B32 B43 B45  0   0   0  |
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

protected:
  void processShape_() override
  {
    assert(dynamic_cast<const BandShape *>(shape_));
    const auto & s = shape();
    assert((!isSymmetric() || s.lowerBandwidth() == s.upperBandwidth())
           && "Symmetric band matrix should have the same lower and upper bandwidth");

    c0_ = std::max(0, -int(s.lowerBandwidth()));
    cn_ = std::min(s.cols(), s.rows() + int(s.upperBandwidth()));
    if(isSymmetric())
      b_ = int(shape().upperBandwidth()) + 1;
    else
      b_ = int(shape().bandwidth());
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

    switch(symmetric_)
    {
      case SymmetricStorage::None:
        return {r + (c - c0_) * b_, false};
      case SymmetricStorage::Lower:
        if(r >= c)
          return {r + c * b_, false};
        else
          return {c + r * b_, true};
      case SymmetricStorage::Upper:
        if(r <= c)
          return {r + c * b_, false};
        else
          return {c + r * b_, true};
      default:
        assert(false);
    }
  }

private:
  const BandShape & shape() const { return static_cast<const BandShape &>(*shape_); }

  int c0_; // first non-zero column
  int cn_; // first zero column after the non-zero ones
  int b_;  // effective bandwith from the storage viewpoint
};

} // namespace mls::internal