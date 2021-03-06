/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/MatrixBase.h>

namespace mls
{
class MLSM_DLLAPI SimpleMatrix : public MatrixBase
{
public:
  int rows() const override { return shape().rows(); }
  int cols() const override { return shape().cols(); }

  void updateSize() override {}

protected:
  int blkRows() const override { return 1; }
  int blkCols() const override { return 1; }

  constTransposableMatrix v_block(int r, int c) const override
  {
    assert(r == 0 && c == 0);
    return this->shared_from_this();
  };

  nonConstTransposableMatrix v_block(int r, int c) override
  {
    assert(r == 0 && c == 0);
    return this->shared_from_this();
  };
};

class MLSM_DLLAPI ZeroMatrix : public SimpleMatrix
{
public:
  ZeroMatrix(int r, int c) : shape_(r, c) {}
  const internal::ShapeBase & shape() const override { return shape_; }
  bool isAutoResizable() const override { return true; }

protected:
  double v_coeffRef(int r, int c) const override { return 0; }
  void v_autoResize(int r, int c) override { shape_ = internal::EmptyShape(r, c); }
  void toDense(MatrixRef D, bool) const override { D.setZero(); }

private:
  internal::EmptyShape shape_;
};

class MLSM_DLLAPI IdentityMatrix : public SimpleMatrix
{
public:
  IdentityMatrix(int r) : shape_(r, r, 0, 0) {}
  const internal::ShapeBase & shape() const override { return shape_; }
  bool isAutoResizable() const override { return true; }

protected:
  double v_coeffRef(int r, int c) const override { return (r == c) ? 1 : 0; }
  void v_autoResize(int r, int c) override
  {
    assert(r == c);
    shape_ = internal::BandShape(r, c, 0, 0);
  }
  void toDense(MatrixRef D, bool) const override
  {
    assert(D.rows() == D.cols());
    D.setIdentity();
  }

private:
  internal::BandShape shape_;
};

class MLSM_DLLAPI MultipleOfIdentityMatrix : public SimpleMatrix
{
public:
  MultipleOfIdentityMatrix(int r, double a) : shape_(r, r, 0, 0), a_(a) {}
  const internal::ShapeBase & shape() const override { return shape_; }
  bool isAutoResizable() const override { return true; }

protected:
  double v_coeffRef(int r, int c) const override { return (r == c) ? a_ : 0; }
  void v_autoResize(int r, int c) override
  {
    assert(r == c);
    shape_ = internal::BandShape(r, c, 0, 0);
  }
  void toDense(MatrixRef D, bool) const override
  {
    assert(D.rows() == D.cols());
    D.setZero();
    D.diagonal().setConstant(a_);
  }

private:
  internal::BandShape shape_;
  double a_;
};

class MLSM_DLLAPI DiagonalMatrix : public SimpleMatrix
{
public:
  DiagonalMatrix(const VectorConstRef & d, bool copy);
  DiagonalMatrix(const VectorRef & d, NonConstRef_t);

  const internal::ShapeBase & shape() const override { return shape_; }
  bool isAutoResizable() const override { return false; }
  void toDense(MatrixRef D, bool) const override
  {
    assert(D.rows() == D.cols());
    D.setZero();
    D.diagonal() = diag_->data();
  }

protected:
  double v_coeffRef(int r, int c) const override
  {
    return (r == c) ? static_cast<const internal::SimpleStorageDense &>(*diag_).data()(r, 0) : 0;
  }
  void v_autoResize(int r, int c) override { assert(false); }

private:
  internal::BandShape shape_;
  std::unique_ptr<internal::SimpleStorageDense> diag_;
};

class MLSM_DLLAPI DenseMatrix : public SimpleMatrix
{
public:
  DenseMatrix(const MatrixConstRef & M, bool copy);
  DenseMatrix(const MatrixRef & M, NonConstRef_t);

  const internal::ShapeBase & shape() const override { return shape_; }
  bool isAutoResizable() const override { return false; }
  void toDense(MatrixRef D, bool transpose) const override
  {
    if(transpose)
      D = mat_->data().transpose();
    else
      D = mat_->data();
  }

protected:
  double v_coeffRef(int r, int c) const override
  {
    return static_cast<const internal::SimpleStorageDense &>(*mat_).data()(r, c);
  }
  void v_autoResize(int r, int c) override { assert(false); }

private:
  internal::DenseShape shape_;
  std::unique_ptr<internal::SimpleStorageDense> mat_;
};
} // namespace mls