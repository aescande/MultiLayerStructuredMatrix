/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/SimpleMatrix.h>

namespace mls
{
DiagonalMatrix::DiagonalMatrix(const VectorConstRef & d, bool copy)
: shape_(static_cast<int>(d.size()), static_cast<int>(d.size()), 0, 0)
{
  if(copy)
    diag_.reset(new internal::SimpleStorageDenseMat(d));
  else
    diag_.reset(new internal::SimpleStorageDenseConstRef(d));
}

DiagonalMatrix::DiagonalMatrix(const VectorRef & d, NonConstRef_t)
: shape_(d.size(), d.size(), 0, 0), diag_(new internal::SimpleStorageDenseNonConstRef(d))
{}

DenseMatrix::DenseMatrix(const MatrixConstRef & M, bool copy)
: shape_(static_cast<int>(M.rows()), static_cast<int>(M.cols()))
{
  if(copy)
    mat_.reset(new internal::SimpleStorageDenseMat(M));
  else
    mat_.reset(new internal::SimpleStorageDenseConstRef(M));
}

DenseMatrix::DenseMatrix(const MatrixRef & M, NonConstRef_t)
: shape_(M.rows(), M.cols()), mat_(new internal::SimpleStorageDenseNonConstRef(M))
{}
} // namespace mls