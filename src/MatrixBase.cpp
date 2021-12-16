/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/MatrixBase.h>

namespace mls
{

TransposableMatrix<true> MatrixBase::block(int r, int c) const
{
  if(r < 0 || r >= blkRows())
    throw std::runtime_error("[MatrixBase::block] Invalid row index");
  if(c < 0 || c >= blkCols())
    throw std::runtime_error("[MatrixBase::block] Invalid column index");
  return v_block(r, c);
}

TransposableMatrix<false> MatrixBase::block(int r, int c)
{
  if(r < 0 || r >= blkRows())
    throw std::runtime_error("[MatrixBase::block] Invalid row index");
  if(c < 0 || c >= blkCols())
    throw std::runtime_error("[MatrixBase::block] Invalid column index");
  return v_block(r, c);
}

} // namespace mls
