/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/BlockMatrix.h>
#include <mlsm/internal/StorageScheme.h>

#include <sstream>

namespace mls
{
void BlockMatrix::setBlock(int r, int c, MatrixPtr M, bool transpose)
{
  if(!storageScheme_->isStored(r, c))
    throw std::runtime_error("[BlockMatrix::setBlock] This block is not stored and thus cannot be set.");
  if(transpose)
    setSize(r, c, M->cols(), M->rows());
  else
    setSize(r, c, M->rows(), M->cols());
  auto [i, tr] = storageScheme_->index(r, c);
  assert(!tr && "Stored matrices should not be transposed. There might be an error in the storage scheme.");
  storage_[i] = {M, transpose};
}

void BlockMatrix::setRowsOfBlock(int r, int rows)
{
  assert(r >= 0 && r < blkRows());
  assert(rows >= 0);

  rowsOfBlock_[r] = rows;
}

void BlockMatrix::setColsOfBlock(int c, int cols)
{
  assert(c >= 0 && c < blkCols());
  assert(cols >= 0);

  colsOfBlock_[c] = cols;
}

void BlockMatrix::resetRowsOfBlock(int r)
{
  assert(r >= 0 && r < blkRows());
  rowsOfBlock_[r] = -1;
}

void BlockMatrix::resetColsOfBlock(int c)
{
  assert(c >= 0 && c < blkCols());
  colsOfBlock_[c] = -1;
}

void BlockMatrix::updateSize()
{
  // Ensure each block size has been updated
  for(auto & M : storage_)
    M.matrix->updateSize();
  
  rows_ = 0;
  //for(int r = 0; r < blkRows(); ++r)
  //{
  //  const auto & M = storage_[storageScheme_->dims(r, 0).first];
  //  rows_ += M.trans ? M.matrix->cols() : M.matrix->rows();
  //}

  //cols_ = 0;
  //for(int c = 0; c < blkCols(); ++c)
  //{
  //  const auto & M = storage_[storageScheme_->dims(0, c).second];
  //  cols_ += M.trans ? M.matrix->rows() : M.matrix->cols();
  //}
}

BlockMatrix::BlockMatrix(internal::ShapePtr shape, std::unique_ptr<internal::StorageScheme> scheme)
: shape_(std::move(shape)), storageScheme_(std::move(scheme))
{
  storageScheme_->setShape(*shape_);
  storage_.resize(storageScheme_->size());
  rowsOfBlock_.resize(shape->rows(), -1);
  colsOfBlock_.resize(shape->cols(), -1);
}

void BlockMatrix::setSize(int r, int c, int rows, int cols)
{
  if(rowsOfBlock_[r] >= 0 && rowsOfBlock_[r] != rows)
  {
    std::stringstream ss;
    ss << "[BlockMatrix::setSize] Newly inserted matrix has a row size (" << rows
       << ") incompatible with the row size of other elements in the same row of blocks (" << rowsOfBlock_[r] << ")\n";
    throw std::runtime_error(ss.str());
  }
  if(colsOfBlock_[c] >= 0 && colsOfBlock_[c] != cols)
  {
    std::stringstream ss;
    ss << "[BlockMatrix::setSize] Newly inserted matrix has a column size (" << cols
       << ") incompatible with the column size of other elements in the same column of blocks (" << colsOfBlock_[c]
       << ")\n";
    throw std::runtime_error(ss.str());
  }
  rowsOfBlock_[r] = rows;
  colsOfBlock_[c] = cols;
}

constTransposableMatrix BlockMatrix::v_block(int r, int c) const
{
  auto [i, tr] = storageScheme_->index(r, c);
  if(tr)
    return static_cast<constTransposableMatrix>(storage_[i].transposed());
  else
    return static_cast<constTransposableMatrix>(storage_[i]);
}

nonConstTransposableMatrix BlockMatrix::v_block(int r, int c)
{
  auto [i, tr] = storageScheme_->index(r, c);
  if(tr)
    return storage_[i].transposed();
  else
    return storage_[i];
}

} // namespace mls