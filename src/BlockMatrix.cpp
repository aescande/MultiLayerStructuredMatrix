/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/BlockMatrix.h>
#include <mlsm/SimpleMatrix.h>
#include <mlsm/internal/StorageScheme.h>

#include <set>
#include <sstream>

namespace mls
{
void BlockMatrix::setBlock(int r, int c, MatrixPtr M, bool transpose)
{
  if(!storageScheme_->isStored(r, c))
    throw std::runtime_error("[BlockMatrix::setBlock] This block is not stored and thus cannot be set.");
  
  M->updateSize();
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
  rowsOfBlock_[r] = undef;
}

void BlockMatrix::resetColsOfBlock(int c)
{
  assert(c >= 0 && c < blkCols());
  colsOfBlock_[c] = undef;
}

void BlockMatrix::updateSize()
{
  // Ensure each block size has been updated
  for(auto & M : storage_)
  {
    if(M.matrix)
      M.matrix->updateSize();
  }

  auto comp = [](const auto & lhs, const auto & rhs) {
    return std::minmax(lhs.first, lhs.second) < std::minmax(rhs.first, rhs.second);
  };
  std::set<std::pair<int, int>, decltype(comp)> toBeResized(comp);
  for(int c = 0; c < blkRows(); ++c)
  {
    for(const auto & e : storageScheme_->col(c))
    {
      const auto [r, idx, tr] = e;
      const auto & M = storage_[idx];
      int rm = tr ? M.cols() : M.rows();
      int cm = tr ? M.rows() : M.cols();
      if(rowsOfBlock_[r] == undef)
        rowsOfBlock_[r] = rm;
      else
      {
        if(rowsOfBlock_[r] != rm)
        {
          if(M.matrix->isAutoResizable())
            toBeResized.insert({r, c});
          else
          {
            std::stringstream ss;
            ss << "[BlockMatrix::updateSize] Invalid row size for block (" << r << ", " << c << "). Expected "
               << rowsOfBlock_[r] << ", but got " << rm << ".\n";
            throw std::runtime_error(ss.str());
          }
        }
      }
      if(colsOfBlock_[c] == undef)
        rowsOfBlock_[c] = cm;
      else
      {
        if(colsOfBlock_[c] != cm)
        {
          if(M.matrix->isAutoResizable())
            toBeResized.insert({r, c});
          else
          {
            std::stringstream ss;
            ss << "[BlockMatrix::updateSize] Invalid column size for block (" << r << ", " << c << "). Expected "
               << colsOfBlock_[c] << ", but got " << cm << ".\n";
            throw std::runtime_error(ss.str());
          }
        }
      }
    }
  }

  rows_ = 0;
  for(int r = 0; r < blkRows(); ++r)
  {
    int rows = rowsOfBlock_[r];
    if(rows == undef)
    {
      std::stringstream ss;
      ss << "[BlockMatrix::updateSize] Size of row " << r << " was not specified.\n";
      throw std::runtime_error(ss.str());
    }
    rows_ += rows;
  }

  cols_ = 0;
  for(int c = 0; c < blkCols(); ++c)
  {
    int cols = colsOfBlock_[c];
    if(cols == undef)
    {
      std::stringstream ss;
      ss << "[BlockMatrix::updateSize] Size of column " << c << " was not specified.\n";
      throw std::runtime_error(ss.str());
    }
    cols_ += cols;
  }

  // Resizing auto-resizable matrices
  for(const auto & p : toBeResized)
  {
    auto [idx, tr] = storageScheme_->index(p.first, p.second);
    if(tr)
      storage_[idx].matrix->autoResize(colsOfBlock_[p.second], rowsOfBlock_[p.first]);
    else
      storage_[idx].matrix->autoResize(rowsOfBlock_[p.first], colsOfBlock_[p.second]);
  }
}

BlockMatrix::BlockMatrix(internal::ShapePtr shape, std::unique_ptr<internal::StorageScheme> scheme)
: shape_(std::move(shape)), storageScheme_(std::move(scheme))
{
  storageScheme_->setShape(*shape_);
  storage_.resize(storageScheme_->size());
  rowsOfBlock_.resize(shape_->rows(), -1);
  colsOfBlock_.resize(shape_->cols(), -1);
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

double BlockMatrix::v_coeffRef(int r, int c) const
{
  assert(r >= 0 && r < rows() && c >= 0 && c < cols());
  auto findIndex = [](int i, const std::vector<int> & sizes) {
    for(size_t k = 0; k < sizes.size(); ++k)
    {
      if(i < sizes[k])
        return std::make_pair(static_cast<int>(k), i);
      else
        i -= sizes[k];
    }
    assert(false);
    return std::make_pair(-1, -1);
  };
  auto [rBlk, rInBlk] = findIndex(r, rowsOfBlock_);
  auto [cBlk, cInBlk] = findIndex(c, colsOfBlock_);

  auto [idx, tr] = storageScheme_->index(rBlk, cBlk);
  if(idx == -1)
    return 0.;
  else
    return storage_[idx].transposed(tr)(rInBlk, cInBlk);
}

constTransposableMatrix BlockMatrix::v_block(int r, int c) const
{
  auto [i, tr] = storageScheme_->index(r, c);
  if(i == -1)
    return {std::make_shared<ZeroMatrix>(rowsOfBlock_[r], colsOfBlock_[c]), false};
  if(tr)
    return static_cast<constTransposableMatrix>(storage_[i].transposed());
  else
    return static_cast<constTransposableMatrix>(storage_[i]);
}

nonConstTransposableMatrix BlockMatrix::v_block(int r, int c)
{
  auto [i, tr] = storageScheme_->index(r, c);
  if(i == -1)
    return {std::make_shared<ZeroMatrix>(rowsOfBlock_[r], colsOfBlock_[c]), false};
  if(tr)
    return storage_[i].transposed();
  else
    return storage_[i];
}

void BlockMatrix::toDense(MatrixRef D, bool transpose) const
{
  int rows = 0;
  for(int r = 0; r < blkRows(); ++r)
  {
    int cols = 0;
    for(int c = 0; c < blkCols(); ++c)
    {
      block(r,c).toDense(D.block(rows, cols, rowsOfBlock_[r], colsOfBlock_[c]), transpose);
      cols += colsOfBlock_[c];
    }
    rows += rowsOfBlock_[r];
  }
}

} // namespace mls