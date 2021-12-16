/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/MatrixBase.h>

namespace mls
{
/** Rules for the size of the blocks:
 *  - initially unspecified (-1)
 *  - first matrix to be set on a row/col imposes its size to the row/col
 *  - subsequent matrices needs to have compatible sizes if on the same row/col
 */
class MLSM_DLLAPI BlockMatrix : public MatrixBase
{
public:
  int rows() const override { return rows_; }
  int cols() const override { return cols_; }

  const internal::ShapeBase & shape() const override { return *shape_; }

  int blkRows() const override { return shape_->rows(); };
  int blkCols() const override { return shape_->cols(); };

  /** Set the block (r,c) to the given matrices. */
  void setBlock(int r, int c, MatrixPtr M, bool transpose);
  void setRowsOfBlock(int r, int rows);
  void setColsOfBlock(int c, int cols);
  void resetRowsOfBlock(int r);
  void resetColsOfBlock(int c);
  // Update the size of each row and column pf blocks
  void updateSize() override;

protected:
  BlockMatrix(internal::ShapePtr shape, std::unique_ptr<internal::StorageScheme>);

  void setSize(int r, int c, int rows, int cols);

  constTransposableMatrix v_block(int r, int c) const override;
  nonConstTransposableMatrix v_block(int r, int c) override;
  bool v_isAutoResizable() const override { return false; }

protected:
  std::unique_ptr<internal::ShapeBase> shape_;
  std::unique_ptr<internal::StorageScheme> storageScheme_;
  std::vector<nonConstTransposableMatrix> storage_;
  std::vector<int> rowsOfBlock_;
  std::vector<int> colsOfBlock_;
  int rows_; // total number of rows
  int cols_; // total number of cols
};

class MLSM_DLLAPI DenseBlockMatrix : public BlockMatrix
{
public:
  DenseBlockMatrix(int blkRows, int blkCols) : BlockMatrix(std::make_unique<internal::DenseShape>(blkRows, blkCols), std::make_unique<internal::DenseStorageScheme>())
  {}

protected:
};
}