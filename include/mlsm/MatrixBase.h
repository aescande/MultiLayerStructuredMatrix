/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <mlsm/api.h>
#include <mlsm/defs.h>

#include <mlsm/ShapeDescriptor.h>
#include <mlsm/internal/Shape.h>
#include <mlsm/internal/SimpleStorage.h>
#include <mlsm/internal/StorageScheme.h>

#include <vector>

namespace mls
{
/** Dummy class used to disembiguate some function calls*/
class NonConstRef_t
{};

class MatrixBase;
using MatrixPtr = std::shared_ptr<MatrixBase>;
using MatrixConstPtr = std::shared_ptr<const MatrixBase>;

/** A pair (matrix ptr, bool) where the pointer is to a const or non-const MatrixBase and the bool
 * indicate whether the matrix is transposed or not. Used to return a submatrix.
 */
template<bool Const>
struct TransposableMatrix
{
  using PtrType = std::conditional_t<Const, MatrixConstPtr, MatrixPtr>;
  using OtherPtrType = std::conditional_t<Const, MatrixPtr, MatrixConstPtr>;

  TransposableMatrix() = default;
  TransposableMatrix(const PtrType & M, bool tr = false) : matrix(M), trans(tr) {}
  TransposableMatrix(const TransposableMatrix &) = default;
  explicit TransposableMatrix(const TransposableMatrix<!Const> & other)
  : matrix(std::const_pointer_cast<typename PtrType::element_type>(other.matrix)), trans(other.trans)
  {}

  int rows() const { return trans ? matrix->cols() : matrix->rows(); }
  int cols() const { return trans ? matrix->rows() : matrix->cols(); }

  PtrType matrix;
  bool trans;

  void transpose() { trans = !trans; }
  TransposableMatrix transposed() const { return {matrix, !trans}; }
};

using constTransposableMatrix = TransposableMatrix<true>;
using nonConstTransposableMatrix = TransposableMatrix<false>;

/** Base class for representing multi-layered structure matrices. */
class MLSM_DLLAPI MatrixBase : public std::enable_shared_from_this<MatrixBase>
{
public:
  virtual ~MatrixBase() {}

  virtual int rows() const = 0;
  virtual int cols() const = 0;

  virtual const internal::ShapeBase & shape() const = 0;

  virtual int blkRows() const = 0;
  virtual int blkCols() const = 0;

  bool isSimple() const { return blkRows() == 1 && blkCols() == 1; }

  double operator()(int r, int c) const
  {
    assert(r >= 0 && r < rows() && c >= 0 && c < cols());
    return v_coeffRef(r, c);
  };
  //double & coeffRef(int r, int c);

  constTransposableMatrix block(int r, int c) const;
  nonConstTransposableMatrix block(int r, int c);

  virtual void updateSize() = 0;

protected:
  virtual constTransposableMatrix v_block(int r, int c) const = 0;
  virtual nonConstTransposableMatrix v_block(int r, int c) = 0;
  virtual bool v_isAutoResizable() const = 0;
  virtual double v_coeffRef(int r, int c) const = 0;
  //virtual double & v_coeffRef(int r, int c) = 0;
};
} // namespace mls
