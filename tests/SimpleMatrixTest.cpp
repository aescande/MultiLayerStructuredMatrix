/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/SimpleMatrix.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace mls;

TEST_CASE("Zero matrix")
{
  MatrixPtr M = std::make_shared<ZeroMatrix>(4, 6);
  FAST_CHECK_EQ(M->rows(), 4);
  FAST_CHECK_EQ(M->cols(), 6);
  FAST_CHECK_EQ(M->blkRows(), 1);
  FAST_CHECK_EQ(M->blkCols(), 1);
  FAST_CHECK_UNARY(M->isSimple());
  FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Empty);
  FAST_CHECK_EQ((*M)(2, 1), 0.);
  FAST_CHECK_EQ((*M)(2, 5), 0.);
}

TEST_CASE("Identity matrix")
{
  MatrixPtr M = std::make_shared<IdentityMatrix>(6);
  FAST_CHECK_EQ(M->rows(), 6);
  FAST_CHECK_EQ(M->cols(), 6);
  FAST_CHECK_EQ(M->blkRows(), 1);
  FAST_CHECK_EQ(M->blkCols(), 1);
  FAST_CHECK_UNARY(M->isSimple());
  FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Band);
  FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).lowerBandwidth(), 0);
  FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).upperBandwidth(), 0);
  FAST_CHECK_EQ((*M)(2, 1), 0.);
  FAST_CHECK_EQ((*M)(2, 5), 0.);
  FAST_CHECK_EQ((*M)(1, 1), 1.);
  FAST_CHECK_EQ((*M)(2, 2), 1.);
}

TEST_CASE("Multiple of identity matrix")
{
  MatrixPtr M = std::make_shared<MultipleOfIdentityMatrix>(6, 3.);
  FAST_CHECK_EQ(M->rows(), 6);
  FAST_CHECK_EQ(M->cols(), 6);
  FAST_CHECK_EQ(M->blkRows(), 1);
  FAST_CHECK_EQ(M->blkCols(), 1);
  FAST_CHECK_UNARY(M->isSimple());
  FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Band);
  FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).lowerBandwidth(), 0);
  FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).upperBandwidth(), 0);
  FAST_CHECK_EQ((*M)(2, 1), 0.);
  FAST_CHECK_EQ((*M)(2, 5), 0.);
  FAST_CHECK_EQ((*M)(1, 1), 3.);
  FAST_CHECK_EQ((*M)(2, 2), 3.);
}

TEST_CASE("Diagonal matrix")
{
  Eigen::VectorXd d = Eigen::VectorXd::LinSpaced(6, 0, 5);

  SUBCASE("Copy version")
  {
    MatrixPtr M = std::make_shared<DiagonalMatrix>(d, true);
    FAST_CHECK_EQ(M->rows(), 6);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Band);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).lowerBandwidth(), 0);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).upperBandwidth(), 0);
    FAST_CHECK_EQ((*M)(2, 1), 0.);
    FAST_CHECK_EQ((*M)(2, 5), 0.);
    FAST_CHECK_EQ((*M)(1, 1), 1.);
    FAST_CHECK_EQ((*M)(2, 2), 2.);
  }

  SUBCASE("Const ref version")
  {
    MatrixPtr M = std::make_shared<DiagonalMatrix>(d, false);
    d[2] = -2;
    FAST_CHECK_EQ(M->rows(), 6);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Band);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).lowerBandwidth(), 0);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).upperBandwidth(), 0);
    FAST_CHECK_EQ((*M)(2, 1), 0.);
    FAST_CHECK_EQ((*M)(2, 5), 0.);
    FAST_CHECK_EQ((*M)(1, 1), 1.);
    FAST_CHECK_EQ((*M)(2, 2), -2.);
  }

  SUBCASE("Non-const ref version")
  {
    MatrixPtr M = std::make_shared<DiagonalMatrix>(d, NonConstRef_t{});
    d[2] = 2;
    FAST_CHECK_EQ(M->rows(), 6);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Band);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).lowerBandwidth(), 0);
    FAST_CHECK_EQ(static_cast<const internal::BandShape &>(M->shape()).upperBandwidth(), 0);
    FAST_CHECK_EQ((*M)(2, 1), 0.);
    FAST_CHECK_EQ((*M)(2, 5), 0.);
    FAST_CHECK_EQ((*M)(1, 1), 1.);
    FAST_CHECK_EQ((*M)(2, 2), 2.);

    // TODO when feature is implemented: test writing into M
  }
}

TEST_CASE("Dense matrix")
{
  Eigen::MatrixXd mat(4, 6);
  // Fills matrix with elements from 0 to 23 (column by column)
  Eigen::Map<Eigen::VectorXd>(mat.data(), 24, 1).setLinSpaced(0, 23);
  
  SUBCASE("Copy version")
  {
    MatrixPtr M = std::make_shared<DenseMatrix>(mat, true);
    FAST_CHECK_EQ(M->rows(), 4);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Dense);
    FAST_CHECK_EQ((*M)(2, 1), 6);
    FAST_CHECK_EQ((*M)(2, 5), 22);
  }

  SUBCASE("Const ref version")
  {
    MatrixPtr M = std::make_shared<DenseMatrix>(mat, false);
    mat(2, 1) = -6;
    mat(2, 5) = -22;
    FAST_CHECK_EQ(M->rows(), 4);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Dense);
    FAST_CHECK_EQ((*M)(2, 1), -6);
    FAST_CHECK_EQ((*M)(2, 5), -22);
  }

  SUBCASE("Non-const ref version")
  {
    MatrixPtr M = std::make_shared<DenseMatrix>(mat, NonConstRef_t{});
    mat(2, 1) = 6;
    mat(2, 5) = 22;
    FAST_CHECK_EQ(M->rows(), 4);
    FAST_CHECK_EQ(M->cols(), 6);
    FAST_CHECK_EQ(M->blkRows(), 1);
    FAST_CHECK_EQ(M->blkCols(), 1);
    FAST_CHECK_UNARY(M->isSimple());
    FAST_CHECK_EQ(M->shape().type(), internal::ShapeType::Dense);
    FAST_CHECK_EQ((*M)(2, 1), 6);
    FAST_CHECK_EQ((*M)(2, 5), 22);

    // TODO when feature is implemented: test writing into M
  }
}