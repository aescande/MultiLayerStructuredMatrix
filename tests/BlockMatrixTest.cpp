/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/BlockMatrix.h>
#include <mlsm/SimpleMatrix.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace mls;

TEST_CASE("Build")
{

  //      5    15     5
  // 5 |  I  |  0  | -I  |  
  //10 |  0  | M22 | M23 |
  // 5 | -I  | M32 | M33 |
  
  Eigen::MatrixXd M23(10,5);
  Eigen::Map<Eigen::VectorXd>(M23.data(), 50).setLinSpaced(0, 49);
  Eigen::MatrixXd M32(5, 15);
  Eigen::Map<Eigen::VectorXd>(M32.data(), 75).setLinSpaced(0, -74);
  Eigen::VectorXd d33 = Eigen::VectorXd::LinSpaced(5, 0, 8);

  auto M22 = std::make_shared<DiagonalBlockMatrix>(3);
  M22->setBlock(0, 0, std::make_shared<DenseMatrix>(Eigen::MatrixXd::Ones(3, 6), true));
  M22->setBlock(1, 1, std::make_shared<IdentityMatrix>(4));
  M22->setBlock(2, 2, std::make_shared<DenseMatrix>(-2*Eigen::MatrixXd::Ones(3, 5), true));

  DenseBlockMatrix M(3, 3);
  M.setBlock(0, 0, std::make_shared<IdentityMatrix>(5));
  M.setBlock(0, 1, std::make_shared<ZeroMatrix>(5, 15));
  M.setBlock(0, 2, std::make_shared<MultipleOfIdentityMatrix>(5, -1.));
  M.setBlock(1, 0, std::make_shared<ZeroMatrix>(10, 5));
  M.setBlock(1, 1, M22);
  M.setBlock(1, 2, std::make_shared<DenseMatrix>(M23, true));
  M.setBlock(2, 0, std::make_shared<MultipleOfIdentityMatrix>(5, -1.));
  M.setBlock(2, 1, std::make_shared<DenseMatrix>(M32, true));
  M.setBlock(2, 2, std::make_shared<DiagonalMatrix>(d33, true));

  M.updateSize();

  FAST_CHECK_EQ(M.rows(), 20);
  FAST_CHECK_EQ(M.cols(), 25);
  FAST_CHECK_EQ(M(0, 0), 1);
  FAST_CHECK_EQ(M(3, 3), 1);
  FAST_CHECK_EQ(M(1, 3), 0);
  FAST_CHECK_EQ(M(4, 1), 0);
  FAST_CHECK_EQ(M(6, 9), 1);
  FAST_CHECK_EQ(M(10, 13), 1);
  FAST_CHECK_EQ(M(2, 13), 0);
  FAST_CHECK_EQ(M(14, 8), 0);
  FAST_CHECK_EQ(M(15, 0), -1);
  FAST_CHECK_EQ(M(15, 8), -15);
  FAST_CHECK_EQ(M(0, 20), -1);
  FAST_CHECK_EQ(M(9, 21), 14);
  FAST_CHECK_EQ(M(18, 23), 6);

  Eigen::MatrixXd C(20, 25);
  Eigen::MatrixXd D(20, 25);
  M.toDense(D, false);

  for(int r = 0; r < 20; ++r)
  {
    for(int c = 0; c < 25; ++c)
    {
      C(r, c) = M(r, c);
    }
  }
  FAST_CHECK_EQ(C, D);
}

TEST_CASE("Symmetric storage")
{
  TriDiagonalBlockMatrix M(6, true);
  double d = 0;
  for(int i=0; i<6; ++i) 
  {
    auto Ti = std::make_shared<TriDiagonalBlockMatrix>(5, true, false);
    for(int j = 0; j < 4; ++j)
    {
      Ti->setBlock(j, j, std::make_shared<MultipleOfIdentityMatrix>(2 + j, ++d));
      Eigen::MatrixXd D(3+j,2+j);
      for(int k = 0; k < 3 + j; ++k)
        D.row(k).setConstant(-(d + k + 1));
      Ti->setBlock(j+1, j, std::make_shared<DenseMatrix>(D, true));
    }
    Ti->setBlock(4, 4, std::make_shared<MultipleOfIdentityMatrix>(6, ++d));
    M.setBlock(i, i, Ti);
    if(i < 5)
      M.setBlock(i, i + 1, std::make_shared<MultipleOfIdentityMatrix>(20, -(++d)));
  }

  M.updateSize();
  //std::cout << static_cast<MatrixBase *>(&M)->toDense() << std::endl;
  FAST_CHECK_EQ(M.rows(), 120);
  FAST_CHECK_EQ(M.cols(), 120);

  Eigen::MatrixXd C(120, 120);
  Eigen::MatrixXd D(120, 120);
  M.toDense(D, false);

  for(int r = 0; r < 120; ++r)
  {
    for(int c = 0; c < 120; ++c)
    {
      C(r, c) = M(r, c);
    }
  }
  FAST_CHECK_EQ(C, D);
  FAST_CHECK_UNARY((C - C.transpose()).isZero());
}