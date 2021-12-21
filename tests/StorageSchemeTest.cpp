/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/internal/StorageScheme.h>

#include <Eigen/Core>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace mls::internal;
using LIV = StorageScheme::LineIterValue;

// Generate a r x c band matrix with lower and upperbandwidth l and u and with 1 on the band
Eigen::MatrixXd bandMatrix(int r, int c, int l, int u, SymmetricStorage sym)
{
  if(l == Size::inf)
    l /= 2; // to avoid overflow
  if(u == Size::inf)
    u /= 2; // to avoid overflow
  // Create a band matrix with 0 on the band and -1 outside
  Eigen::MatrixXd B(r, c);
  B.setOnes();
  for(int i = 0; i < r; ++i)
  {
    for(int j = 0; j < c; ++j)
    {
      if(j < i - l || j > i + u)
        B(i, j) = 0;
    }
  }
  if(sym == SymmetricStorage::Lower)
  {
    for(int i = 0; i < r; ++i)
    {
      for(int j = i + 1; j < c; ++j)
        B(i, j) = 0;
    }
  }
  else if(sym == SymmetricStorage::Upper)
  {
    for(int i = 0; i < r; ++i)
    {
      for(int j = 0; j < i; ++j)
        B(i, j) = 0;
    }
  }

  return B;
}

// Based on the output of bandMatrix, generates a matrix where the values are
//  - for elements inside a band (le,ue): the storage index of this element
//  - for element outside the band: -1
// le and ue must be the effective upper and lower bandwidth, as computed by StorageScheme::setShape
Eigen::MatrixXd indexedBandMatrix(Eigen::MatrixXd B, int le, int ue, SymmetricStorage sym)
{
  const int r = B.rows();
  const int c = B.cols();

  int idx = 1;
  for(int j = 0; j < c; ++j)
  {
    // if empty col, we skip the iteration
    if(B.col(j).sum() == 0)
      continue;
    for(int i = j - ue; i <= j + le; ++i)
    {
      if(0 <= i && i < r)
        B(i, j) = idx;
      ++idx;
    }
  }
  B.array() -= 1;

  // copying the symmetric part
  if(sym == SymmetricStorage::Lower)
  {
    for(int i = 0; i < r; ++i)
    {
      for(int j = i + 1; j < c; ++j)
        B(i, j) = B(j, i);
    }
  }
  else if(sym == SymmetricStorage::Upper)
  {
    for(int i = 0; i < r; ++i)
    {
      for(int j = 0; j < i; ++j)
        B(i, j) = B(j, i);
    }
  }

  return B;
}

void testBandShape(int r, int c, int l, int u, SymmetricStorage sym)
{
  Eigen::MatrixXd B = bandMatrix(r, c, l, u, sym);
  BandShape shape(r, c, l, u);
  std::unique_ptr<StorageScheme> storage = std::make_unique<BandStorageScheme>(sym);
  storage->setShape(shape);

  // Test effective bandwidth
  int le = static_cast<const BandStorageScheme &>(*storage).effectiveLowerBandwidth();
  int ue = static_cast<const BandStorageScheme &>(*storage).effectiveUpperBandwidth();
  Eigen::MatrixXd C = bandMatrix(r, c, le, ue, sym);
  FAST_CHECK_EQ(B, C);

  Eigen::MatrixXd I = indexedBandMatrix(B, le, ue, sym);

  for(int i = 0; i < r; ++i)
  {
    for(int j = 0; j < c; ++j)
    {
      const auto & res = storage->index(i, j);
      FAST_CHECK_EQ(res.first, I(i, j));
      if(res.first >= 0)
      {
        switch(sym)
        {
          case SymmetricStorage::None:
            FAST_CHECK_UNARY_FALSE(res.second);
            break;
          case SymmetricStorage::Lower:
            FAST_CHECK_EQ(res.second, j > i);
            break;
          case SymmetricStorage::Upper:
            FAST_CHECK_EQ(res.second, j < i);
            break;
          default:
            assert(false);
        }
      }
    }
  }
}

TEST_CASE("Band storage")
{
  constexpr SymmetricStorage None = SymmetricStorage::None;
  constexpr SymmetricStorage Lower = SymmetricStorage::Lower;
  constexpr SymmetricStorage Upper = SymmetricStorage::Upper;
  constexpr int inf = Size::inf;
  for(int l = -8; l <= 8; ++l)
  {
    for(int u = -l; u <= 8; ++u)
    {
      testBandShape(7, 7, l, u, None);  // square
      testBandShape(4, 7, l, u, None);  // wide
      testBandShape(7, 4, l, u, None);  // tall
    }
  }
  for(int i = -8; i <= 8; ++i)
  {
    testBandShape(7, 7, inf, i, None); // square
    testBandShape(4, 7, inf, i, None); // wide
    testBandShape(7, 4, inf, i, None); // tall
    testBandShape(7, 7, i, inf, None);
    testBandShape(4, 7, i, inf, None);
    testBandShape(7, 4, i, inf, None); 
  }

  // Symmetric
  for(int i = 0; i < 8; ++i)
  {
    testBandShape(7, 7, i, i, Lower);
    testBandShape(7, 7, i, i, Upper);
  }
  testBandShape(7, 7, inf, inf, Lower);
  testBandShape(7, 7, inf, inf, Upper);
}

TEST_CASE("Dense storage iterators")
{
  DenseShape shape(5, 8);
  std::unique_ptr<StorageScheme> storage = std::make_unique<DenseStorageScheme>();
  storage->setShape(shape);

  auto row = storage->row(2);
  auto it = row.begin();
  FAST_CHECK_EQ(*it, LIV{0, 2, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{1, 7, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{2, 12, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{3, 17, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{4, 22, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{5, 27, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{6, 32, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{7, 37, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{});
  FAST_CHECK_EQ(it, row.end());

  int i = 0;
  int idx = 15;
  for(const auto & k : storage->col(3))
  {
    FAST_CHECK_EQ(k, LIV{i, idx, false});
    ++i;
    ++idx;
  }
}

TEST_CASE("Band storage iterators")
{
  BandShape shape(5, 9, 1, 2); // Same matrix as in doc of BandStorageScheme
  std::unique_ptr<StorageScheme> storage = std::make_unique<BandStorageScheme>();
  storage->setShape(shape);

  auto row = storage->row(3);
  auto it = row.begin();
  FAST_CHECK_EQ(*it, LIV{2, 11, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{3, 14, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{4, 17, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{5, 20, false});
  ++it;
  FAST_CHECK_EQ(*it, LIV{});
  FAST_CHECK_EQ(it, row.end());

  int i = 1;
  int idx = 12;
  for(const auto & k : storage->col(3))
  {
    FAST_CHECK_EQ(k, LIV{i, idx, false});
    ++i;
    ++idx;
  }
}