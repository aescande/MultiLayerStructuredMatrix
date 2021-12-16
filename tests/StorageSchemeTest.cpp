/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/internal/StorageScheme.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace mls::internal;

TEST_CASE("Dense storage iterators")
{
  DenseShape shape(5, 8);
  std::unique_ptr<StorageScheme> storage = std::make_unique<DenseStorageScheme>();
  storage->setShape(shape);

  auto row = storage->row(2);
  auto it = row.begin();
  FAST_CHECK_EQ(*it, 2);
  ++it;
  FAST_CHECK_EQ(*it, 7);
  ++it;
  FAST_CHECK_EQ(*it, 12);
  ++it;
  FAST_CHECK_EQ(*it, 17);
  ++it;
  FAST_CHECK_EQ(*it, 22);
  ++it;
  FAST_CHECK_EQ(*it, 27);
  ++it;
  FAST_CHECK_EQ(*it, 32);
  ++it;
  FAST_CHECK_EQ(*it, 37);
  ++it;
  FAST_CHECK_EQ(*it, -1);
  FAST_CHECK_EQ(it, row.end());

  int i = 15;
  for(auto k : storage->col(3))
  {
    FAST_CHECK_EQ(k, i);
    ++i;
  }
}