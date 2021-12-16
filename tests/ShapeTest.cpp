/** Copyright 2021 CNRS-AIST JRL*/

#include <mlsm/internal/Shape.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace mls::internal;

TEST_CASE("Empty shape")
{
  ShapePtr s = std::make_unique<EmptyShape>(3, 5);
  FAST_CHECK_EQ(s->type(), ShapeType::Empty);
  FAST_CHECK_EQ(s->rows(), 3);
  FAST_CHECK_EQ(s->cols(), 5);
  FAST_CHECK_EQ(s->minDim(), 3);
  FAST_CHECK_EQ(s->maxDim(), 5);
  FAST_CHECK_UNARY(s->checkIndices(0, 0));
  FAST_CHECK_UNARY(s->checkIndices(1, 3));
  FAST_CHECK_UNARY(s->checkIndices(2, 4));
  FAST_CHECK_UNARY_FALSE(s->checkIndices(-1, 0));
  FAST_CHECK_UNARY_FALSE(s->checkIndices(3, 0));
  FAST_CHECK_UNARY_FALSE(s->checkIndices(0, -1));
  FAST_CHECK_UNARY_FALSE(s->checkIndices(0, 5));
}

TEST_CASE("Band shape")
{
  SUBCASE("Diagonal")
  {
    ShapePtr s = std::make_unique<BandShape>(3, 5, 0, 0);
    const auto & d = static_cast<const BandShape &>(*s);
    FAST_CHECK_EQ(s->type(), ShapeType::Band);
    FAST_CHECK_EQ(s->rows(), 3);
    FAST_CHECK_EQ(s->cols(), 5);
    FAST_CHECK_UNARY(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
    FAST_CHECK_UNARY(d.isNonZero(1, 1));
    FAST_CHECK_UNARY(d.isNonZero(2, 2));
    FAST_CHECK_UNARY_FALSE(d.isNonZero(1, 2));
    FAST_CHECK_UNARY_FALSE(d.isNonZero(2, 1));
  }

  SUBCASE("Lower bidiagonal")
  {
    BandShape d(3, 5, 1, 0);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }
  
  SUBCASE("Upper bidiagonal")
  {
    BandShape d(3, 5, 0, 1);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }

  SUBCASE("Tridiagonal")
  {
    BandShape d(3, 5, 1, 1);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }

  SUBCASE("Lower triangular")
  {
    BandShape d(3, 5, Size::inf, 0);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }
  
  SUBCASE("Upper triangular")
  {
    BandShape d(3, 5, 0, 5);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }
  SUBCASE("Lower Hessenberg")
  {
    BandShape d(3, 5, Size::inf, 1);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }

  SUBCASE("Upper Hessenberg")
  {
    BandShape d(3, 5, 1, 5);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }

  SUBCASE("Empty")
  {
    BandShape d(3, 5, -6, 8);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY(d.isEmpty());
    FAST_CHECK_UNARY_FALSE(d.isDense());
  }

  SUBCASE("Dense")
  {
    BandShape d(3, 5, 6, 8);
    FAST_CHECK_UNARY_FALSE(d.isDiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isUpperBidiagonal());
    FAST_CHECK_UNARY_FALSE(d.isTridiagonal());
    FAST_CHECK_UNARY_FALSE(d.isLowerTriangular());
    FAST_CHECK_UNARY_FALSE(d.isUpperTriangular());
    FAST_CHECK_UNARY_FALSE(d.isLowerHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isUpperHessenberg());
    FAST_CHECK_UNARY_FALSE(d.isEmpty());
    FAST_CHECK_UNARY(d.isDense());
  }
}

TEST_CASE("Dense shape")
{
  ShapePtr s = std::make_unique<DenseShape>(3, 5);
  FAST_CHECK_EQ(s->type(), ShapeType::Dense);
  FAST_CHECK_EQ(s->rows(), 3);
  FAST_CHECK_EQ(s->cols(), 5);
}

TEST_CASE("Copy")
{
  SUBCASE("Empty")
  {
    ShapePtr s = std::make_unique<EmptyShape>(3, 5);
    ShapePtr c = s->copy();
    FAST_CHECK_EQ(c->type(), ShapeType::Empty);
    FAST_CHECK_EQ(c->rows(), 3);
    FAST_CHECK_EQ(c->cols(), 5);
  }

  SUBCASE("Band")
  {
    ShapePtr s = std::make_unique<BandShape>(3, 5, 2, 1);
    ShapePtr c = s->copy();
    FAST_CHECK_EQ(c->type(), ShapeType::Band);
    FAST_CHECK_EQ(c->rows(), 3);
    FAST_CHECK_EQ(c->cols(), 5);
    const auto & b = static_cast<const BandShape &>(*c);
    FAST_CHECK_EQ(b.lowerBandwidth(), 2);
    FAST_CHECK_EQ(b.upperBandwidth(), 1);
  }

  SUBCASE("Dense")
  {
    ShapePtr s = std::make_unique<DenseShape>(3, 5);
    ShapePtr c = s->copy();
    FAST_CHECK_EQ(c->type(), ShapeType::Dense);
    FAST_CHECK_EQ(c->rows(), 3);
    FAST_CHECK_EQ(c->cols(), 5);
  }
}

TEST_CASE("Transposed")
{
  SUBCASE("Empty")
  {
    ShapePtr s = std::make_unique<EmptyShape>(3, 5);
    ShapePtr t = s->transposed();
    FAST_CHECK_EQ(t->type(), ShapeType::Empty);
    FAST_CHECK_EQ(t->rows(), 5);
    FAST_CHECK_EQ(t->cols(), 3);
  }

  SUBCASE("Band")
  {
    ShapePtr s = std::make_unique<BandShape>(3, 5, 2, 1);
    ShapePtr t = s->transposed();
    FAST_CHECK_EQ(t->type(), ShapeType::Band);
    FAST_CHECK_EQ(t->rows(), 5);
    FAST_CHECK_EQ(t->cols(), 3);
    const auto & b = static_cast<const BandShape &>(*t);
    FAST_CHECK_EQ(b.lowerBandwidth(), 1);
    FAST_CHECK_EQ(b.upperBandwidth(), 2);
  }

  SUBCASE("Dense")
  {
    ShapePtr s = std::make_unique<DenseShape>(3, 5);
    ShapePtr t = s->transposed();
    FAST_CHECK_EQ(t->type(), ShapeType::Dense);
    FAST_CHECK_EQ(t->rows(), 5);
    FAST_CHECK_EQ(t->cols(), 3);
  }
}

TEST_CASE("Addition of ShapeType")
{
  ShapePtr e = std::make_unique<EmptyShape>(5, 7);
  ShapePtr b = std::make_unique<BandShape>(5, 7, 2, 1);
  ShapePtr d = std::make_unique<DenseShape>(5, 7);

  auto ee = add(*e, *e);
  FAST_CHECK_EQ(ee->type(), ShapeType::Empty);
  FAST_CHECK_EQ(ee->rows(), 5);
  FAST_CHECK_EQ(ee->cols(), 7);

  auto eb = add(*e, *b);
  FAST_CHECK_EQ(eb->type(), ShapeType::Band);
  FAST_CHECK_EQ(eb->rows(), 5);
  FAST_CHECK_EQ(eb->cols(), 7);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*eb).lowerBandwidth(), 2);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*eb).upperBandwidth(), 1);

  auto ed = add(*e, *d);
  FAST_CHECK_EQ(ed->type(), ShapeType::Dense);
  FAST_CHECK_EQ(ed->rows(), 5);
  FAST_CHECK_EQ(ed->cols(), 7);

  auto be = add(*b, *e);
  FAST_CHECK_EQ(be->type(), ShapeType::Band);
  FAST_CHECK_EQ(be->rows(), 5);
  FAST_CHECK_EQ(be->cols(), 7);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*be).lowerBandwidth(), 2);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*be).upperBandwidth(), 1);

  ShapePtr b2 = std::make_unique<BandShape>(5, 7, 1, 3);
  auto bb = add(*b, *b2);
  FAST_CHECK_EQ(bb->type(), ShapeType::Band);
  FAST_CHECK_EQ(bb->rows(), 5);
  FAST_CHECK_EQ(bb->cols(), 7);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*bb).lowerBandwidth(), 2);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*bb).upperBandwidth(), 3);

  auto bd = add(*b, *d);
  FAST_CHECK_EQ(bd->type(), ShapeType::Dense);
  FAST_CHECK_EQ(bd->rows(), 5);
  FAST_CHECK_EQ(bd->cols(), 7);

  auto de = add(*d, *e);
  FAST_CHECK_EQ(de->type(), ShapeType::Dense);
  FAST_CHECK_EQ(de->rows(), 5);
  FAST_CHECK_EQ(de->cols(), 7);

  auto db = add(*d, *b);
  FAST_CHECK_EQ(db->type(), ShapeType::Dense);
  FAST_CHECK_EQ(db->rows(), 5);
  FAST_CHECK_EQ(db->cols(), 7);

  auto dd = add(*d, *d);
  FAST_CHECK_EQ(dd->type(), ShapeType::Dense);
  FAST_CHECK_EQ(dd->rows(), 5);
  FAST_CHECK_EQ(dd->cols(), 7);
}

TEST_CASE("Multiplication of ShapeType")
{
  ShapePtr e1 = std::make_unique<EmptyShape>(5, 7);
  ShapePtr e2 = std::make_unique<EmptyShape>(7, 5);
  ShapePtr b1 = std::make_unique<BandShape>(5, 7, 2, 1);
  ShapePtr b2 = std::make_unique<BandShape>(7, 5, 0, 1);
  ShapePtr d1 = std::make_unique<DenseShape>(5, 7);
  ShapePtr d2 = std::make_unique<DenseShape>(7, 5);

  auto ee = mult(*e1, *e2);
  FAST_CHECK_EQ(ee->type(), ShapeType::Empty);
  FAST_CHECK_EQ(ee->rows(), 5);
  FAST_CHECK_EQ(ee->cols(), 5);

  auto eb = mult(*e1, *b2);
  FAST_CHECK_EQ(eb->type(), ShapeType::Empty);
  FAST_CHECK_EQ(eb->rows(), 5);
  FAST_CHECK_EQ(eb->cols(), 5);

  auto ed = mult(*e1, *d2);
  FAST_CHECK_EQ(ed->type(), ShapeType::Empty);
  FAST_CHECK_EQ(ed->rows(), 5);
  FAST_CHECK_EQ(ed->cols(), 5);

  auto be = mult(*b1, *e2);
  FAST_CHECK_EQ(be->type(), ShapeType::Empty);
  FAST_CHECK_EQ(be->rows(), 5);
  FAST_CHECK_EQ(be->cols(), 5);

  auto bb = mult(*b1, *b2);
  FAST_CHECK_EQ(bb->type(), ShapeType::Band);
  FAST_CHECK_EQ(bb->rows(), 5);
  FAST_CHECK_EQ(bb->cols(), 5);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*bb).lowerBandwidth(), 2);
  FAST_CHECK_EQ(static_cast<const BandShape &>(*bb).upperBandwidth(), 2);

  auto bd = mult(*b1, *d2);
  FAST_CHECK_EQ(bd->type(), ShapeType::Dense);
  FAST_CHECK_EQ(bd->rows(), 5);
  FAST_CHECK_EQ(bd->cols(), 5);

  auto de = mult(*d1, *e2);
  FAST_CHECK_EQ(de->type(), ShapeType::Empty);
  FAST_CHECK_EQ(de->rows(), 5);
  FAST_CHECK_EQ(de->cols(), 5);

  auto db = mult(*d1, *b2);
  FAST_CHECK_EQ(db->type(), ShapeType::Dense);
  FAST_CHECK_EQ(db->rows(), 5);
  FAST_CHECK_EQ(db->cols(), 5);

  auto dd = mult(*d1, *d2);
  FAST_CHECK_EQ(dd->type(), ShapeType::Dense);
  FAST_CHECK_EQ(dd->rows(), 5);
  FAST_CHECK_EQ(dd->cols(), 5);
}