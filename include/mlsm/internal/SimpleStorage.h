/* Copyright 2021 CNRS-AIST JRL */

#pragma once

#include <mlsm/api.h>
#include <mlsm/defs.h>

#include <Eigen/Core>

#include <type_traits>

namespace mls::internal
{
/** Base class wrapping a Eigen::MatrixXd or a Eigen::Ref that can be const or non-const.*/
class SimpleStorageDense
{
public:
  virtual MatrixConstRef data() const = 0;
  virtual MatrixRef data() = 0;
};

class SimpleStorageDenseConstRef : public SimpleStorageDense
{
public:
  SimpleStorageDenseConstRef(const MatrixConstRef & ref) : ref_(ref) {}

  SimpleStorageDenseConstRef & operator=(const SimpleStorageDenseConstRef &) = delete;
  SimpleStorageDenseConstRef & operator=(SimpleStorageDenseConstRef &&) = delete;

  MatrixConstRef data() const override { return ref_; }
  MatrixRef data() override
  {
    throw std::runtime_error("This data is constant. Cannot return a non-const reference on it.");
  }

private:
  MatrixConstRef ref_;
};

class SimpleStorageDenseNonConstRef : public SimpleStorageDense
{
public:
  SimpleStorageDenseNonConstRef(const MatrixRef & ref) : ref_(ref) {}

  MatrixConstRef data() const override { return ref_; }
  MatrixRef data() override { return ref_; }

private:
  MatrixRef ref_;
};

class SimpleStorageDenseMat : public SimpleStorageDense
{
public:
  SimpleStorageDenseMat(const MatrixConstRef & mat) : mat_(mat), ref_(mat_) {}

  MatrixConstRef data() const override { return ref_; }
  MatrixRef data() override { return ref_; }

private:
  Eigen::MatrixXd mat_;
  MatrixRef ref_;
};

} // namespace mls::internal