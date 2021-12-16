/* Copyright 2021 CNRS-AIST JRL */

#pragma once

#include <Eigen/Core>
#include <Eigen/Jacobi>

#include <memory>

namespace mls
{
using MatrixConstRef = Eigen::Ref<const Eigen::MatrixXd>;
using MatrixRef = Eigen::Ref<Eigen::MatrixXd>;
using VectorConstRef = Eigen::Ref<const Eigen::VectorXd>;
using VectorRef = Eigen::Ref<Eigen::VectorXd>;
inline const Eigen::MatrixXd EmptyMatrix = Eigen::MatrixXd(0, 0);
inline const Eigen::VectorXd EmptyVector = Eigen::VectorXd(0);

using Givens = Eigen::JacobiRotation<double>;

} // namespace mls
