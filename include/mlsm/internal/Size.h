/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <assert.h>
#include <numeric>

#define MLSM_BANDSIZE_OPERATION_DOES_NOT_OVERFLOW(op, a, b, lim) (std::abs(double(a) op double(b)) < lim)

namespace mls::internal
{
/** Represent a size for band matrices, including an infinite size.
 *
 *  \internal This wraps an int with a representation for +/-infinity, paying attention to overflow
 * and underflow.
 */
class Size
{
public:
  static_assert(std::numeric_limits<int>::max() + std::numeric_limits<int>::min() == -1);
  static constexpr int inf = std::numeric_limits<int>::max();

  Size(int size) : size_(size) {}
  explicit operator int() const
  {
    assert(isfinite());
    return size_;
  }

  Size operator-() const
  {
    if(isnan()) return nan;
    return -size_;
  }

  Size & operator+=(const Size & other)
  {
    if(isnan() || other.isnan())
      size_ = nan;
    else
    {
      if(size_ == -inf)
      {
        if(other.size_ == inf)
          size_ = nan;
        // otherwise, we keep -inf
      }
      else if(size_ == inf)
      {
        if(other.size_ == -inf)
          size_ = nan;
        // otherwise, we keep inf
      }
      else
      {
        if(other.size_ == -inf)
          size_ = -inf;
        else if(other.size_ == inf)
          size_ = inf;
        else
        {
          assert(MLSM_BANDSIZE_OPERATION_DOES_NOT_OVERFLOW(+, size_, other.size_, inf));
          size_ += other.size_;
        }
      }
    }

    return *this;
  }

  Size & operator-=(const Size & other)
  {
    return operator+=(-other);
  }

  bool isfinite() const
  {
    return -inf < size_ && size_ < inf;
  }

  bool isinf() const
  {
    return !isfinite();
  }

  bool isnan() const
  {
    return size_ == nan;
  }

private:
  static constexpr int nan = std::numeric_limits<int>::min();
  int size_;

  friend bool operator==(const Size & lhs, const Size & rhs);
  friend bool operator<(const Size & lhs, const Size & rhs);
};

inline Size operator+(Size lhs, const Size & rhs)
{
  // lhs is taken by copy so as to create a new instance
  lhs += rhs;
  return lhs;
}

inline Size operator-(Size lhs, const Size & rhs)
{
  // lhs is taken by copy so as to create a new instance
  lhs -= rhs;
  return lhs;
}

inline bool operator==(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return lhs.size_ == rhs.size_;
}
inline bool operator!=(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return !operator==(lhs, rhs);
}
inline bool operator<(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return lhs.size_ < rhs.size_;
}
inline bool operator>(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return operator<(rhs, lhs);
}
inline bool operator<=(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return !operator>(lhs, rhs);
}
inline bool operator>=(const Size & lhs, const Size & rhs)
{
  if(lhs.isnan() || rhs.isnan()) return false;
  return !operator<(lhs, rhs);
}

} // namespace mls::internal