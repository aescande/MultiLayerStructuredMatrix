/** Copyright 2021 CNRS-AIST JRL*/

#pragma once

#include <memory>

namespace mls::internal
{
/** To be specialized for each type a LineIterator can point to
 * need to define a static constexpr T End, that will be used as value to
 * signify the end iterator
 */
template<typename T>
struct LineIteratorTraits
{
  static constexpr T End = {};
};

template<typename T>
class LineIteratorBase
{
public:
  LineIteratorBase(int l) : l_(l) {}
  virtual T val() const { return LineIteratorTraits<T>::End; }
  virtual void next(){};
  int line() const { return l_; }

protected:
  int l_;
};

template<typename T>
class ContinuousLineIterator : public LineIteratorBase<T>
{
public:
  ContinuousLineIterator(int l, const T & start, const T & end) : LineIteratorBase(l), val_(start), end_(end) {}
  T val() const override { return val_; }
  void next() override
  {
    if(val_ != LineIteratorTraits<T>::End)
    {
      ++val_;
      if(val_ >= end_)
        val_ = LineIteratorTraits<T>::End;
    }
  }

private:
  T val_;
  T end_;
};

template<typename T, class Provider, bool Row>
class Line
{
public:
  using UnderlyingIt = LineIteratorBase<T>;
  using UnderlyingItPtr = std::unique_ptr<UnderlyingIt>;
  class Iterator
  {
  public:
    Iterator & operator++()
    {
      it_->next();
      return *this;
    }
    T operator*() const { return it_->val(); }

    friend bool operator==(const Iterator & a, const Iterator & b)
    {
      return a.it_->line() == b.it_->line() && *a == *b;
    };
    friend bool operator!=(const Iterator & a, const Iterator & b) { return !(a == b); };

  private:
    Iterator(int l) : it_(new UnderlyingIt(l)) {}
    Iterator(const Provider & p, int l)
    {
      if constexpr(Row)
        it_ = p.rowIterator(l);
      else
        it_ = p.colIterator(l);
    }
    UnderlyingItPtr it_;
    friend class Line<T, Provider, true>;
    friend class Line<T, Provider, false>;
  };

  Iterator begin() const { return {p_, l_}; }
  Iterator end() const { return {l_}; }
  int size() const
  {
    if constexpr(Row)
      return p_.rowNNZ(l_);
    else
      return p_.colNNZ(l_);
  }

private:
  Line(const Provider & p, int l) : l_(l), p_(p) {}
  int l_;
  const Provider & p_;
  friend Provider;
};

}