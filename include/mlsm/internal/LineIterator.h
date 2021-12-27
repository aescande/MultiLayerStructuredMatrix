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

/** An iterator-like structure to serve as an underlying implementation of a forward iterator over
 * a "line" of a matrix (i.e a row or a column).
 *
 * \tparam T the type of the dereferenced iterator.
 */
template<typename T>
class LineIteratorBase
{
public:
  /** \param l index of the line on which to iterate.*/
  LineIteratorBase(int l) : l_(l) {}
  /** Value of the element the iterator currently points to.*/
  virtual T val() const { return LineIteratorTraits<T>::End; }
  /** Moving the iterator to the next element.*/
  virtual void next(){};
  /** Return the index of the line that is being iterated on.*/
  int line() const { return l_; }

protected:
  int l_; // The index of the line in the matrix
};

/** Specific implementation of a LineIteratorBase for when (non-zero) elements in a line are
 * contiguous.
 * 
 * \tparam T the type of the dereferenced iterator.
 */
template<typename T>
class ContinuousLineIterator : public LineIteratorBase<T>
{
public:
  /** \param l index of the line on which to iterate.
   *  \param start initial value
   *  \param end value after the last valid element
   */
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
  T val_; // current value 
  T end_; // end value (first value after the last valid element) 
};

/** A class to represent a row or a colum of a matrix-like class.
 * It provides an iterator on its non-zero elements that is based on an underlying
 * LineIteratorBase, as well as \c begin and \c end() methods that makes it compatible with
 * numerous iterator-based idioms of C++.
 * 
 * \tparam T The type of the dereferenced iterator
 * \tparam Provider The class that will provide the underlying row and column iterators, through
 * \c rowIterator and \c colIterator methods.
 * \tparam Row \c true for a row \c false for a column.
 */
template<typename T, class Provider, bool Row>
class Line
{
public:
  using UnderlyingIt = LineIteratorBase<T>;
  using UnderlyingItPtr = std::unique_ptr<UnderlyingIt>;
  /** A forward iterator wrapping a LineIteratorBase, with basic operator defines
   *
   * \internal We have this wrapping to allow for different implementation of the iterator, made by
   * deriving LineIteratorBase, but keep an iterator interface on top of the virtual calls.
   */
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

  /** Return the number of non-zero on the line.*/
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