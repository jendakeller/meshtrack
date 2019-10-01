#ifndef SPARSE_VECTOR_H_
#define SPARSE_VECTOR_H_

#include <vector>
#include <utility>

#include <base/stack_container.h>

//#define STACK_VECTOR_SIZE 8
#define STACK_VECTOR_SIZE 61

template<typename T>
class SparseVector
{
public:
  SparseVector();
  explicit SparseVector(int pos); // constructs vector of zeros with 1 on given position
  SparseVector(const SparseVector<T>& a);
  ~SparseVector();

  SparseVector& operator=(const SparseVector<T>& a);
  int size() const;

  int n;
  StackVector<int,STACK_VECTOR_SIZE> ind;
  StackVector<T,STACK_VECTOR_SIZE>   val;
};

template<typename T> SparseVector<T> operator-(const SparseVector<T>& x);
template<typename T> SparseVector<T> operator+(const SparseVector<T>& x, const SparseVector<T>& y);
template<typename T> SparseVector<T> operator-(const SparseVector<T>& x, const SparseVector<T>& y);

template<typename T> SparseVector<T> operator*(const T x,const SparseVector<T>& y);
template<typename T> SparseVector<T> operator*(const SparseVector<T>& x,const T y);
template<typename T> SparseVector<T> operator/(const SparseVector<T>& x,const T y);

template<typename T>
SparseVector<T>::SparseVector() : n(0) {}

template<typename T>
SparseVector<T>::SparseVector(int pos)
{
  n = 1;

  ind = StackVector<int,STACK_VECTOR_SIZE>(); ind->push_back(pos);
  val = StackVector<T,STACK_VECTOR_SIZE>(); val->push_back(T(1));
}

template<typename T>
SparseVector<T>::SparseVector(const SparseVector<T>& u)
{
  n = u.n;

  ind = u.ind;
  val = u.val;
}

template<typename T>
SparseVector<T>& SparseVector<T>::operator=(const SparseVector<T>& u)
{
  if (this!=&u)
  {
    n = u.n;
    ind = u.ind;
    val = u.val;
  }

  return *this;
}

template<typename T>
SparseVector<T>::~SparseVector()
{
  // if (n>61)
  // printf("%d\n", n);
  n = 0;
}

template<typename T>
int SparseVector<T>::size() const
{
  return n;
}

template<typename T>
SparseVector<T> operator-(const SparseVector<T>& x)
{
  SparseVector<T> z(x);
  for(int i=0;i<z.size();i++) { z.val[i] = -z.val[i]; }
  return z;
}

template<typename T>
SparseVector<T> operator+(const SparseVector<T>& x,const SparseVector<T>& y)
{
  SparseVector<T> z;

  {
    if      (y.size()==0) { return x; }
    else if (x.size()==0) { return y; }

    int xp = 0;
    int yp = 0;
    z.n = 0;

    while(xp < x.n || yp < y.n)
    {
      if(xp >= x.n)
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(y.val[yp++]);
      }
      else if(yp >= y.n)
      {
        z.ind->push_back(x.ind[xp]);
        z.val->push_back(x.val[xp++]);
      }
      else if(y.ind[yp] < x.ind[xp])
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(y.val[yp++]);
      }
      else if(x.ind[xp] < y.ind[yp])
      {
        z.ind->push_back(x.ind[xp]);
        z.val->push_back(x.val[xp++]);
      }
      else
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(x.val[xp++] + y.val[yp++]);
      }

      z.n++;
    }
  }

  return z;
}

template<typename T>
SparseVector<T> operator-(const SparseVector<T>& x,const SparseVector<T>& y)
{
  SparseVector<T> z;

  {
    if      (y.size()==0) { return  x; }
    else if (x.size()==0) { return -y; }

    int xp = 0;
    int yp = 0;
    z.n = 0;

    while(xp < x.n || yp < y.n)
    {
      if(xp >= x.n)
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(-y.val[yp++]);
      }
      else if(yp >= y.n)
      {
        z.ind->push_back(x.ind[xp]);
        z.val->push_back(x.val[xp++]);
      }
      else if(y.ind[yp] < x.ind[xp])
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(-y.val[yp++]);
      }
      else if(x.ind[xp] < y.ind[yp])
      {
        z.ind->push_back(x.ind[xp]);
        z.val->push_back(x.val[xp++]);
      }
      else
      {
        z.ind->push_back(y.ind[yp]);
        z.val->push_back(x.val[xp++] - y.val[yp++]);
      }

      z.n++;
    }
  }

  return z;
}

template<typename T>
SparseVector<T> operator*(const T x,const SparseVector<T>& y)
{
  SparseVector<T> z(y);
  for(int i=0;i<z.size();i++) { z.val[i] *= x; }
  return z;
}

template<typename T>
SparseVector<T> operator*(const SparseVector<T>& x,const T y)
{
  SparseVector<T> z(x);
  for(int i=0;i<z.size();i++) { z.val[i] *= y; }
  return z;
}

template<typename T>
SparseVector<T> operator/(const SparseVector<T>& x,const T y)
{
  SparseVector<T> z(x);
  for(int i=0;i<z.size();i++) { z.val[i] /= y; }
  return z;
}

#endif
