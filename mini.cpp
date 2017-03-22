#include <iostream>
#include <cassert>
#include <immintrin.h>

static const int W = 8; // avx2 width

// generic opencl style vectors
template <int N, typename T = float> using vec = T __attribute__((ext_vector_type(N)));

template <typename OS, int N, typename T> OS & operator<<(OS & os, vec<N,T> v)  {  
  os << '{';
  for (int i =0;i<N;++i) { 
    os << ' ' << v[i];
    if (i != N - 1) os << ',';
  }
  return os << '}';
}  

typedef vec<W,int> mask_type;

static thread_local mask_type mask = -1; // execution mask, defaults to all on

// raii machinery that will grab the current execution mask and restore on exit
struct scope {
  mask_type old;
  scope() noexcept : old(mask) {}
  ~scope() noexcept { mask = old; }
};

static inline int mask_bits() noexcept { 
#if (W == 8) && __AVX2__
  return _mm256_movemask_ps(_mm256_castsi256_ps(mask));
#else
  int result = 0;
  for (int i = 0, b = 1;i < W; ++i, b <<= 1)
    if (mask[i]) result |= b;
  return result;
#endif
}

static inline int bsf(int v) { int r = 0; asm ("bsf %1,%0" : "=r"(r) : "r"(v)); return r; }
static inline int btc(int v, int i) { int r = 0; asm ("btc %1,%0" : "=r"(r) : "r"(i), "0"(v) : "flags" ); return r; }

template <typename F> void foreach_active(F f) {
  scope block;
  int m = mask_bits();
  while (m) {
    int i = bsf(m);
    m = btc(m,i);
    mask_type new_mask(0);
    new_mask[i] = -1;
    mask = new_mask;
    f(i);
  }
}

// iterate over the active bits in the mask, but don't fiddle with our mask
template <typename F> void each(F f) {
  int m = mask_bits();
  while (m) {
    int i = bsf(m);
    m = btc(m,i);
    f(i);
  }
}


// casting vectors
template <typename T, typename S> static inline T vector_cast(S src) noexcept {
  return __builtin_convertvector(src,T);
}

// slow path to be safe
template <typename T>
static inline void blend(vec<8,T> & o, vec<8,T> n, mask_type m) { 
  each([&](int i) { o[i] = m[i] ? n[i] : o[i]; });
}

#ifdef __SSE4_1__
//static inline void blend(vec<8,int16_t> & o, vec<8,int16_t> n, mask_type m) { 
//  o = _mm_blend_epi16(o,n,mask_bits()); // needs a constant mask
//}
#endif

#if (W == 8) && __AVX2__
static inline void blend(vec<8,int> & o, vec<8,int> n, mask_type m) { 
  o = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(o), _mm256_castsi256_ps(n), _mm256_castsi256_ps(m)));
}

static inline void blend(vec<8,float> & o, vec<8,float> n, mask_type m) { 
  o = _mm256_blendv_ps(o,n,_mm256_castsi256_ps(m));
}

static inline void blend(vec<8,double> & o, vec<8,double> n, mask_type m) { 
  union { 
    vec<8,double> v8;
    vec<4,double> v4[2];
  } ou, nu;
  ou.v8 = o;
  nu.v8 = n;
  ou.v4[0] = _mm256_blendv_ps(ou.v4[0], nu.v4[0], _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(m,0))));
  ou.v4[1] = _mm256_blendv_ps(ou.v4[1], nu.v4[1], _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(m,1))));
  o = ou.v8;
}
#endif

template <typename T, int N> static inline void blend(T o[N], const T n[N], mask_type m) { 
  #pragma clang loop unroll(full)
  for (int i=0;i<N;++i)
    blend(o[i],n[i],m);
}

template <typename T> struct rep { 
  typedef T type; 
  static constexpr type embed(T t) { return t; }
  static constexpr T project(type t) { return t; }
};

template <> struct rep<bool> {
  typedef int type;
  static constexpr int embed(bool t) { return t ? ~0 : 0; }
  static constexpr bool project(int t) { return t; }
};

template <typename T> struct varying;

template <typename T> struct lane_ref {
  varying<T> & t;
  int i;
  lane_ref & operator = (T rhs) noexcept;
  operator T () const noexcept;
};

template <typename T> struct varying<T&>;

static const struct ProgramIndexTag { } programIndexTag {};

template <typename T> struct varying {
  typedef typename rep<T>::type R;
  vec<W,R> value;
  varying() {}
  varying(T a) { value = rep<T>::embed(a); }
  varying(ProgramIndexTag) { for (int i=0;i<W;++i) value[i] = i; }
  varying(vec<W,R> rhs) : value(rhs) {}
  varying(const varying & rhs) noexcept : value(rhs.value) {} // copy
  template <typename S> varying(varying<S> & rhs) noexcept : value(vector_cast<vec<W,R>>(rhs.value)) {} // cast

  // triggers https://bugs.llvm.org//show_bug.cgi?id=15781
  // varying & operator =(varying & rhs) noexcept { value = mask ? value : rhs.value; return *this;} 

  varying & operator =(varying & rhs) noexcept { value = blend(value, rhs.value, mask); return *this; };
  varying & operator +=(varying & rhs) noexcept { (*this) = (*this) + rhs; return *this; };
  varying & operator *=(varying & rhs) noexcept { (*this) = (*this) * rhs; return *this; };
  varying & operator /=(varying & rhs) noexcept { (*this) = (*this) / rhs; return *this; };
  varying & operator &=(varying & rhs) noexcept { (*this) = (*this) & rhs; return *this; };
  varying & operator |=(varying & rhs) noexcept { (*this) = (*this) | rhs; return *this; };

  T lane(int i) const noexcept { return value[i]; }
  lane_ref<T> lane(int i) noexcept { return lane_ref<T> { *this, i }; }

  template <typename S> auto operator [](S s) -> decltype(index(*this,s)) { return index(*this,s); }

  // TODO: now we need a reference type
  template <typename S> typename std::enable_if<std::is_same<T,int>::value,varying<S>>::type operator[](const S * array) {
    varying<S> result;
    each([&](int i) { result.lane(i) = array[lane(i)]; });
    return result;
  }
};

// support cout
template <typename OS, typename T> OS & operator<<(OS & os, varying<T> v)  {  
  os << "varying(";
  for (int i =0;i<W;++i) { 
    os << ' ';
    if (mask[i]) os << v.lane(i);
    else os << '_';
    if (i != W - 1) os << ',';
  }
  os << ')';
  return os;  
}  

template <typename T> lane_ref<T> & lane_ref<T>::operator = (T rhs) noexcept {
  t.value[i] = rhs;
  return *this;
}

template <typename T> lane_ref<T>::operator T () const noexcept {
  return t.value[i];
}

template <typename S, typename T> varying<decltype(S() & T())> operator & (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() & T())>(l.value & r.value); }
template <typename S, typename T> varying<decltype(S() | T())> operator | (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() | T())>(l.value | r.value); }
template <typename S, typename T> varying<decltype(S() + T())> operator + (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() + T())>(l.value + r.value); }
template <typename S, typename T> varying<decltype(S() - T())> operator - (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() - T())>(l.value - r.value); }
template <typename S, typename T> varying<decltype(S() * T())> operator * (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() * T())>(l.value * r.value); }
template <typename S, typename T> varying<decltype(S() / T())> operator / (varying<S> l, varying<T> r) noexcept { return varying<decltype(S() / T())>(l.value / r.value); }
template <typename T> varying<T> operator ~ (varying<T> l) noexcept { return varying<T>(~l.value); }
template <typename T> varying<T> operator ! (varying<T> l) noexcept { return varying<T>(!l.value); }

template <typename S, typename T> varying<decltype(S() & T())> operator & (varying<S> l, T r) noexcept { return varying<decltype(S() & T())>(l.value & r); }
template <typename S, typename T> varying<decltype(S() | T())> operator | (varying<S> l, T r) noexcept { return varying<decltype(S() | T())>(l.value | r); }
template <typename S, typename T> varying<decltype(S() + T())> operator + (varying<S> l, T r) noexcept { return varying<decltype(S() + T())>(l.value + r); }
template <typename S, typename T> varying<decltype(S() - T())> operator - (varying<S> l, T r) noexcept { return varying<decltype(S() - T())>(l.value - r); }
template <typename S, typename T> varying<decltype(S() * T())> operator * (varying<S> l, T r) noexcept { return varying<decltype(S() * T())>(l.value * r); }
template <typename S, typename T> varying<decltype(S() / T())> operator / (varying<S> l, T r) noexcept { return varying<decltype(S() / T())>(l.value / r); }

template <typename S, typename T> varying<decltype(S() & T())> operator & (S l, varying<T> r) noexcept { return varying<decltype(S() & T())>(l & r.value); }
template <typename S, typename T> varying<decltype(S() | T())> operator | (S l, varying<T> r) noexcept { return varying<decltype(S() | T())>(l | r.value); }
template <typename S, typename T> varying<decltype(S() + T())> operator + (S l, varying<T> r) noexcept { return varying<decltype(S() + T())>(l + r.value); }
template <typename S, typename T> varying<decltype(S() - T())> operator - (S l, varying<T> r) noexcept { return varying<decltype(S() - T())>(l - r.value); }
template <typename S, typename T> varying<decltype(S() * T())> operator * (S l, varying<T> r) noexcept { return varying<decltype(S() * T())>(l * r.value); }
template <typename S, typename T> varying<decltype(S() / T())> operator / (S l, varying<T> r) noexcept { return varying<decltype(S() / T())>(l / r.value); }

template <typename F, typename T> void if_(varying<bool> v, T t, F f) {
  scope block;
  mask &= v.value;
  if (mask_bits()) t();
  mask = ~mask & block.old;
  if (mask_bits()) f();
}

static const varying<int> programIndex(programIndexTag); // 0,1,2,3,4,5,6,7);

template <typename F> void foreach(int i, int j, F f) {
  // this will reset the mask
  scope block;
  for (; i < j-W;i += W) { 
    mask = -1;
    f(programIndex + i); // todo: create linear<int> so we can do coherent loads
  }
  varying<int> vi(programIndex + i);
#if (W == 8) && __AVX2__
  mask = _mm256_cmpgt_epi32(vec<W,int>(j), vi.value);
#else
  mask_type m = 0;
  #pragma clang loop unroll(full)
  for (int k = 0, mk = j - i;k<mk;++k) m[k] = ~0;
  mask = m;
#endif
  f(vi);
  // mask off the remainder 
}

// aos so we can use the mask
template <typename T, int N> struct varying<vec<N,T>> {
  varying<T> value[N];

  varying & operator =(varying & rhs) noexcept { value = rhs.value; return *this; };
  varying & operator +=(varying & rhs) noexcept { (*this) = (*this) + rhs; return *this; };
  varying & operator *=(varying & rhs) noexcept { (*this) = (*this) * rhs; return *this; };
  varying & operator /=(varying & rhs) noexcept { (*this) = (*this) / rhs; return *this; };
  varying & operator &=(varying & rhs) noexcept { (*this) = (*this) & rhs; return *this; };
  varying & operator |=(varying & rhs) noexcept { (*this) = (*this) | rhs; return *this; };

  vec<N,T> lane(int j) const noexcept { 
    vec<N,T> result;
    for (int i=0;i<W;++i) {
      result[i] = value[i].lane(j); // cut against the grain
    }
    return result;
  }
  // we need a proxy object for item(j)  as a reference
};

int main (int argc, char ** argv) {
  vec<4,int> i { 1,2,3,4};
  i.wx += 1;
  auto d = vector_cast<vec<4,double>>(i);
  varying<bool> b;
  varying<vec<4>> v;
  v.value[0].value.w += 1;
  int foo [9] = { 1,0,1,2,1,2,3,4,5 };
  foreach(0,9, [&](varying<int> k) { 
    std::cout << k << " " << k[foo] << "\n";
  });
}
