#pragma once

#include <array>
#include <immintrin.h>
#include <utility>
#include "cpu.h"
#include "avx2_math.h"

// force the avx2 feature set on, so we can compile these combinators.
// this allows the user to leave the avx2 feature disabled, so their
// code won't get contaminated with avx2 stuf and yet for us to still
// automatically select this ISA.
#define SPMD_AVX2 __attribute__((target("avx2")))

namespace spmd {
  // spmd on simd computation kernel using avx2
  namespace avx2 {
    static const bool allow_slow_path = true;
    static const int items = 8;

    // forward template declaration
    template <typename T> struct varying;

    // execution mask
    struct mask {
      __m256i value;

      // default to initialized.
      SPMD_AVX2 mask() noexcept : value(_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())) {}
      SPMD_AVX2 mask(bool b) noexcept : value(_mm256_set1_epi32(b ? ~0 : 0)) {}
      SPMD_AVX2 mask(__m256i value) noexcept : value(value) {}
      SPMD_AVX2 mask(const mask & m) noexcept : value(m.value) {}
      // mask(mask && m) noexcept : value(std::forward(m.value)) {} // move constructor

      static SPMD_AVX2 mask on() noexcept { return mask(); }
      static SPMD_AVX2 mask off() noexcept { return _mm256_setzero_si256(); }

      // no threads in the "warp" are active
      SPMD_AVX2 bool any() const noexcept {
        return _mm256_movemask_ps(_mm256_castsi256_ps(value)) != 0;
      }

      // all 8 threads in the "warp" are active, fast path
      SPMD_AVX2 bool all() const noexcept {
        return _mm256_movemask_ps(_mm256_castsi256_ps(value)) == 0xff;
      }

      SPMD_AVX2 mask operator & (const mask & that) const noexcept {
        return _mm256_and_si256(value, that.value);
      }

      SPMD_AVX2 mask operator | (const mask & that) const noexcept {
        return _mm256_or_si256(value, that.value);
      }

      SPMD_AVX2 mask operator ^ (const mask & that) const noexcept {
        return _mm256_xor_si256(value, that.value);
      }

      SPMD_AVX2 mask & operator &= (const mask & rhs) noexcept {
        value = _mm256_and_si256(value, rhs.value);
        return *this;
      }

      SPMD_AVX2 mask & operator |= (const mask & rhs) noexcept {
        value = _mm256_or_si256(value, rhs.value);
        return *this;
      }

      SPMD_AVX2 mask & operator ^= (const mask & rhs) noexcept {
        value = _mm256_or_si256(value, rhs.value);
        return *this;
      }

      mask operator ~ () noexcept {
        return (*this) ^ on();
      }

      template <typename F> SPMD_AVX2 void each(F f) {
        for (int m = _mm256_movemask_ps(_mm256_castsi256_ps(value));m;) {
          auto i = __builtin_ffs(m);
          f(i);
          m &= ~ (1 << i);
        }
      }

      // active worker count
      SPMD_AVX2 int active() const {
        return __builtin_popcount(_mm256_movemask_ps(_mm256_castsi256_ps(value)));
      }
    };

    extern thread_local mask execution_mask; // shared execution mask

    // primary template
    template <typename T>
    struct varying {
      static_assert(allow_slow_path, "slow path disabled");

      std::array<T,items> value;

      varying() {}
      varying(const T & x) {
        execution_mask.each([&](int i) { *(value[i]) = x; });
      }
      explicit varying(std::array<T,items> value) noexcept : value(value) {}
      varying(const varying & that) : value(that.value) {}
      varying masked() const { // replace w/ default where mask is disabled
        (~execution_mask).each([&](int i) { value[i] = T(); });
      }
      T & item(int i) noexcept { return value[i]; }
      const T & item(int i) const noexcept { return value[i]; }

      template <typename F> static varying make(F f) {
        varying result;
        execution_mask.each([&](int i) { result.value[i] = f(i); });
        return result;
      }

      varying operator + (const varying & rhs) const {
        return make([&](int i) { return item(i) + rhs.item(i); });
      }
      varying operator - (const varying & rhs) const {
        return make([&](int i) { return item(i) - rhs.item(i); });
      }
      varying operator * (const varying & rhs) const {
        return make([&](int i) { return item(i) * rhs.item(i); });
      }
      varying operator / (const varying & rhs) const {
        return make([&](int i) { return item(i) / rhs.item(i); });
      }
      varying operator % (const varying & rhs) const {
        return make([&](int i) { return item(i) % rhs.item(i); });
      }
      varying operator & (const varying & rhs) const {
        return make([&](int i) { return item(i) & rhs.item(i); });
      }
      varying operator | (const varying & rhs) const {
        return make([&](int i) { return item(i) | rhs.item(i); });
      }
      varying operator ^ (const varying & rhs) const {
        return make([&](int i) { return item(i) ^ rhs.item(i); });
      }
      varying operator ~ () const {
        return make([&](int i) { return ~(item(i)); });
      }
      varying operator ! () const {
        return make([&](int i) { return !(item(i)); });
      }
      varying & operator = (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] = rhs.value[i]; });
        return *this;
      }
      varying & operator += (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] += rhs.value[i]; });
        return *this;
      }
      varying & operator -= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] -= rhs.value[i]; });
        return *this;
      }
      varying & operator *= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] *= rhs.value[i]; });
        return *this;
      }
      varying & operator /= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] /= rhs.value[i]; });
        return *this;
      }
      varying & operator %= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] %= rhs.value[i]; });
        return *this;
      }
      varying & operator |= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] |= rhs.value[i]; });
        return *this;
      }
      varying & operator &= (const varying & rhs) {
        execution_mask.each([&](int i) { value[i] &= rhs.value[i]; });
        return *this;
      }
    };


    // varying<bool>
    template<>
    struct varying<bool> {
      __m256i value;

      SPMD_AVX2 varying() noexcept {}
      SPMD_AVX2 varying(bool b) noexcept : value(_mm256_set1_epi32(b ? ~0 : 0)) {}
      SPMD_AVX2 varying(__m256i value) noexcept : value(value) {}
      SPMD_AVX2 varying(const varying & that) noexcept : value(that.value) {} // copy constructor

      SPMD_AVX2 varying masked() const noexcept {
        return varying {
          _mm256_castps_si256(_mm256_blendv_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(value), _mm256_castsi256_ps(execution_mask.value)))
        };
      }


      SPMD_AVX2 varying operator || (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      SPMD_AVX2 varying operator && (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      SPMD_AVX2 varying operator | (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      SPMD_AVX2 varying operator & (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      SPMD_AVX2 varying operator ! () const noexcept { return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), value); }

      // masked assignment operator
      SPMD_AVX2 varying & operator = (const varying & rhs) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      SPMD_AVX2 varying & operator|=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_or_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      SPMD_AVX2 varying & operator&=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_and_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      // varying<bool> reference to a slot in a simd'd boolean
      struct item_ref {
         varying<bool> & m;
         int i;
         // item_ref() = delete;
         // item_ref(const item_ref &) = delete;
         // item_ref(item_ref &&) = delete;
         item_ref(varying<bool> & m, int i) noexcept : m(m), i(i) {}
         operator bool() const noexcept {
           return reinterpret_cast<int32_t*>(&m.value)[i] != 0;
         }
         item_ref & operator = (bool b) noexcept {
           reinterpret_cast<int32_t*>(&m.value)[i] = b ? ~0 : 0;
           return *this;
         }
         item_ref & operator &= (bool b) noexcept {
           reinterpret_cast<int32_t*>(&m.value)[i] &= b ? ~0 : 0;
           return *this;
         }
         item_ref & operator |= (bool b) noexcept {
           reinterpret_cast<int32_t*>(&m.value)[i] |= b ? ~0 : 0;
           return *this;
         }
         item_ref & operator ^= (bool b) noexcept {
           reinterpret_cast<int32_t*>(&m.value)[i] ^= b ? ~0 : 0;
           return *this;
         }
      };

      struct const_item_ref {
        const varying<bool> & m;
        int i;
        const_item_ref(const varying<bool> & m, int i) noexcept : m(m), i(i) {}
        operator bool() const noexcept {
          return reinterpret_cast<const int32_t*>(&m.value)[i] != 0;
        }
      };

      struct item_ptr {
        varying<bool> * m;
        int i;
        item_ptr() noexcept : m(nullptr), i(0) {}
        item_ptr(const item_ptr & p) noexcept : m(p.m), i(p.i) {}
        item_ptr(item_ptr && p) noexcept : m(std::move(p.m)), i(std::move(p.i)) {}
        explicit item_ptr(varying<bool> * m, int i) noexcept : m(m), i(i) {}
        item_ptr & operator = (const item_ptr & rhs) noexcept {
          m = rhs.m;
          i = rhs.i;
          return *this;
        }
        bool operator == (const item_ptr & rhs) const noexcept {
          return (m == rhs.m) && (i == rhs.i);
        }
        bool operator != (const item_ptr & rhs) const noexcept {
          return (m != rhs.m) || (i == rhs.i);
        }
        bool operator <= (const item_ptr & rhs) const noexcept {
          return (m < rhs.m) || ((m == rhs.m) && (i <= rhs.i));
        }
        bool operator >= (const item_ptr & rhs) const noexcept {
          return (m >= rhs.m) || ((m == rhs.m) && (i >= rhs.i));
        };
        bool operator < (const item_ptr & rhs) const noexcept {
          return (m < rhs.m) || ((m == rhs.m) && (i < rhs.i));
        }
        bool operator > (const item_ptr & rhs) const noexcept {
          return (m > rhs.m) || ((m == rhs.m) && (i > rhs.i));
        }
      };

      struct const_item_ptr {
        const varying<bool> * m;
        int i;
        const_item_ptr() noexcept : m(nullptr), i(0) {}
        const_item_ptr(const const_item_ptr & p) noexcept : m(p.m), i(p.i) {}
        const_item_ptr(const_item_ptr && p) noexcept : m(std::move(p.m)), i(std::move(p.i)) {}
        explicit const_item_ptr(varying<bool> * m, int i) noexcept : m(m), i(i) {}
        const_item_ptr & operator = (const const_item_ptr & rhs) noexcept {
          m = rhs.m;
          i = rhs.i;
          return *this;
        }
        bool operator == (const const_item_ptr & rhs) const noexcept {
          return (m == rhs.m) && (i == rhs.i);
        }
        bool operator != (const const_item_ptr & rhs) const noexcept {
          return (m != rhs.m) || (i == rhs.i);
        }
        bool operator <= (const const_item_ptr & rhs) const noexcept {
          return (m < rhs.m) || ((m == rhs.m) && (i <= rhs.i));
        }
        bool operator >= (const const_item_ptr & rhs) const noexcept {
          return (m >= rhs.m) || ((m == rhs.m) && (i >= rhs.i));
        };
        bool operator < (const const_item_ptr & rhs) const noexcept {
          return (m < rhs.m) || ((m == rhs.m) && (i < rhs.i));
        }
        bool operator > (const const_item_ptr & rhs) const noexcept {
          return (m > rhs.m) || ((m == rhs.m) && (i > rhs.i));
        }
      };

      item_ref item(int i) { return item_ref(*this,i); }
      const_item_ref item(int i) const { return const_item_ref(*this,i); }

      template <typename F> static varying make(F f) {
        varying result;
        execution_mask.each([&](int i) { result.item(i) = f(i); });
        return result;
      }
    }; // varying<bool>

    static inline const varying<bool>::item_ptr operator & (const varying<bool>::item_ref & ref) {
      return varying<bool>::item_ptr(&ref.m,ref.i);
    }

    static inline varying<bool>::item_ptr operator & (varying<bool>::item_ref & ref) {
      return varying<bool>::item_ptr(&ref.m,ref.i);
    }

    static inline const varying<bool>::item_ref operator * (const varying<bool>::item_ptr & ptr) {
      return varying<bool>::item_ref(*ptr.m,ptr.i);
    }

    static inline varying<bool>::item_ref operator * (varying<bool>::item_ptr & ptr) {
      return varying<bool>::item_ref(*ptr.m,ptr.i);
    }

    // default, horrible implementations that work for everything
    template <typename T> static inline varying<bool> operator == (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] == rhs.value[i]; });
    }
    template <typename T> static inline varying<bool> operator != (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] != rhs.value[i]; });
    }
    template <typename T> static inline varying<bool> operator <= (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] <= rhs.value[i]; });
    }
    template <typename T> static inline varying<bool> operator >= (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] >= rhs.value[i]; });
    }
    template <typename T> static inline varying<bool> operator < (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] < rhs.value[i]; });
    }
    template <typename T> static inline varying<bool> operator > (const varying<T> & lhs, const varying<T> & rhs) {
      return varying<bool>::make([&](int i) { return lhs.value[i] > rhs.value[i]; });
    }

    // overload for varying<bool> comparisons
    static SPMD_AVX2 inline varying<bool> operator != (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return _mm256_xor_si256(lhs.value,rhs.value); }
    static SPMD_AVX2 inline varying<bool> operator == (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return !(lhs != rhs); }
    static SPMD_AVX2 inline varying<bool> operator < (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return _mm256_andnot_si256(lhs.value,rhs.value); }
    static SPMD_AVX2 inline varying<bool> operator > (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return _mm256_andnot_si256(rhs.value,lhs.value); }
    static SPMD_AVX2 inline varying<bool> operator <= (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return (!lhs) || rhs; }
    static SPMD_AVX2 inline varying<bool> operator >= (const varying<bool> & lhs, const varying<bool> & rhs) noexcept { return lhs || (!rhs); }

    SPMD_AVX2  static inline mask & operator &= (mask & lhs, const varying<bool> & rhs) noexcept {
      lhs.value = _mm256_and_si256(lhs.value, rhs.value);
      return lhs;
    }

    template <typename T> struct varying<T*> {
#ifdef SPMD_64
      __m256i value[2];
      explicit SPMD_AVX2 varying(const __m256i value[2]) noexcept : value { value[0],value[1] } {}
      SPMD_AVX2 varying & operator =(const varying & that) noexcept {
        value[0] = _mm256_castps_si256(
          _mm256_blendv_ps(
            _mm256_castsi256_ps(value[0]),
            _mm256_castsi256_ps(that.value[0]),
            _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(execution_mask.value,0)))
          )
        );
        value[1] = _mm256_castps_si256(
          _mm256_blendv_ps(
            _mm256_castsi256_ps(value[1]),
            _mm256_castsi256_ps(that.value[1]),
            _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(execution_mask.value,1)))
          )
        );
        return *this;
      }
#else
      __m256i value;
      explicit SPMD_AVX2 varying(const __m256i value) noexcept : value(value) {}
      SPMD_AVX2 varying & operator =(const varying<T*> & that) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
#endif
      SPMD_AVX2 T * & item(int i) { return reinterpret_cast<T**>(value)[i]; }
      SPMD_AVX2 const T * & item(int i) const { return reinterpret_cast<const T**>(value)[i]; }
    };

    template <typename T> struct varying<T&> {
#ifdef SPMD_64
      __m256i value[2];
      explicit SPMD_AVX2 varying(const __m256i value[2]) noexcept : value { value[0], value[1] } {}
#else
      __m256i value;
      explicit SPMD_AVX2 varying(const __m256i value) noexcept : value(value) {}
#endif
      varying & operator =(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) = rhs.item(i); });
        return *this;
      }
      varying & operator +=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) += rhs.item(i); });
        return *this;
      }
      varying & operator -=( const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) -= rhs.item(i); });
        return *this;
      }
      varying & operator *=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) *= rhs.item(i); });
        return *this;
      }
      varying & operator /=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) /= rhs.item(i); });
        return *this;
      }
      varying & operator &=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) &= rhs.item(i); });
        return *this;
      }
      varying & operator |=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { item(i) |= rhs.item(i); });
        return *this;
      }
      T & item(int i) noexcept { return *(reinterpret_cast<T**>(value)[i]); }
      const T & item(int i) const noexcept { return *(reinterpret_cast<T**>(value)[i]); }
      operator varying<T>() const {
        varying<T> result;
        execution_mask.each([&](int i) { result.item(i) = item(i); });
        return result;
      }
    };

    template <typename T>
    varying<T*> operator & (const varying<T&> & r) noexcept {
      return varying<T*>(r.value);
    }

    template <typename T>
    varying<T&> operator * (const varying<T*> & p) noexcept {
      return varying<T&>(p.value);
    }

    template <> struct varying<float> {
      __m256 value;
      SPMD_AVX2 varying() noexcept : value() {}
      SPMD_AVX2 varying(float rhs) noexcept : value(_mm256_set1_ps(rhs)) {}
      SPMD_AVX2 varying(const varying & rhs) noexcept : value(rhs.value) {}
      SPMD_AVX2 explicit varying(const __m256 & value) noexcept : value(value) {}
      float & item(int i) noexcept { return reinterpret_cast<float*>(&value)[i]; }
      const float & item(int i) const noexcept { return reinterpret_cast<const float*>(&value)[i]; }
      SPMD_AVX2 varying masked() const noexcept { return varying(_mm256_blendv_ps(_mm256_setzero_ps(), value, _mm256_castsi256_ps(execution_mask.value))); }
      SPMD_AVX2 varying operator + (const varying & rhs) const noexcept {
        return varying(_mm256_add_ps(value,rhs.value));
      }
      SPMD_AVX2 varying operator - (const varying & rhs) const noexcept {
        return varying(_mm256_sub_ps(value,rhs.value));
      }
      SPMD_AVX2 varying operator * (const varying & rhs) const noexcept {
        return varying(_mm256_mul_ps(value,rhs.value));
      }
      SPMD_AVX2 varying operator / (const varying & rhs) const noexcept {
        return varying(_mm256_div_ps(value,rhs.value));
      }
      SPMD_AVX2 varying & operator += (const varying & rhs) noexcept {
        return (*this = (*this) + rhs);
      }
      varying & operator -= (const varying & rhs) noexcept {
        return (*this = (*this) - rhs);
      }
      varying & operator *= (const varying & rhs) noexcept {
        return (*this = (*this) * rhs);
      }
      varying & operator /= (const varying & rhs) noexcept {
        return (*this = (*this) / rhs);
      }
      SPMD_AVX2 varying & operator ++ () noexcept {
        value = _mm256_add_ps(value,_mm256_set1_ps(1.f));
        return *this;
      }
      SPMD_AVX2 varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm256_add_ps(value,_mm256_set1_ps(1.f));
        return varying(old);
      }
      SPMD_AVX2 varying & operator -- () noexcept {
        value = _mm256_sub_ps(value,_mm256_set1_ps(1.f));
        return *this;
      }
      SPMD_AVX2 varying operator -- (int) noexcept {
        auto old = value;
        value = _mm256_sub_ps(value,_mm256_set1_ps(1.f));
        return varying(old);
      }
      SPMD_AVX2 varying sin() const noexcept {
        return varying(detail::sin256_ps(value));
      }
      SPMD_AVX2 varying cos() const noexcept {
        return varying(detail::cos256_ps(value));
      }
      SPMD_AVX2 varying log() const noexcept {
        return varying(detail::log256_ps(value));
      }
      SPMD_AVX2 varying exp() const noexcept {
        return varying(detail::exp256_ps(value));
      }
      SPMD_AVX2 void sincos(varying & s, varying & c) const noexcept {
        detail::sincos256_ps(value, &s.value, &c.value);
      }
    };

    template <> struct varying<int> {
      __m256i value;
      SPMD_AVX2 varying<int> (int a, int b, int c, int d, int e, int f, int g, int h) noexcept : value(
        _mm256_set_ps(a,b,c,d,e,f,g,h)
      ) {}
      SPMD_AVX2 varying<int> (const varying<int&> & ref) noexcept : value(
        // gather
#ifdef SPMD_64
        _mm256_set_m128i(
          _mm256_mask_i64gather_epi32(
            _mm_setzero_si128()
          , nullptr
          , ref.value[1]
          , _mm256_extracti128_si256(execution_mask.value,1) // upper half of mask
          , 1
          )
        , _mm256_mask_i64gather_epi32(
            _mm_setzero_si128()
          , nullptr // base
          , ref.value[0]
          , _mm256_extracti128_si256(execution_mask.value,0) // low half of mask
          , 1 // stride
          )
        )
#else
        _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), nullptr, value, execution_mask.value, 1)
#endif
      ) {}
      SPMD_AVX2 varying() noexcept : value() {}
      SPMD_AVX2 varying(int rhs) noexcept : value(_mm256_set1_epi32(rhs)) {}
      SPMD_AVX2 varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit SPMD_AVX2 varying(const __m256i & value) noexcept : value(value) {}
      SPMD_AVX2 int & item(int i) { return reinterpret_cast<int*>(&value)[i]; }
      SPMD_AVX2 const int & item(int i) const { return reinterpret_cast<const int*>(&value)[i]; }
      SPMD_AVX2 varying masked() const { return varying(value & execution_mask.value); }
      SPMD_AVX2 varying operator + (const varying & rhs) const noexcept {
        return varying(_mm256_add_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying operator - (const varying & rhs) const noexcept {
        return varying(_mm256_sub_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying operator * (const varying & rhs) const noexcept {
        return varying(_mm256_mullo_epi32(value,rhs.value));
      }
      //varying operator / (const varying & rhs) const noexcept {
      //  return varying(_mm256_div_epi32(value,rhs.value));
      //}
      //varying operator % (const varying & rhs) const noexcept {
      //  return varying(_mm256_rem_epi32(value,rhs.value));
      //}
      SPMD_AVX2 varying operator && (const varying & rhs) const noexcept {
        return varying(_mm256_and_si256(value,rhs.value));
      }
      SPMD_AVX2 varying operator || (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      SPMD_AVX2 varying operator & (const varying & rhs) const noexcept {
        return varying(_mm256_and_si256(value,rhs.value));
      }
      SPMD_AVX2 varying operator | (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      SPMD_AVX2 varying operator ^ (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      SPMD_AVX2 varying operator ~ () const noexcept {
        return varying(_mm256_xor_si256(value,_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())));
      }
      SPMD_AVX2 varying operator ! () const noexcept {
        return varying(_mm256_xor_si256(value,_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())));
      }
      SPMD_AVX2 varying & operator = (const varying & rhs) noexcept {
         value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
      SPMD_AVX2 varying & operator += (const varying & rhs) noexcept {
        return (*this = (*this) + rhs);
      }
      SPMD_AVX2 varying & operator -= (const varying & rhs) noexcept {
        return (*this = (*this) - rhs);
      }
      SPMD_AVX2 varying & operator *= (const varying & rhs) noexcept {
        return (*this = (*this) * rhs);
      }
      //varying & operator /= (const varying & rhs) noexcept {
      //  return (*this = (*this) / rhs);
      //}
      //varying & operator %= (const varying & rhs) noexcept {
      //  return (*this = (*this) % rhs);
      //}
      SPMD_AVX2 varying & operator &= (const varying & rhs) noexcept {
        return (*this = (*this) & rhs);
      }
      SPMD_AVX2 varying & operator |= (const varying & rhs) noexcept {
        return (*this = (*this) | rhs);
      }
      SPMD_AVX2 varying & operator ++ () noexcept {
        value = _mm256_add_epi32(value,_mm256_set1_epi32(1));
        return *this;
      }
      SPMD_AVX2 varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm256_add_epi32(value,_mm256_set1_epi32(1));
        return varying(old);
      }
      SPMD_AVX2 varying & operator -- () noexcept {
        value = _mm256_sub_epi32(value,_mm256_set1_epi32(1));
        return *this;
      }
      SPMD_AVX2 varying operator -- (int) noexcept {
        auto old = value;
        value = _mm256_sub_epi32(value,_mm256_set1_epi32(1));
        return varying(old);
      }
      SPMD_AVX2 varying<bool> operator == (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpeq_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying<bool> operator != (const varying & rhs) const noexcept {
        return !varying<bool>(_mm256_cmpeq_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying<bool> operator <= (const varying & rhs) const noexcept {
        return !varying<bool>(_mm256_cmpgt_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying<bool> operator >= (const varying & rhs) const noexcept {
        return !varying<bool>(_mm256_cmpgt_epi32(rhs.value,value));
      }
      SPMD_AVX2 varying<bool> operator < (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpgt_epi32(rhs.value,value));
      }
      SPMD_AVX2 varying<bool> operator > (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpgt_epi32(value,rhs.value));
      }
      SPMD_AVX2 varying<bool> operator () () noexcept {
        return !varying<bool>(_mm256_cmpeq_epi32(value,_mm256_setzero_si256()));
      }
    };


    // TODO
    // struct varying<double> {
    //  __m256 value[2];
    //};

    // raii, copy existing execution mask, restore on exit
    struct execution_mask_scope {
      mask old_execution_mask;
      SPMD_AVX2 execution_mask_scope() : old_execution_mask(execution_mask) {}
      SPMD_AVX2 execution_mask_scope(varying<bool> cond) : old_execution_mask(execution_mask) { execution_mask &= cond; }
      SPMD_AVX2 ~execution_mask_scope() { execution_mask = old_execution_mask; }

      // if we masked off some threads since we started, flip the mask to just the others.
      SPMD_AVX2 void flip() const noexcept {
        execution_mask.value = _mm256_andnot_si256(execution_mask.value, old_execution_mask.value);
      }
    };

    template <typename T> static inline void if_(bool cond, T then_branch) {
      if (cond) then_branch();
    }

    // varying if_
    template <typename T> static inline void SPMD_AVX2 if_(varying<bool> cond, T then_branch) {
      execution_mask_scope scope(cond);
      if (execution_mask.any()) then_branch();
    }

    // boring uniform if_
    template <typename T, typename F> static inline void if_(bool cond, T then_branch, F else_branch) {
      if (cond) then_branch();
      else else_branch();
    }

    // varying if_
    template <typename T, typename F> static SPMD_AVX2 inline void if_(varying<bool> cond, T then_branch, F else_branch) {
      execution_mask_scope scope(cond);
      if (execution_mask.any()) then_branch();
      scope.flip();
      if (execution_mask.any()) else_branch();
    }

    // linear structures are 'uniform'ish, w/ constant offsets
    template <typename T> struct linear;

    static const varying<int> programIndex = varying<int>(7,6,5,4,3,2,1,0);
    static const int programCount = 8; // number of programs in a warp

    // linear int, useful for indices.
    template <> struct linear<int> {
      int base;
      explicit linear(int base) noexcept : base(base) {}
      linear(const linear & rhs) : base(rhs.base) {}
      linear(linear && rhs) : base(std::forward<int>(rhs.base)) {}

      template <typename T> SPMD_AVX2 varying<int> operator[](int const * p) const noexcept {
        // perform a masked load. This prevents reading past end of page, or doing other bad things out of bounds
        return varying<int>(_mm256_maskload_epi32(p + base, execution_mask.value));
      }

      // downgrade to a varying int
      SPMD_AVX2 operator varying<int> () const noexcept {
        // return varying<int>(_mm256_add_ps(_mm256_set_ps(7,6,5,4,3,2,1,0), _mm256_set1_epi32(base)));
        return programIndex + varying<int>(base);
      };
      linear operator + (int i) const noexcept {
        return linear(base + i);
      }
      linear operator - (int i) const noexcept {
        return linear(base - i);
      }
      linear & operator ++ () noexcept {
        ++base;
        return *this;
      }
      linear operator ++ (int) noexcept {
        return linear(base++);
      }
      linear & operator -- () noexcept {
        --base;
        return *this;
      }
      linear operator -- (int) noexcept {
        return linear(base--);
      }
      linear & operator = (linear & rhs) {
        base = rhs.base;
        return *this;
      }
      linear & operator += (int rhs) {
        base += rhs;
        return *this;
      }
      linear & operator -= (int rhs) {
        base -= rhs;
        return *this;
      }
    };


/*
    template <typename F> void foreach(int i, int j, F fun) {
      // reset the execution mask here
      // execution_mask.
      for (linear<int> t = linear(i);t.base < steps - 7; t += 8) {
         fun(t);
      }
      // now toggle off the mask
    }
*/

    static inline linear<int> operator + (int i, linear<int> j) noexcept {
      return linear<int>(i + j.base);
    }

    struct kernel {
      // detect avx2
      static inline bool available() {
        auto i = cpu::system_isa();
        return i >= cpu::isa::avx2 && i <= cpu::isa::max_intel;
      }
      typedef avx2::mask mask;
      typedef avx2::execution_mask_scope execution_mask_scope;
      template <typename T> using varying = avx2::varying<T>;
      template <typename T> using linear = avx2::linear<T>;
      template <typename ... Ts> static void if_(Ts ... ts) {
        avx2::if_<Ts...>(std::forward(ts)...);
      }
    };
  };
}
