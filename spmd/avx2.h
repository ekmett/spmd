#pragma once

#include <immintrin.h>
#include <utility>
#include "cpu.h"

namespace spmd {
  // spmd on simd computation kernel using avx2
  template <bool allow_slow_path = false>
  struct avx2 { 
    static const int items = 8;
 
    // detect avx2
    static inline bool available() {
      auto i = cpu::system_isa();
      return i >= cpu::isa::avx2 && i <= cpu::isa::max_intel;
    }
  
    // forward template declaration
    template <typename T> struct varying;

    // execution mask
    struct mask {
      __m256i value;
  
      // default to initialized.
      mask() noexcept : value(_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())) {}
      mask(bool b) noexcept : value(_mm256_set1_epi32(b ? ~0 : 0)) {}
      mask(__m256i value) noexcept : value(value) {}
      mask(const mask & m) noexcept : value(m.value) {}
      // mask(mask && m) noexcept : value(std::forward(m.value)) {} // move constructor
  
      static mask on() noexcept { return mask(); }
      static mask off() noexcept { return _mm256_setzero_si256(); }
  
      // no threads in the "warp" are active
      bool any() const noexcept {
        return _mm256_movemask_ps(_mm256_castsi256_ps(value)) != 0;
      }
  
      // all 8 threads in the "warp" are active, fast path
      bool all() const noexcept {
        return _mm256_movemask_ps(_mm256_castsi256_ps(value)) == 0xff;
      }
  
      mask operator & (const mask & that) const noexcept {
        return _mm256_and_si256(value, that.value);
      }
  
      mask operator | (const mask & that) const noexcept {
        return _mm256_or_si256(value, that.value);
      }

      mask operator ^ (const mask & that) const noexcept {
        return _mm256_xor_si256(value, that.value);
      }
  
      mask & operator &= (const mask & rhs) noexcept {
        value = _mm256_and_si256(value, rhs.value);
        return *this;
      }
  
      mask & operator |= (const mask & rhs) noexcept {
        value = _mm256_or_si256(value, rhs.value);
        return *this;
      }

      mask & operator ^= (const mask & rhs) noexcept {
        value = _mm256_or_si256(value, rhs.value);
        return *this;
      }
  
      // friend mask & operator &= (const varying<bool> & rhs) noexcept;

      mask operator ~ () noexcept {
        return (*this) ^ on();
      }
    };

    static thread_local mask execution_mask; // shared execution mask

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
        execution_mask.each([&](int i) { value[i] = f(i); }
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
      varying<bool> operator == (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] == rhs.value[i]; });
      }
      varying<bool> operator != (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] != rhs.value[i]; });
      }
      varying<bool> operator <= (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] <= rhs.value[i]; });
      }
      varying<bool> operator >= (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] >= rhs.value[i]; });
      }
      varying<bool> operator < (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] < rhs.value[i]; });
      }
      varying<bool> operator > (const varying & rhs) const {
        return varying<bool>::make([&](int i) { return value[i] > rhs.value[i]; });
      }
    };
  
    // varying<bool>
    template<>
    struct varying<bool> {
      __m256i value;
  
      varying() noexcept {}
      varying(bool b) noexcept : value(_mm256_set1_epi32(b ? ~0 : 0)) {}
      varying(__m256i value) noexcept : value(value) {}
      varying(const varying & that) noexcept : value(that.value) {} // copy constructor
  
      varying masked() const noexcept {
        return varying {
          _mm256_castps_si256(_mm256_blendv_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(value), _mm256_castsi256_ps(execution_mask.value)))
        };
      }
  
      varying operator != (const varying & rhs) const noexcept { return _mm256_xor_si256(value,rhs.value); }
      varying operator < (const varying & rhs) const noexcept { return _mm256_andnot_si256(value,rhs.value); }
      varying operator > (const varying & rhs) const noexcept { return _mm256_andnot_si256(rhs.value,value); }
      varying operator == (const varying & rhs) const noexcept { return ! ((*this) != rhs); }
      varying operator <= (const varying & rhs) const noexcept { return (!(*this)) || rhs; }
      varying operator >= (const varying & rhs) const noexcept { return (*this) || (!rhs); }
      varying operator || (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      varying operator && (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      varying operator | (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      varying operator & (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      varying operator ! () const noexcept { return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), value); }
  
      // masked assignment operator
      varying & operator = (const varying & rhs) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
  
      varying & operator|=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_or_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
  
      varying & operator&=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_and_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
 
      // varying<bool> reference to a slot in a simd'd boolean
      struct item_ref {
         varying<bool> & m;
         int i;
         item_ref() = delete;
         item_ref(const item_ref &) = delete;
         item_ref(item_ref &&) = delete;
         explicit item_ref(varying<bool> & m, int i) noexcept : m(m), i(i) {}
         bool operator() const noexcept {
           return m.value.m256i_i32[i] != 0;
         }
         item_ref & operator = (bool b) noexcept { 
           m.value.m256i_i32[i] = b ? ~0 : 0;
         }
         item_ref & operator &= (bool b) noexcept {
           m.value.m256i_i32[i] &= b ? ~0 : 0;
         }
         item_ref | operator |= (bool b) noexcept {
           m.value.m256i_i32[i] |= b ? ~0 : 0;
         }
         item_ref | operator ^= (bool b) noexcept {
           m.value.m256i_i32[i] ^= b ? ~0 : 0;
         }
      };

      struct item_ptr {
        varying<bool> * m;
        int i;
        item_ptr() noexcept : m(nullptr, 0) {}
        item_ptr(const item_ptr & p) noexcept : m(p.m), i(p.i) {}
        item_ptr(item_ptr && p) noexcept : m(std::move(p.m)), i(std::move(p.i)) {}
        exlicit item_ptr(varying<bool> * m, int i) noexcept : m(m), i(i) {}
        item_ptr & operator = (const item_ptr & rhs) noexcept {
          m = rhs.m
          i = rhs.i
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

      const item_ptr operator & (const item_ref & ref) { 
        return item_ptr(ref.m,ref.i);
      }

      item_ptr operator & (item_ref & ref) { 
        return item_ptr(ref.m,ref.i);
      }

      const item_ref operator * (const item_ptr & ptr) { 
        return item_ref(ptr.m,ptr.i);
      }

      item_ref operator * (item_ptr & ptr) { 
        return item_ref(ptr.m,ptr.i);
      }

      item_ref item(int i) {
        return item_ref(*this,i);
      }
      const item_ref item(int i) const {
        return item_ref(*this,i);
      }
    }; // varying<bool>
  
    mask & operator &= (mask & lhs, const varying<bool> & rhs) noexcept {
      lhs.value = _mm256_and_si256(lhs.value, rhs.value);
      return lhs;
    }

    template <typename T> varying<T> operator () (const varying<T&> & ref) noexcept {
      varying<T> result;
      execution_mask.each([&](int i) { result.item(i) = ref.item(i); }); 
      return result;
    }

    template <typename T> 
    varying<T&> & operator =(varying<T&> & lhs, const varying<T> & rhs) {
      execution_mask.each([&](int i) { lhs.item(i) = rhs.item(i); })
      return lhs;
    }
  
#ifdef SPMD_64
    template <typename T> struct varying<T*> {
      __m256i value[2];
      explicit varying(const __m256i value[2]) noexcept : value { value[0],value[1] } {}
      varying & operator =(const varying & that) noexcept {
        for (int i : {0, 1}) {
          value[i] = _mm256_castps_si256(
            _mm256_blendv_ps(
              _mm256_castsi256_ps(value[i]),
              _mm256_castsi256_ps(that.value[i]),
              _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(execution_mask.value,i)))
            )
          );
        }
        return *this;
      }
      T * & item(int i) { return reinterpret_cast<T**>(value)[i]; }
      const T * & item(int i) const { return reinterpret_cast<T**>(value)[i]; }
    };
  
    template <typename T> struct varying<T&> {
      __m256i value[2];
      explicit varying(const __m256i value[2]) noexcept : value { value[0], value[1] } {}
      varying & operator =(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) = rhs.item(i); });
        return lhs;
      }
      varying & operator +=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) += rhs.item(i); });
        return lhs;
      }
      varying & operator -=( const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) -= rhs.item(i); });
        return lhs;
      }
      varying & operator *=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) *= rhs.item(i); });
        return lhs;
      }
      varying & operator /=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) /= rhs.item(i); });
        return lhs;
      }
      varying & operator &=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) &= rhs.item(i); });
        return lhs;
      }
      varying & operator |=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) |= rhs.item(i); });
        return lhs;
      }
      T & item(int i) { return *(reinterpret_cast<T**>(value)[i]); }
      const T & item(int i) const { return *(reinterpret_cast<T**>(value)[i]); }
    };

    template <typename T>
    varying<T*> operator & (const varying<T&> & r) const noexcept {
      return varying<T*>(r.value);
    }
  
    template <typename T>
    varying<T&> operator * (const varying<T*> & p) const noexcept { 
      return varying<T&>(p.value);
    }
  
#else
    template <typename T> struct varying<T*> {
      __m256i value;
      explicit varying(const __m256i value) noexcept : value(value) {}
      varying & operator =(const varying<T*> & that) noexcept { 
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }
      T * & item(int i) { return reinterpret_cast<T*&>(value.m256i_i32[i]); }
      const T * & item(int i) const { return reinterpret_cast<const T*&>(value.m256i_i32[i]); }
    };

    template <typename T> struct varying<T&> {
      __m256i value;
      explicit varying(const __m256i value) noexcept : value(value) {}
      T & item(int i) { return *(reinterpret_cast<T*>(value.m256i_i32[i])); }
      const T & item(int i) const { return *(reinterpret_cast<const T*>(value.m256i_i32[i])); }
      varying & operator =(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) = rhs.item(i); });
        return lhs;
      }
      varying & operator +=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) += rhs.item(i); });
        return lhs;
      }
      varying & operator -=( const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) -= rhs.item(i); });
        return lhs;
      }
      varying & operator *=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) *= rhs.item(i); });
        return lhs;
      }
      varying & operator /=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) /= rhs.item(i); });
        return lhs;
      }
      varying & operator &=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) &= rhs.item(i); });
        return lhs;
      }
      varying & operator |=(const varying<T> & rhs) noexcept {
        execution_mask.each([&](int i) { item(i) |= rhs.item(i); });
        return lhs;
      }
    };
#endif

    template <> struct varying<float> {
      __m256 value;
      varying() noexcept : value() {}
      varying(float rhs) noexcept : value(_mm256_set1_ps(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m256 & value) noexcept : value(value) {}
      int & item(int i) { return value.m256_f32[i]; }
      const int & item(int i) const { return value.m256_f32[i]; }
      varying masked() const { return varying(_mm256_blendv_ps(_mm256_setzero_ps(), value, _mm256_castsi256_ps(execution_mask.value))); }
      varying operator + (const varying & rhs) const noexcept {
        return varying(_mm256_add_ps(value,rhs.value));
      }
      varying operator - (const varying & rhs) const noexcept {
        return varying(_mm256_sub_ps(value,rhs.value));
      }
      varying operator * (const varying & rhs) const noexcept {
        return varying(_mm256_mul_ps(value,rhs.value));
      }
      varying operator / (const varying & rhs) const noexcept {
        return varying(_mm256_div_ps(value,rhs.value));
      }
      varying & operator += (const varying & rhs) noexcept {
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
    };
  
    template <> struct varying<int> {
      __m256i value;
      varying() noexcept : value() {}
      varying(int rhs) noexcept : value(_mm256_set1_epi32(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m256i & value) noexcept : value(value) {}
      int & item(int i) { return value.m256_i32[i]; }
      const int & item(int i) const { return value.m256_i32[i]; }
      varying masked() const { return varying(value & execution_mask.value); }
      varying operator + (const varying & rhs) const noexcept {
        return varying(_mm256_add_epi32(value,rhs.value));
      }
      varying operator - (const varying & rhs) const noexcept {
        return varying(_mm256_sub_epi32(value,rhs.value));
      }
      varying operator * (const varying & rhs) const noexcept {
        return varying(_mm256_mullo_epi32(value,rhs.value));
      }
      varying operator / (const varying & rhs) const noexcept {
        return varying(_mm256_div_epi32(value,rhs.value));
      }
      varying operator % (const varying & rhs) const noexcept {
        return varying(_mm256_rem_epi32(value,rhs.value));
      }
      varying operator & (const varying & rhs) const noexcept {
        return varying(_mm256_and_si256(value,rhs.value));
      }
      varying operator | (const varying | rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      varying operator ^ (const varying ^ rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      varying operator ~ () const noexcept {
        return varying(_mm256_xor_si256(value,_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())));
      }
      varying & operator = (const varying & rhs) noexcept {
         value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return this;
      }
      varying & operator += (const varying & rhs) noexcept {
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
      varying & operator %= (const varying & rhs) noexcept {
        return (*this = (*this) % rhs);
      }
      varying & operator &= (const varying & rhs) noexcept {
        return (*this = (*this) & rhs);
      }
      varying & operator |= (const varying & rhs) noexcept {
        return (*this = (*this) | rhs);
      }
      varying & operator ++ () noexcept { 
        value = _mm256_add_epi32(value,_mm256_set1_epi32(1));
        return *this;
      }
      varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm256_add_epi32(value,_mm256_set1_epi32(1));
        return varying(old);
      }
      varying & operator -- () noexcept { 
        value = _mm256_sub_epi32(value,_mm256_set1_epi32(1));
        return *this;
      }
      varying operator -- (int) noexcept {
        auto old = value;
        value = _mm256_sub_epi32(value,_mm256_set1_epi32(1));
        return varying(old);
      }
      varying<bool> operator == (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpeq_epi32(value,rhs.value));
      }
      varying<bool> operator != (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpneq_epi32(value,rhs.value));
      }
      varying<bool> operator <= (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmple_epi32(value,rhs.value));
      }
      varying<bool> operator >= (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpge_epi32(value,rhs.value));
      }
      varying<bool> operator < (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmplt_epi32(value,rhs.value));
      }
      varying<bool> operator > (const varying & rhs) const noexcept {
        return varying<bool>(_mm256_cmpgt_epi32(value,rhs.value));
      }
      varying<bool> operator () () noexcept {
        return varying<bool>(_mm256_cmpneq_epi32(value,_mm256_setzero_si256()));
      }
    };

    // overload varying<int> dereference to use avx2 gathers
    template <> varying<int> operator () (varying<int&> & ref) noexcept {
      __m256i zero = _mm256_setzero_si256();
      return varying<int>(
#ifdef SPMD_64
        _mm256_insertf128_ps(
            _mm256_mask_i64gather_epi32(zero, nullptr, value[0], lo_mask, 1);
          , _mm256_extracti128_si256(_mm256_mask_i64gather_epi32(zero, nullptr, value[1], hi_mask, 1),0)
          , 1
          )
#else
        _mm256_mask_i32gather_epi32(zero, nullptr, value, execution_mask.value, 1)
#endif
      );
    }
  
    // TODO
    struct varying<double> {
      __m256 value[2];
    };

    // raii, copy existing execution mask, restore on exit
    struct execution_mask_scope {
      mask old_execution_mask;
      execution_mask_scope() : old_execution_mask(execution_mask) {}
      execution_mask_scope(varying<bool> cond) : old_execution_mask(execution_mask) { execution_mask &= cond; }
      ~execution_mask_scope() { execution_mask = old_execution_mask; }
  
      // if we masked off some threads since we started, flip the mask to just the others.
      void flip() const noexcept {
        execution_mask.value = _mm256_andnot_si256(execution_mask.value, old_execution_mask.value);
      }
    };

    template <typename T> void if_(bool cond, T then_branch) const {
      if (cond) then_branch();
    } if_;
  
    // varying if_
    template <typename T> void if_(varying<bool> cond, T then_branch) {
      execution_mask_scope scope(cond);
      if (execution_mask.any()) then_branch();
    }
  
    // boring uniform if_
    template <typename T, typename F> void if_(bool cond, T then_branch, F else_branch) {
      if (cond) then_branch();
      else else_branch();
    }
  
    // varying if_
    template <typename T, typename F> void if_(varying<bool> cond, T then_branch, F else_branch) {
      execution_mask_scope scope(cond);
      if (execution_mask.any()) then_branch();
      scope.flip();
      if (execution_mask.any()) else_branch();
    }
  
    // linear structures are 'uniform'ish, w/ constant offsets
    template <typename T> struct linear;
  
    // linear int, useful for indices.
    template <> struct linear<int> {
      int base;
      explicit linear(int base) noexcept : base(base) {}
      linear(const linear & rhs) : base(rhs.base) {}
      linear(linear && rhs) : base(std::forward<int>(rhs.base)) {}
  
      template <typename T> varying<int> operator[](int const * p) const noexcept {
        // perform a masked load. This prevents reading past end of page, or doing other bad things out of bounds
        return _mm256_maskload_epi32(p + base, execution_mask.value);
      }
  
      // downgrade to a varying int
      varying<int> operator()() const noexcept {
        return _mm256_add_ps(_mm256_set_ps(7,6,5,4,3,2,1,0), _mm256_set1_epi32(base));
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
  
    linear<int> operator + (int i, linear<int> j) noexcept {
      return linear<int>(i + j.base);
    }
  };

  namespace target { 
    namespace avx2 { // import this namespace consistently through your code and you can just code like normal
      typedef typename ::spmd::avx2<false> kernel;
      template <typename T> using varying = kernel::varying<T>;
      template <typename T> using linear = kernel::linear<T>;
      typedef typename kernel::execution_mask execution_mask;
      typedef typename kernel::execution_mask_scope execution_mask_scope;
      template <typename ... Ts> void if_(Ts... ts) {
        kernel::if_<Ts...>(std::forward(ts)....);
      }
    }
  }
}
