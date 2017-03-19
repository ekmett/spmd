#pragma once

#include <array>
#include <immintrin.h>
#include <utility>
#include "cpu.h"
#include "avx2_math.h"

// TODO: make a generic item_ref type to share work between instances?

namespace spmd {
    static const int default_width = 8;
    static const bool allow_slow_path = false;

    // forward template declaration
    template <typename T, size_t N = default_width> struct varying;
    // specializations

    // execution mask, primary template
    // model
    template <int N = default_width> struct mask {
      static_assert(N<=32,"vector width too wide");
      static_assert(allow_slow_path,"slow path disabled");
      static const uint32_t all_on = (1<<N)-1;
      uint32_t value;
      mask() noexcept : value(all_on) {}
      mask(bool b) noexcept : value(b ? all_on : 0) {}
      mask(uint32_t value) noexcept : value(value) }
      mask(const mask & m) noexcept : value(m.value) {}
      static mask on() noexcept { return mask(); }
      static mask off() noexcept { return mask(0); }
      bool any() const noexcept { return mask != 0; }
      bool all() const noexcept { return mask == all_on; }
      mask operator & (const mask & that) const noexcept {
        return value & that.value;
      }
      mask operator | (const mask & that) const noexcept {
        return value | that.value;
      }

      mask operator ^ (const mask & that) const noexcept {
        return value ^ that.value;
      }

      mask & operator &= (const mask & that) noexcept {
        value &= that.value;
        return *this;
      }

      mask & operator |= (const mask & that) noexcept {
        value |= that.value;
        return *this;
      }

      mask & operator ^= (const mask & that) noexcept {
        value ^= that.value;
        return *this;
      }

      mask operator ~ () const noexcept {
        return value ^ all_on;
      }

      template <typename F> void each(F f) {
        for (uint32_t m = value;m;) {
          auto i = __builtin_ffs(m);
          f(i);
          m &= ~ (1u << i);
        }
      }

      uint32_t bits() const noexcept {
        return value;
      }

      // active worker count
      int active() const noexcept {
        return __builtin_popcount(value);
      }

      static thread_local mask exec;
    };

#ifdef __AVX__
    template <> struct mask<4> {
      __m128i value;
      // default to initialized.
      mask() noexcept : value(_mm_cmpeq_epi32(_mm_setzero_si128(),_mm256_setzero_si128())) {}
      mask(bool b) noexcept : value(_mm_set1_epi32(b ? ~0 : 0)) {}
      mask(__m256i value) noexcept : value(value) {}
      mask(const mask & m) noexcept : value(m.value) {}
      // mask(mask && m) noexcept : value(std::forward(m.value)) {} // move constructor

      static mask on() noexcept { return mask(); }
      static mask off() noexcept { return _mm_setzero_si128(); }

      // no threads in the "warp" are active
      bool any() const noexcept {
        return _mm_movemask_ps(_mm_castsi128_ps(value)) != 0;
      }

      // all 4 threads in the "warp" are active, fast path
      bool all() const noexcept {
        return _mm_movemask_ps(_mm_castsi128_ps(value)) == 0xf;
      }

      mask operator & (const mask & that) const noexcept {
        return _mm_and_si128(value, that.value);
      }

      mask operator | (const mask & that) const noexcept {
        return _mm_or_si128(value, that.value);
      }

      mask operator ^ (const mask & that) const noexcept {
        return _mm_xor_si128(value, that.value);
      }

      mask & operator &= (const mask & rhs) noexcept {
        value = _mm_and_si128(value, rhs.value);
        return *this;
      }

      mask & operator |= (const mask & rhs) noexcept {
        value = _mm_or_si128(value, rhs.value);
        return *this;
      }

      mask & operator ^= (const mask & rhs) noexcept {
        value = _mm_xor_si128(value, rhs.value);
        return *this;
      }

      mask operator ~ () const noexcept {
        return (*this) ^ on();
      }

      uint32_t bits() const noexcept {
        return _mm_movemask_ps(_mm_castsi128_ps(value));
      }

      template <typename F> void each(F f) {
        for (auto m = bits();m;) {
          auto i = __builtin_ffs(m);
          f(i);
          m &= ~ (1 << i);
        }
      }

      // active worker count
      int active() const noexcept {
        return __builtin_popcount(bits());
      }

      static thread_local mask exec;
    };
#endif

#ifdef __AVX2__
    template <> struct mask<8> {
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
        value = _mm256_xor_si256(value, rhs.value);
        return *this;
      }

      mask operator ~ () noexcept {
        return (*this) ^ on();
      }

      uint32_t bits() const noexcept {
        return _mm256_movemask_ps(_mm256_castsi256_ps(value));
      }

      template <typename F> void each(F f) {
        for (auto m = bits();m;) {
          auto i = __builtin_ffs(m);
          f(i);
          m &= ~ (1 << i);
        }
      }

      // active worker count
      int active() const noexcept {
        return __builtin_popcount(bits());
      }

      static thread_local mask exec;
    };
#endif

    // primary template
    template <typename T, size_t N = default_width>
    struct varying {
      static_assert(allow_slow_path, "slow path disabled");

      std::array<T,items> value;

      varying() {}
      varying(const T & x) {
        mask<N>::exec.each([&](int i) { *(value[i]) = x; });
      }
      varying(std::array<T,items> value) : value(value) {} // not part of the model
      varying(const varying & that) : value(that.value) {}
      varying masked() const { // replace w/ default where mask is disabled
        (~mask<N>::exec).each([&](int i) { value[i] = T(); });
      }

      typedef T & item_ref;
      typedef const T & const_item_ref;
      T & item(int i) noexcept { return value[i]; }
      const T & item(int i) const noexcept { return value[i]; }

      varying & operator = (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] = rhs.value[i]; });
        return *this;
      }
      varying & operator += (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] += rhs.value[i]; });
        return *this;
      }
      varying & operator -= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] -= rhs.value[i]; });
        return *this;
      }
      varying & operator *= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] *= rhs.value[i]; });
        return *this;
      }
      varying & operator /= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] /= rhs.value[i]; });
        return *this;
      }
      varying & operator %= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] %= rhs.value[i]; });
        return *this;
      }
      varying & operator |= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] |= rhs.value[i]; });
        return *this;
      }
      varying & operator &= (const varying & rhs) {
        mask<N>::exec.each([&](int i) { value[i] &= rhs.value[i]; });
        return *this;
      }
    };

    template <size_t N = default_width, typename F, bool force = allow_slow_path> static auto make_varying(F f) -> varying<decltype(f(0)),N> {
      static_assert(force);
      varying<decltype(f(0)),N> result;
      mask<N>::exec.each([&](int i) { result.value[i] = f(i); });
      return result;
    }

    template <typename T, size_t N = default_width>
    varying<T,N> operator + (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) + rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator - (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) - rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator * (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) * rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator / (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) / rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator % (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) % rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator & (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) & rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator | (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) | rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator ^ (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) ^ rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator && (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) && rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator || (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.item(i) || rhs.item(i); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator ~ (const varying<T,N> & that) {
      return make_varying<N>([&](int i) { return ~(that.item(i)); });
    }
    template <typename T, size_t N = default_width>
    varying<T,N> operator ! (const varying<T,N> & that) {
      return make_varying<N>([&](int i) { return !(that.item(i)); });
    }

    template <typename T, size_t N = default_width>
    struct item_ptr {
      varying<T,N> * m;
      int i;
      item_ptr() noexcept : m(nullptr), i(0) {}
      item_ptr(const item_ptr & p) noexcept : m(p.m), i(p.i) {}
      item_ptr(item_ptr && p) noexcept : m(std::move(p.m)), i(std::move(p.i)) {}
      explicit item_ptr(varying<T,N> * m, int i) noexcept : m(m), i(i) {}
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
      item_ptr & operator --() noexcept {
        i = (i + N - 1) % N;
        if (i == N - 1) --m;
        return *this;
      }
      item_ptr & operator ++() noexcept {
        i = (i + 1) % N;
        if (i == 0) ++m;
        return *this;
      }
      item_ptr operator --(int) noexcept {
        item_ptr result(*this);
        --(*this);
        return result;
      }
      item_ptr operator ++(int) noexcept {
        item_ptr result(*this);
        ++(*this);
        return result;
      }
    };

    template <typename T, size_t N = default_width>
    struct const_item_ptr {
      const varying<T,N> * m;
      int i;
      const_item_ptr() noexcept : m(nullptr), i(0) {}
      const_item_ptr(const const_item_ptr & p) noexcept : m(p.m), i(p.i) {}
      const_item_ptr(const_item_ptr && p) noexcept : m(std::move(p.m)), i(std::move(p.i)) {}
      explicit const_item_ptr(varying<T,N> * m, int i) noexcept : m(m), i(i) {}
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
      const_item_ptr & operator --() noexcept {
        i = (i + N - 1) % N;
        if (i == N - 1) --m;
        return *this;
      }
      const_item_ptr & operator ++() noexcept {
        i = (i + 1) % N;
        if (i == 0) ++m;
        return *this;
      }
      const_item_ptr operator --(int) noexcept {
        const_item_ptr result(*this);
        --(*this);
        return result;
      }
      const_item_ptr operator ++(int) noexcept {
        const_item_ptr result(*this);
        ++(*this);
        return result;
      }
    };

    template <typename T, size_t N = default_width> using item_ref = typename varying<T,N>::item_ref;
    template <typename T, size_t N = default_width> using const_item_ref = typename varying<T,N>::const_item_ref;

    // boolean item refs, as booleans are always specialized
    namespace detail {
      template <size_t N = default_width>
      struct bool_item_ref {
        varying<bool,N> & m;
        int i;
        bool_item_ref(varying<bool> & m, int i) noexcept : m(m), i(i) {}
        operator bool() const noexcept {
          return (m.bits() & (1u << i)) != 0;
        }
        // these will be specialized to different widths
        bool_item_ref & operator = (bool b) noexcept;
        bool_item_ref & operator &= (bool b) noexcept;
        bool_item_ref & operator |= (bool b) noexcept;
        bool_item_ref & operator ^= (bool b) noexcept;
      };

      template <size_t N = default_width>
      struct const_bool_item_ref {
        const varying<bool,N> & m;
        int i;
        const_bool_item_ref(const varying<bool> & m, int i) noexcept : m(m), i(i) {}
        operator bool() const noexcept {
          return (m.bits() & (1u << i)) != 0;
        }
      };
    }

    template<typename A, typename B, size_t N = default_width>
    struct varying<std::pair<A,B>,N> {
      typedef A first_type;
      typedef B second_type; 
      varying<A> first;
      varying<B> second;
      varying & operator=(const varying & rhs) { first = rhs.first; second = rhs.second; } 
      void swap(const varying & rhs) { std::swap(first,rhs.first); std::swap(second,rhs.second); }
      // we need an item ref type, etc.
    };

/*
    template <size_t N = default_width, typename ... Ts>
    struct varying<std::tuple<Ts...,N> {
      std::tuple<varying<Ts>...> value;
    };
*/

    template<size_t N = default_width>
    struct varying<bool,N> {
      static_assert(allow_slow_path);
      uint32_t value;
      varying() noexcept {}
      varying(bool b) noexcept : value(b ? ~0 : 0) {}
      varying(const varying & that) noexcept : value(that.value) {}
      varying(uint32_t value) : value(value) {}
      varying masked() const noexcept {
        return varying(value & mask<N>::exec.bits());
      }
      varying operator || (const varying & that) const noexcept { return value | that.value }
      varying operator && (const varying & that) const noexcept { return value & that.value }
      varying operator | (const varying & that) const noexcept { return value | that.value }
      varying operator & (const varying & that) const noexcept { return value & that.value }
      varying operator ! () const noexcept { return ~value; }
      varying operator ~ () const noexcept { return ~value; }

      uint32_t bits () const noexcept { return value & ((1u << N) - 1); }

      varying & operator = (const varying & rhs) noexcept {
        uint32_t m = mask<N>::exec.bits();
        value = (value & ~m) | (rhs.value & m)
        return *this;
      }
      varying & operator &= (const varying & rhs) noexcept {
        uint32_t m = mask<N>::exec.bits();
        value = (value & ~m) | (rhs.value & value & m)
        return *this;
      }
      varying & operator |= (const varying & rhs) noexcept {
        uint32_t m = mask<N>::exec.bits();
        value |= (rhs.value & m);
        return *this;
      }
      varying & operator ^= (const varying & rhs) noexcept {
        uint32_t m = mask<N>::exec.bits();
        value ^= (rhs.value & m);
        return *this;
      }

      typedef detail::bool_item_ref<N> item_ref;
      typedef detail::const_bool_item_ref<N> const_item_ref;
      item_ref item(int i) noexcept { return item_ref(*this,i); }
      const_item_ref item(int i) const noexcept { return const_item_ref(*this,i); }

    };

#ifdef __AVX2__
    // varying<bool>
    template<>
    struct varying<bool,8> {
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

      varying operator || (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      varying operator && (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      varying operator | (const varying& that) const noexcept { return _mm256_or_si256(value, that.value); }
      varying operator & (const varying& that) const noexcept { return _mm256_and_si256(value, that.value); }
      varying operator ! () const noexcept { return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), value); }
      varying operator ~ () const noexcept { return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), value); }

      // masked assignment operator
      varying & operator = (const varying & rhs) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      varying & operator |=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_or_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      varying & operator &=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_and_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      varying & operator ^=(const varying& that) noexcept {
        value = _mm256_castps_si256(
           _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_xor_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
      }

      typedef detail::bool_item_ref<N> item_ref;
      typedef detail::const_bool_item_ref<N> const_item_ref;
      item_ref item(int i) noexcept { return item_ref(*this,i); }
      const_item_ref item(int i) const noexcept { return const_item_ref(*this,i); }

    }; // varying<bool>
#endif

    // instantiate detail
    namespace detail {
      template <size_t N> bool_item_ref<N> & bool_item_ref::operator = (bool b) noexcept {
        static_assert(allow_slow_path);
        uint32_t mask = 1u << i;
        m.value = b ? (m.value | mask) : (m.value & ~mask)
        return *this;
      }
      template <size_t N> bool_item_ref<N> & bool_item_ref::operator &= (bool b) noexcept {
        static_assert(allow_slow_path);
        uint32_t mask = 1u << i;
        m.value = b ? m.value : (m.value & ~mask);
        return *this;
      }
      template <size_t N> bool_item_ref<N> & bool_item_ref::operator |= (bool b) noexcept {
        static_assert(allow_slow_path);
        uint32_t mask = 1u << i;
        m.value = b ? (m.value | mask) : m.value;
        return *this;
      }
      template <size_t N> bool_item_ref<N> & bool_item_ref::operator ^= (bool b) noexcept {
        static_assert(allow_slow_path);
        uint32_t mask = 1u << i;
        m.value = b ? (m.value ^ mask) : m.value;
        return *this;
      }
#ifdef __AVX__
      template <> bool_item_ref<4> & bool_item_ref<4>::operator = (bool b) noexcept {
        __mm_insert_epi32(m.value,b ? ~0 : 0, i);
        return *this;
      }
      template <> bool_item_ref<4> & bool_item_ref<4>::operator &= (bool b) noexcept {
        m.value = __mm_and_si128(m.value,_mm_insert_epi32(m.value,b ? ~0 : 0, i));
        return *this;
      }
      template <> bool_item_ref<4> & bool_item_ref<4>::operator |= (bool b) noexcept {
        m.value = __mm_or_si128(m.value,_mm_insert_epi32(m.value,b ? ~0 : 0, i));
        return *this;
      }
      template <> bool_item_ref<4> & bool_item_ref<4>::operator ^= (bool b) noexcept {
        m.value = __mm_xor_si128(m.value,_mm_insert_epi32(_mm_set1_epi32(0),b ? ~0 : 0, i));
        return *this;
      }
#endif
#ifdef __AVX2__
      template <> bool_item_ref<8> & bool_item_ref<8>::operator = (bool b) noexcept {
        __mm256_insert_epi32(m.value,b ? ~0 : 0, i);
        return *this;
      }
      template <> bool_item_ref<8> & bool_item_ref<8>::operator &= (bool b) noexcept {
        m.value = __mm256_and_si256(m.value,_mm256_insert_epi32(m.value,b ? ~0 : 0, i));
        return *this;
      }
      template <> bool_item_ref<8> & bool_item_ref<8>::operator |= (bool b) noexcept {
        m.value = __mm256_or_si256(m.value,_mm256_insert_epi32(m.value,b ? ~0 : 0, i));
        return *this;
      }
      template <> bool_item_ref<8> & bool_item_ref<8>::operator ^= (bool b) noexcept {
        m.value = __mm256_xor_si256(m.value,_mm256_insert_epi32(_mm256_set1_epi32(0),b ? ~0 : 0, i));
        return *this;
      }
#endif

      template <size_t N = default_width>
      static inline const item_ptr<bool,N> operator & (const item_ref<bool,N> & ref) {
        return item_ptr<bool,N>(&ref.m,ref.i);
      }

      template <size_t N = default_width>
      static inline item_ptr<bool,N> operator & (item_ref<bool,N> & ref) {
        return item_ptr<bool,N>(&ref.m,ref.i);
      }

      template <size_t N = default_width>
      static inline const item_ptr<bool,N> operator * (const item_ptr<bool,N> & ptr) {
        return item_ref<bool,N>(*ptr.m,ptr.i);
      }

      template <size_t N = default_width>
      static inline item_ref<bool,N> operator * (item_ptr<bool,N> & ptr) {
        return item_ref<bool,N>(*ptr.m,ptr.i);
      }

      template <size_t N = default_width>
      static inline const const_item_ptr<bool,N> operator & (const const_item_ref<bool,N> & ref) {
        return const_item_ptr<bool,N>(&ref.m,ref.i);
      }

      template <size_t N = default_width>
      static inline const_item_ptr<bool,N> operator & (const_item_ref<bool,N> & ref) {
        return const_item_ptr<bool,N>(&ref.m,ref.i);
      }

      template <size_t N = default_width>
      static inline const const_item_ptr<bool,N> operator * (const const_item_ptr<bool,N> & ptr) {
        return const_item_ref<bool,N>(*ptr.m,ptr.i);
      }

      template <size_t N = default_width>
      static inline const_item_ref<bool,N> operator * (const_item_ptr<bool,N> & ptr) {
        return const_item_ref<bool,N>(*ptr.m,ptr.i);
      }
    }

    template <typename T, size_t N = default_width> static inline varying<bool,N> operator == (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] == rhs.value[i]; });
    }
    template <typename T, size_t N = default_width> static inline varying<bool,N> operator != (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] != rhs.value[i]; });
    }
    template <typename T, size_t N = default_width> static inline varying<bool,N> operator <= (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] <= rhs.value[i]; });
    }
    template <typename T, size_t N = default_width> static inline varying<bool,N> operator >= (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] >= rhs.value[i]; });
    }
    template <typename T, size_t N = default_width> static inline varying<bool,N> operator < (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] < rhs.value[i]; });
    }
    template <typename T, size_t N = default_width> static inline varying<bool,N> operator > (const varying<T,N> & lhs, const varying<T,N> & rhs) {
      return make_varying<N>([&](int i) { return lhs.value[i] > rhs.value[i]; });
    }

    template <size_t N = default_width>
    static inline varying<bool,N> operator == (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return !(lhs != rhs); }
    template <size_t N = default_width>
    static inline varying<bool,N> operator <= (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return (!lhs) || rhs; }
    template <size_t N = default_width>
    static inline varying<bool,N> operator >= (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return lhs || (!rhs); }

    template <size_t N = default_width>
    static inline varying<bool,N> operator != (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return lhs.value ^ rhs.value; }
    template <size_t N = default_width>
    static inline varying<bool,N> operator < (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return (~lhs.value) & rhs.value; }
    template <size_t N = default_width>
    static inline varying<bool,N> operator > (const varying<bool,N> & lhs, const varying<bool,N> & rhs) noexcept { return (~rhs.value) & lhs.value; }

    template <size_t N = default_width>
    static inline mask<N> & operator &= (mask<N> & lhs, const varying<bool,N> & rhs) noexcept {
      lhs.value = lhs.value & rhs.value;
      return lhs;
    }

#ifdef __AVX__
    static inline varying<bool,4> operator != (const varying<bool,4> & lhs, const varying<bool,4> & rhs) noexcept { return _mm_xor_si128(lhs.value,rhs.value); }
    static inline varying<bool,4> operator < (const varying<bool,4> & lhs, const varying<bool,4> & rhs) noexcept { return _mm_andnot_si128(lhs.value,rhs.value); }
    static inline varying<bool,4> operator > (const varying<bool,4> & lhs, const varying<bool,4> & rhs) noexcept { return _mm_andnot_si128(rhs.value,lhs.value); }
    static inline mask<4> & operator &= (mask & lhs, const varying<bool,4> & rhs) noexcept {
      lhs.value = _mm_and_si128(lhs.value, rhs.value);
      return lhs;
    }
#endif

#ifdef __AVX2__
    static inline varying<bool,8> operator != (const varying<bool,8> & lhs, const varying<bool,8> & rhs) noexcept { return _mm256_xor_si256(lhs.value,rhs.value); }
    static inline varying<bool,8> operator < (const varying<bool,8> & lhs, const varying<bool,8> & rhs) noexcept { return _mm256_andnot_si256(lhs.value,rhs.value); }
    static inline varying<bool,8> operator > (const varying<bool,8> & lhs, const varying<bool,8> & rhs) noexcept { return _mm256_andnot_si256(rhs.value,lhs.value); }
    static inline mask<8> & operator &= (mask & lhs, const varying<bool,8> & rhs) noexcept {
      lhs.value = _mm256_and_si256(lhs.value, rhs.value);
      return lhs;
    }
#endif

#ifdef __AVX__
    template <typename T> struct varying<T*,4> {
#ifdef SPMD_64
#ifdef __AVX2__
      __m256i value; // 64x4 in 1 register
      explicit varying(const __m256i value) noexcept : value (value) {}
      varying & operator = (const varying & that) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(
            _mm256_castsi256_ps(value),
            _mm256_castsi256_ps(that.value),
            _mm256_castsi256_ps(_mm256_cvtepi32_epi64(mask<4>::exec.value))
          )
        );
        return *this;
      }
#else
      __m128i value[2]; // 64x4 in 2 registers
      explicit varying(const __m128i value[2]) noexcept : value (value) {}
      varying & operator = (const varying & that) noexcept {
        // TODO: improve this?
        mask<4>::exec.each([&](int i) { item(i) = that.item(i); }
        return *this;
      }
#endif
#else
      __m128i value; // 32x4
      explicit varying(const __m128i value) noexcept : value (value) {}
      varying & operator = (const varying & that) noexcept {
#ifdef __AVX2__
        value = _mm_castps_si128(
          _mm_blendv_ps(
            _mm_castsi128_ps(value),
            _mm_castsi128_ps(that.value),
            _mm_castsi128_ps(mask<4>::exec.value)
          )
        );
#else
        mask<4>::exec.each([&](int i) { item(i) = that.item(i); }
#endif
        return *this;
      }
#endif
      typedef T * & item_ref;
      typedef const T * & const_item_ref;
      T * & item(int i) { return reinterpret_cast<T**>(value)[i]; }
      const T * & item(int i) const { return reinterpret_cast<const T**>(value)[i]; }
    };
#endif // __AVX__

#ifdef __AVX2__
    template <typename T> struct varying<T*,8> {
#ifdef SPMD_64
      // TODO: allow AVX512 to upgrade this to a single __m512i?
      __m256i value[2];
      explicit varying(const __m256i value[2]) noexcept : value(value) {}
      varying & operator =(const varying & that) noexcept {
        value[0] = _mm256_castps_si256(
          _mm256_blendv_ps(
            _mm256_castsi256_ps(value[0]),
            _mm256_castsi256_ps(that.value[0]),
            _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(mask<8>::exec.value,0)))
          )
        );
        value[1] = _mm256_castps_si256(
          _mm256_blendv_ps(
            _mm256_castsi256_ps(value[1]),
            _mm256_castsi256_ps(that.value[1]),
            _mm256_castsi256_ps(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(mask<8>::exec.value,1)))
          )
        );
        return *this;
      }
#else
      __m256i value;
      explicit varying(const __m256i value) noexcept : value(value) {}
      varying & operator =(const varying<T*> & that) noexcept {
        value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(mask<8>::exec.value))
        );
        return *this;
      }
#endif
      typedef T * & item_ref;
      typedef const T * & const_item_ref;
      T * & item(int i) { return reinterpret_cast<T**>(value)[i]; }
      const T * & item(int i) const { return reinterpret_cast<const T**>(value)[i]; }
    };
#endif

#ifdef __AVX__
    template <typename T> struct varying<T&,4> {
#ifdef SPMD_64
#ifdef __AVX2__
      __m256i value;
      explicit varying(const __m256i value) noexcept : value(value) {}
#else
      __m128i value[2]; // 4x64bit pointers in 2 registers
      explicit varying(const __m128i value[2]) noexcept : value(value) {}
#endif
#else 
      __m128i value; // 4x32bit pointer
      explicit varying(const __m128i value[2]) noexcept : value(value) {}
#endif
      // the "slow" path is the fast path here
      varying & operator =(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) = rhs.item(i); });
        return *this;
      }
      varying & operator +=(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) += rhs.item(i); });
        return *this;
      }
      varying & operator -=( const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) -= rhs.item(i); });
        return *this;
      }
      varying & operator *=(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) *= rhs.item(i); });
        return *this;
      }
      varying & operator /=(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) /= rhs.item(i); });
        return *this;
      }
      varying & operator &=(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) &= rhs.item(i); });
        return *this;
      }
      varying & operator |=(const varying<T> & rhs) {
        mask<4>::exec.each([&](int i) { item(i) |= rhs.item(i); });
        return *this;
      }
      typedef T & item_ref;
      typedef const T & const_item_ref;
      T & item(int i) noexcept { return *(reinterpret_cast<T**>(value)[i]); }
      const T & item(int i) const noexcept { return *(reinterpret_cast<T**>(value)[i]); }

      // gather
      operator varying<T>() const {
        varying<T> result;
        execution_mask.each([&](int i) { result.item(i) = item(i); });
        return result;
      }
    }
#endif

#ifdef __AVX2__
    template <typename T> struct varying<T&,8> {
#ifdef SPMD_64
      __m256i value[2];
      explicit varying(const __m256i value[2]) noexcept : value(value) {}
#else
      __m256i value;
      explicit varying(const __m256i value) noexcept : value(value) {}
#endif
      varying & operator =(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) = rhs.item(i); });
        return *this;
      }
      varying & operator +=(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) += rhs.item(i); });
        return *this;
      }
      varying & operator -=( const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) -= rhs.item(i); });
        return *this;
      }
      varying & operator *=(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) *= rhs.item(i); });
        return *this;
      }
      varying & operator /=(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) /= rhs.item(i); });
        return *this;
      }
      varying & operator &=(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) &= rhs.item(i); });
        return *this;
      }
      varying & operator |=(const varying<T> & rhs) {
        mask<8>::exec.each([&](int i) { item(i) |= rhs.item(i); });
        return *this;
      }
      typedef T & item_ref;
      typedef const T & const_item_ref;
      T & item(int i) noexcept { return *(reinterpret_cast<T**>(value)[i]); }
      const T & item(int i) const noexcept { return *(reinterpret_cast<T**>(value)[i]); }

      operator varying<T>() const {
        varying<T> result;
        execution_mask.each([&](int i) { result.item(i) = item(i); });
        return result;
      }
    };
#endif

    template <typename T, size_t N = default_width>
    varying<T*,N> operator & (const varying<T&,N> & r) noexcept {
      return varying<T*,N>(r.value);
    }

    template <typename T, size_t N = default_width>
    varying<T&,N> operator * (const varying<T*,N> & p) noexcept {
      return varying<T&,N>(p.value);
    }

#ifdef __AVX__
    template <> struct varying<float,4> {
      __m128 value;
      varying() noexcept : value() {}
      varying(float rhs) noexcept : value(_mm_set1_ps(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m256 & value) noexcept : value(value) {}

      typedef float & item_ref;
      typedef const float & const_item_ref;
      float & item(int i) noexcept { return reinterpret_cast<float*>(&value)[i]; }
      const float & item(int i) const noexcept { return reinterpret_cast<const float*>(&value)[i]; }

      varying masked() const noexcept { return varying(_mm_blendv_ps(_mm_setzero_ps(), value, _mm_castsi128_ps(mask<4>::exec.value))); }
      varying operator + (const varying & rhs) const noexcept {
        return varying(_mm_add_ps(value,rhs.value));
      }
      varying operator - (const varying & rhs) const noexcept {
        return varying(_mm_sub_ps(value,rhs.value));
      }
      varying operator * (const varying & rhs) const noexcept {
        return varying(_mm_mul_ps(value,rhs.value));
      }
      varying operator / (const varying & rhs) const noexcept {
        return varying(_mm_div_ps(value,rhs.value));
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
      varying & operator ++ () noexcept {
        value = _mm_add_ps(value,_mm_set1_ps(1.f));
        return *this;
      }
      varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm_add_ps(value,_mm_set1_ps(1.f));
        return varying(old);
      }
      varying & operator -- () noexcept {
        value = _mm_sub_ps(value,_mm_set1_ps(1.f));
        return *this;
      }
      varying operator -- (int) noexcept {
        auto old = value;
        value = _mm_sub_ps(value,_mm_set1_ps(1.f));
        return varying(old);
      }
      // varying sin() const noexcept { return varying(detail::sin128_ps(value)); }
      // varying cos() const noexcept { return varying(detail::cos128_ps(value)); }
      // varying log() const noexcept { return varying(detail::log128_ps(value)); }
      // varying exp() const noexcept { return varying(detail::exp128_ps(value)); }
      // void sincos(varying & s, varying & c) const noexcept { detail::sincos128_ps(value, &s.value, &c.value); }
    };
#endif

#ifdef __AVX2__
    template <> struct varying<float,8> {
      __m256 value;
      varying() noexcept : value() {}
      varying(float rhs) noexcept : value(_mm256_set1_ps(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m256 & value) noexcept : value(value) {}
      float & item(int i) noexcept { return reinterpret_cast<float*>(&value)[i]; }
      const float & item(int i) const noexcept { return reinterpret_cast<const float*>(&value)[i]; }
      varying masked() const noexcept { return varying(_mm256_blendv_ps(_mm256_setzero_ps(), value, _mm256_castsi256_ps(execution_mask.value))); }
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
      varying & operator ++ () noexcept {
        value = _mm256_add_ps(value,_mm256_set1_ps(1.f));
        return *this;
      }
      varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm256_add_ps(value,_mm256_set1_ps(1.f));
        return varying(old);
      }
      varying & operator -- () noexcept {
        value = _mm256_sub_ps(value,_mm256_set1_ps(1.f));
        return *this;
      }
      varying operator -- (int) noexcept {
        auto old = value;
        value = _mm256_sub_ps(value,_mm256_set1_ps(1.f));
        return varying(old);
      }

      // placeholder mathematical operations

      varying sin() const noexcept {
        return varying(detail::sin256_ps(value));
      }
      varying cos() const noexcept {
        return varying(detail::cos256_ps(value));
      }
      varying log() const noexcept {
        return varying(detail::log256_ps(value));
      }
      varying exp() const noexcept {
        return varying(detail::exp256_ps(value));
      }
      void sincos(varying & s, varying & c) const noexcept {
        detail::sincos256_ps(value, &s.value, &c.value);
      }
    };
#endif


#ifdef __AVX__
    template <> struct varying<int32_t,4> {
      __m128i value;
      varying(int32_t a, int32_t b, int32_t c, int32_t d) noexcept : value(
        _mm_set_ps(a,b,c,d)
      ) {}
      varying() noexcept : value() {}
      varying(int32_t rhs) noexcept : value(_mm_set1_epi32(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m128i & value) noexcept : value(value) {}

      typedef int32_t & item_ref;
      typedef const int32_t & const_item_ref;
      int32_t & item(int i) { return reinterpret_cast<int32_t*>(&value)[i]; }
      const int32_t & item(int i) const { return reinterpret_cast<const int32_t*>(&value)[i]; }

      varying masked() const { return varying(__mm_and_si128(value,mask<4>::exec.value)); }

      varying operator + (const varying & rhs) const noexcept {
        return varying(_mm_add_epi32(value,rhs.value));
      }
      varying operator - (const varying & rhs) const noexcept {
        return varying(_mm_sub_epi32(value,rhs.value));
      }
      varying operator * (const varying & rhs) const noexcept {
        return varying(_mm_mullo_epi32(value,rhs.value));
      }
      //varying operator / (const varying & rhs) const noexcept {
      //  return varying(_mm_div_epi32(value,rhs.value));
      //}
      //varying operator % (const varying & rhs) const noexcept {
      //  return varying(_mm_rem_epi32(value,rhs.value));
      //}
      varying operator && (const varying & rhs) const noexcept {
        return varying(_mm_and_si128(value,rhs.value));
      }
      varying operator || (const varying & rhs) const noexcept {
        return varying(_mm_or_si128(value,rhs.value));
      }
      varying operator & (const varying & rhs) const noexcept {
        return varying(_mm_and_si128(value,rhs.value));
      }
      varying operator | (const varying & rhs) const noexcept {
        return varying(_mm_or_si128(value,rhs.value));
      }
      varying operator ^ (const varying & rhs) const noexcept {
        return varying(_mm_or_si128(value,rhs.value));
      }
      varying operator ~ () const noexcept {
        return varying(_mm_xor_si128(value,_mm_cmpeq_epi32(_mm_setzero_si128(),_mm_setzero_si128())));
      }
      varying operator ! () const noexcept {
        return varying(_mm_xor_si128(value,_mm_cmpeq_epi32(_mm_setzero_si128(),_mm_setzero_si128())));
      }
      varying & operator = (const varying & rhs) noexcept {
         value = _mm_castps_si128(
          _mm_blendv_ps(_mm_castsi128_ps(value), _mm_castsi128_ps(rhs.value), _mm_castsi128_ps(mask<4>::exec.value))
        );
        return *this;
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
      //varying & operator /= (const varying & rhs) noexcept {
      //  return (*this = (*this) / rhs);
      //}
      //varying & operator %= (const varying & rhs) noexcept {
      //  return (*this = (*this) % rhs);
      //}
      varying & operator &= (const varying & rhs) noexcept {
        return (*this = (*this) & rhs);
      }
      varying & operator |= (const varying & rhs) noexcept {
        return (*this = (*this) | rhs);
      }
      varying & operator ++ () noexcept {
        value = _mm_add_epi32(value,_mm_set1_epi32(1));
        return *this;
      }
      varying operator ++ (int) noexcept {
        auto old = value;
        value = _mm_add_epi32(value,_mm_set1_epi32(1));
        return varying(old);
      }
      varying & operator -- () noexcept {
        value = _mm_sub_epi32(value,_mm_set1_epi32(1));
        return *this;
      }
      varying operator -- (int) noexcept {
        auto old = value;
        value = _mm_sub_epi32(value,_mm_set1_epi32(1));
        return varying(old);
      }
      varying<bool,8> operator == (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm_cmpeq_epi32(value,rhs.value));
      }
      varying<bool,8> operator != (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm_cmpeq_epi32(value,rhs.value));
      }
      varying<bool,8> operator <= (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm_cmpgt_epi32(value,rhs.value));
      }
      varying<bool,8> operator >= (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm_cmpgt_epi32(rhs.value,value));
      }
      varying<bool,8> operator < (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm_cmpgt_epi32(rhs.value,value));
      }
      varying<bool,8> operator > (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm_cmpgt_epi32(value,rhs.value));
      }
      varying<bool,8> operator () () noexcept {
        return !varying<bool,8>(_mm_cmpeq_epi32(value,_mm_setzero_si128()));
      }
    };
#endif

#ifdef __AVX2__
    template <> struct varying<int32_t,8> {
      __m256i value;
      varying(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e, int32_t f, int32_t g, int32_t h) noexcept : value(
        _mm256_set_ps(a,b,c,d,e,f,g,h)
      ) {}
      varying(const varying<int32_t&,8> & ref) noexcept : value(
        // gather
#ifdef SPMD_64
        _mm256_set_m128i(
          _mm256_mask_i64gather_epi32(
            _mm_setzero_si128()
          , nullptr
          , ref.value[1]
          , _mm256_extracti128_si256(mask<8>::exec.value,1) // upper half of mask
          , 1
          )
        , _mm256_mask_i64gather_epi32(
            _mm_setzero_si128()
          , nullptr // base
          , ref.value[0]
          , _mm256_extracti128_si256(mask<8>::exec.value,0) // low half of mask
          , 1 // stride
          )
        )
#else
        _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), nullptr, value, mask<8>::exec.value, 1)
#endif
      ) {}
      varying() noexcept : value() {}
      varying(int32_t rhs) noexcept : value(_mm256_set1_epi32(rhs)) {}
      varying(const varying & rhs) noexcept : value(rhs.value) {}
      explicit varying(const __m256i & value) noexcept : value(value) {}

      typedef int32_t & item_ref;
      typedef const int32_t & const_item_ref;
      int32_t & item(int i) { return reinterpret_cast<int32_t*>(&value)[i]; }
      const int32_t & item(int i) const { return reinterpret_cast<const int32_t*>(&value)[i]; }

      varying masked() const { return varying(__mm256_and_si256(value,mask<8>::exec.value)); }

      varying operator + (const varying & rhs) const noexcept {
        return varying(_mm256_add_epi32(value,rhs.value));
      }
      varying operator - (const varying & rhs) const noexcept {
        return varying(_mm256_sub_epi32(value,rhs.value));
      }
      varying operator * (const varying & rhs) const noexcept {
        return varying(_mm256_mullo_epi32(value,rhs.value));
      }
      //varying operator / (const varying & rhs) const noexcept {
      //  return varying(_mm256_div_epi32(value,rhs.value));
      //}
      //varying operator % (const varying & rhs) const noexcept {
      //  return varying(_mm256_rem_epi32(value,rhs.value));
      //}
      varying operator && (const varying & rhs) const noexcept {
        return varying(_mm256_and_si256(value,rhs.value));
      }
      varying operator || (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      varying operator & (const varying & rhs) const noexcept {
        return varying(_mm256_and_si256(value,rhs.value));
      }
      varying operator | (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      varying operator ^ (const varying & rhs) const noexcept {
        return varying(_mm256_or_si256(value,rhs.value));
      }
      varying operator ~ () const noexcept {
        return varying(_mm256_xor_si256(value,_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())));
      }
      varying operator ! () const noexcept {
        return varying(_mm256_xor_si256(value,_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())));
      }
      varying & operator = (const varying & rhs) noexcept {
         value = _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
        );
        return *this;
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
      //varying & operator /= (const varying & rhs) noexcept {
      //  return (*this = (*this) / rhs);
      //}
      //varying & operator %= (const varying & rhs) noexcept {
      //  return (*this = (*this) % rhs);
      //}
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
      varying<bool,8> operator == (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm256_cmpeq_epi32(value,rhs.value));
      }
      varying<bool,8> operator != (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm256_cmpeq_epi32(value,rhs.value));
      }
      varying<bool,8> operator <= (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm256_cmpgt_epi32(value,rhs.value));
      }
      varying<bool,8> operator >= (const varying & rhs) const noexcept {
        return !varying<bool,8>(_mm256_cmpgt_epi32(rhs.value,value));
      }
      varying<bool,8> operator < (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm256_cmpgt_epi32(rhs.value,value));
      }
      varying<bool,8> operator > (const varying & rhs) const noexcept {
        return varying<bool,8>(_mm256_cmpgt_epi32(value,rhs.value));
      }
      varying<bool,8> operator () () noexcept {
        return !varying<bool,8>(_mm256_cmpeq_epi32(value,_mm256_setzero_si256()));
      }
    };
#endif

    // TODO
    // struct varying<double> {
    //  __m256 value[2];
    //};

    // raii, copy existing execution mask, restore on exit
    template <size_t N = default_size>
    struct execution_mask_scope {
      mask<N> old_execution_mask;
      execution_mask_scope() : old_execution_mask(mask<N>::exec) {}
      execution_mask_scope(varying<bool> cond) : old_execution_mask(mask<N>::exec) { mask<N>::exec &= cond; }
      ~execution_mask_scope() { mask<N>::exec = old_execution_mask; }

      // if we masked off some threads since we started, flip the mask to just the others.
      void flip() const noexcept {
        mask<N>::exec.value = _mm256_andnot_si256(execution_mask.value, old_execution_mask.value);
      }
    };

    template <size_t N> void execution_mask_scope<N>::flip() const noexcept {
      mask<N>::exec = ~mask<N>::exec & old_execution_mask;
    }

#ifdef __AVX__
    template <> void execution_mask_scope<4>::flip() const noexcept {
      mask<4>::exec.value = _mm_andnot_si128(execution_mask.value, old_execution_mask.value);
    }
#endif

#ifdef __AVX2__
    template <> void execution_mask_scope<8>::flip() const noexcept {
      mask<8>::exec.value = _mm256_andnot_si256(execution_mask.value, old_execution_mask.value);
    }
#endif

    template <typename T> static inline void if_(bool cond, T then_branch) {
      if (cond) then_branch();
    }

    // varying if_
    template <size_t N, typename T> static inline void if_(varying<bool, N> cond, T then_branch) {
      execution_mask_scope<N> scope(cond);
      if (execution_mask.any()) then_branch();
    }

    // boring uniform if_
    template <typename T, typename F> static inline void if_(bool cond, T then_branch, F else_branch) {
      if (cond) then_branch();
      else else_branch();
    }

    // varying if_
    template <size_t N, typename T, typename F> static inline void if_(varying<bool, N> cond, T then_branch, F else_branch) {
      execution_mask_scope<N> scope(cond);
      if (execution_mask.any()) then_branch();
      scope.flip();
      if (execution_mask.any()) else_branch();
    }

    template <size_t N = default_width> varying<int32_t,N> programIndex() {
      return make_varying<N,force = true>([&](int i) { return i });
    }

#ifdef __AVX2__
    template <> varying<int32_t,8> programIndex() { 
      return varying<int32_t,8>(7,6,5,4,3,2,1,0)
    }
#endif

    static const int programCount = default_width;

    // linear structures are 'uniform'ish, w/ constant offsets
    template <typename T, size_t N = default_width> struct linear {
      T base;
      explicit linear(T base) : base(base) {}
      linear(const linear & rhs) : base(rhs.base) {}
      linear(linear && rhs) : base(std::move(rhs.base)) {}
      template <typename S> auto operator[](S const * p) const -> varying<decltype(*(base + p)),N>;
      operator varying<int> () const {
        return programIndex + varying<int>(base);
      };

      linear operator + (int i) const {
        return linear(base + i);
      }
      linear operator - (int i) const {
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
      linear & operator = (linear & rhs) noexcept {
        base = rhs.base;
        return *this;
      }
      linear & operator += (int rhs) noexcept {
        base += rhs;
        return *this;
      }
      linear & operator -= (int rhs) noexcept {
        base -= rhs;
        return *this;
      }
      struct const_item_ref {
        const linear & l;
        int offset;
        explicit const_item_ref(linear & l, int offset) : l(l), offset(offset) {}
        operator auto () -> decltype(l.base + offset) { return l.base + offset; }
      };
      const_item_ref item(int i) const {
        return const_item_ref(*this, i);
      }
    };

    template <typename T, size_t N> template <typename S> auto linear<T,N>::operator[](S const * p) const noexcept -> varying<decltype(*(base + 0 + p)),N> {
      return make_varying<N>([&](int i) { return *(base + i + p); }
    }
#if __AVX2__
    template <> template <> varying<int32_t,4> linear<int32_t,4>::operator[](int32_t const * p) const noexcept {
      return varying<int32_t,4>(_mm_maskload_epi32(p + base, mask<4>::exec.value));
    }
    template <> template <> varying<int32_t,8> linear<int32_t,8>::operator[](int32_t const * p) const noexcept {
      return varying<int32_t,8>(_mm256_maskload_epi32(p + base, mask<8>::exec.value));
    }
#endif

    template <typename T, size_t N>
    static inline linear<T,N> operator + (int i, linear<T,N> j) noexcept {
      return linear<T,N>(i + j.base);
    }

    template <typename T, size_t N = default_width> using item_ref = typename varying<T,N>::item_ref;
    template <typename T, size_t N = default_width> using const_item_ref = typename varying<T,N>::const_item_ref;

    // hybrid soa array
    template <typename T, std::size_t L, std::size_t N = default_width>
    template soa {
      static_assert(N>0)
      varying<T,N> value[(L+N-1)/N];
      item_ref<T> operator[](int i) {
        return value[i / N].item(i % N);
      }
      varying<item_ref<T,N>> operator[](varying<int> i) {
        varying<item_ref<T,N>> result;
        execution_mask.each([&](int i) { result[i] = (*this)[i]; });
        return result;
      }
      // varying<item_ref<T>> operator[](linear<int> i) {
      const const_item_ref<T> operator[](int i) const {
        return value[i / N].item(i % N);
      }
      // const varying<const_item_ref<T>> operator[](varying<int> i);
      // const varying<const_item_ref<T>> operator[](linear<int> i);
      operator item_ptr<T,N>() { return value[0].item(0); }
      operator const_item_ptr<T,N>() const { return value[0].item(0); }
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

}
