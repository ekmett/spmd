#pragma once
#include <immintrin.h>
#include <utility>
#include "cpu.h"

// spmd on simd as a library
// currently assuming avx2
// eventually this namespace will become a struct so we can instantiate this several ways and use template magic to compile multiple times and do dispatch

namespace spmd {
  // detect avx2
  static inline bool available() {
    auto i = cpu::system_isa();
    return i >= cpu::isa::avx2 && i <= cpu::isa::max_intel;
  }

  struct vbool;

  // execution mask
  struct mask {
    __m256i value;

    // default to initialized.
    mask() noexcept : value(_mm256_cmpeq_epi32(_mm256_setzero_si256(),_mm256_setzero_si256())) {}
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

    mask & operator &= (const mask & rhs) noexcept {
      value = _mm256_and_si256(value, rhs.value);
      return *this;
    }

    mask & operator |= (const mask & rhs) noexcept {
      value = _mm256_or_si256(value, rhs.value);
      return *this;
    }

    mask & operator &= (const vbool & rhs) noexcept;
  };

  extern thread_local mask execution_mask; // shared execution mask

  struct vbool {
    __m256i value;

    // initialize a varying bool from a uniform bool
    vbool(bool b) noexcept : value(_mm256_set1_epi32(b ? ~0 : 0)) {}
    vbool(__m256i value) noexcept : value(value) {}
    vbool(const vbool & that) noexcept : value(that.value) {} // copy constructor
    // vbool(vbool && that) noexcept : value(std::forward(that.value)) {} // move constructor

    // explicitly initialize a varying bool from vector type without masking

    // vectorized comparisons

    vbool operator != (const vbool & rhs) const noexcept {
      return _mm256_xor_si256(value,rhs.value);
    }

    // vector boolean comparison
    vbool operator == (const vbool & rhs) const noexcept {
      return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), _mm256_xor_si256(value,rhs.value));
    }

    // F < F = F
    // F < T = T 
    // T < F = F
    // T < T = F
    // ~a & b
    vbool operator < (const vbool & rhs) const noexcept {
       return _mm256_andnot_si256(value,rhs.value);
    }

    // F <= F = T
    // F <= T = T
    // T <= F = F
    // T <= T = T
    vbool operator <= (const vbool & rhs) const noexcept {
      // negate first 
      return (!(*this)) || rhs;
    }

    vbool operator >= (const vbool & rhs) const noexcept {
      return (*this) || (!rhs);
    }

    vbool operator > (const vbool & rhs) const noexcept {
      return _mm256_andnot_si256(rhs.value,value);
    }

    vbool operator||(const vbool& that) const noexcept {
      return _mm256_or_si256(value, that.value);
    }

    vbool operator&&(const vbool& that) const noexcept {
      return _mm256_and_si256(value, that.value);
    }

    vbool operator|(const vbool& that) const noexcept {
      return _mm256_or_si256(value, that.value);
    }

    vbool operator&(const vbool& that) const noexcept {
      return _mm256_and_si256(value, that.value);
    }

    vbool operator!() const noexcept {
      return _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), value);
    }

    // masked assignment operator
    vbool & operator = (const vbool & rhs) noexcept {
      value = _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(rhs.value), _mm256_castsi256_ps(execution_mask.value))
      );
      return *this;
    }

    vbool & operator|=(const vbool& that) noexcept {
      value = _mm256_castps_si256(
         _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_or_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
      );
      return *this;
    }

    vbool & operator&=(const vbool& that) noexcept {
      value = _mm256_castps_si256(
         _mm256_blendv_ps(_mm256_castsi256_ps(value), _mm256_castsi256_ps(_mm256_and_si256(value, that.value)), _mm256_castsi256_ps(execution_mask.value))
      );
      return *this;
    }

/*
    // explicitly 0 inactive items
    vbool masked() const noexcept {
      return vbool {
        _mm256_castps_si256(_mm256_blendv_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(value), _mm256_castsi256_ps(execution_mask.value)))
      };
    }

    // and results for all of the active threads in this "warp"
    bool any() const noexcept {
      return _mm256_movemask_ps(_mm256_blendv_ps(_mm256_setzero_ps(), _mm256_castsi256_ps(value), _mm256_castsi256_ps(execution_mask.value))) != 0;
    }

    // or results for all of the active threads in this "warp"
    bool all() const noexcept {
      __m256 z = _mm256_setzero_ps();
      return _mm256_movemask_ps(_mm256_blendv_ps(_mm256_cmpeq_epi32(z,z), _mm256_castsi256_ps(value), _mm256_castsi256_ps(execution_mask.value))) == 0xff;
    }
*/

  };

  mask & mask::operator &= (const vbool & rhs) noexcept {
    value = _mm256_and_si256(value, rhs.value);
    return *this;
  }

  struct vdouble {
    __m256 value[2];
  };

  struct vfloat {
    __m256 value;
  };

  struct vlonglong {
    __m256i value[2];
  };

  struct vint {
    __m256i value;
    // explicitly initialize a varying int from vector type without masking
    explicit vint(const __m256i & value) noexcept : value(value) {}
  };

/*
  -- we need to static_assert<> to pick one or the other implementation
  template <typename T> struct vpointer {
    __m256 value[2]; // on 64 bit platforms
    __m256 value; // on 32 bit platforms
  };
*/

  // linear int, useful for indices.
  struct lint {
    int base; 
    explicit lint(int base) noexcept : base(base) {}

    template <typename T> vint operator[](int const * p) const noexcept {
      // perform a masked load. This prevents reading past end of page, or doing other bad things out of bounds
      return vint { _mm256_maskload_epi32(p + base, execution_mask.value) };
    }

    // downgrade to a varying int
    vint operator()() const noexcept { 
      return vint { _mm256_add_ps(_mm256_set_ps(7,6,5,4,3,2,1,0), _mm256_set1_epi32(base)) };
    };
  };

  // raii, copy existing execution mask, restore on exit
  struct execution_mask_scope {
    mask old_execution_mask;
    execution_mask_scope() : old_execution_mask(execution_mask) {}
    execution_mask_scope(vbool cond) : old_execution_mask(execution_mask) { execution_mask &= cond; }
    ~execution_mask_scope() { execution_mask = old_execution_mask; }

    // if we masked off some threads since we started, flip the mask to just the others.
    void flip() const noexcept {
      execution_mask.value = _mm256_andnot_si256(execution_mask.value, old_execution_mask.value);
    }
  };

  // boring uniform if_
  template <typename T> void if_(bool cond, T then_branch) {
    if (cond) then_branch();
  }

  // varying if_
  template <typename T> void if_(vbool cond, T then_branch) { 
    execution_mask_scope scope(cond);
    if (execution_mask.any()) then_branch();
  }

  // boring uniform if_
  template <typename T, typename F> void if_(bool cond, T then_branch, F else_branch) {
    if (cond) then_branch();
    else else_branch();
  }

  // varying if_
  template <typename T, typename F> void if_(vbool cond, T then_branch, F else_branch) {
    execution_mask_scope scope(cond);
    if (execution_mask.any()) then_branch();
    scope.flip();
    if (execution_mask.any()) else_branch();
  }
}

#define SPMD_IF_THEN(x,y) ::spmd::if_((x),[&] { y })
#define SPMD_IF_THEN_ELSE(x,y,z) ::spmd::if_((x),[&] { y }, [&] { z })
