#include <cstddef> // std::size_t
#include <type_traits> // static_assert
namespace spmd {
  // portable "generic" varying implementation based on loops.
  // no real speedups expected, used for measuring SIMD improvements
  namespace generic {

    static const int items = 8;

    static inline bool available() {
      return true;
    }

    template <typename T> struct varying;
    struct mask {
      static const uint32_t on_mask = (1 << items) - 1;
      
      uint32_t value;
      mask() noexcept : value(on_mask) {}
      mask(bool b) noexcept : value(b ? on_mask : 0) {}
      mask(int value) : value(value) {}
      mask(const mask & m) noexcept : value(m.value) {}

      static mask on() noexcept { return mask(on_mask); }
      static mask off() noexcept { return mask(0); }
      bool any() const noexcept {
        return value != 0;
      }
      bool all() const noexcept {
        return value == on_mask;
      }
      mask operator & (const mask & that) const noexcept {
        return mask(value & that.value);
      }
      mask operator ^ (const mask & that) const noexcept {
        return mask(value ^ that.value);
      }
      mask operator | (const mask & that) const noexcept {
        return mask(value | that.value);
      }
      mask & operator &= (const mask & that) noexcept {
        value &= that.value;
        return *this;
      }
      mask & operator |= (const mask & that) noexcept {
        value &= that.value;
        return *this;
      }
      mask & operator ^= (const mask & that) noexcept {
        value &= that.value;
        return *this;
      }

      mask & operator &= (const varying<bool> & that) noexcept;

      mask operator ~ () const noexcept {
        return mask(value ^ on_mask);
      }

      template <typename F> void each(F f) {
        if (value == on_mask) {
          // dense
          for (int i=0;i<items;++i) f(i);
        } else {
          // sparse
          for (uint32_t t = value;t;) {
            auto i = __builtin_ffs(t);
            f(i);
            t &= ~ (1 << i);
          }
        }
      }
    };

    static thread_local mask execution_mask;

    template <typename T>
    struct varying<T> {
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
        execution_mask.each([&](int i) { value[i] = x; });
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

    template <typename T>
    struct varying<T&> { 
      std::array<T*,items> value;
      varying() {}
      varying(const T & x) {
        if (execution_mask.all()) {
          for (auto && e : value) *e = x;
        } else execution_mask.each([&](int i) { *(value[i]) = x; });
      }
      explicit varying(std::array<T*,items> value) noexcept : value(value) {}
      varying & operator =(const varying<T> & rhs) {
        execution_mask.each([&](int i) { *(value[i]) = rhs.item(i); });
      }
      varying & operator +=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { *(value[i]) += rhs.item(i); });
      }
      varying & operator -=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { *(value[i]) -= rhs.item(i); });
      }
      varying & operator *=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { *(value[i]) *= rhs.item(i); });
      }
      varying & operator /=(const varying<T> & rhs) {
        execution_mask.each([&](int i) { *(value[i]) /= rhs.item(i); });
      }
      T & item(int i) noexcept { return *(value[i]); }
      const T & item(int i) const noexcept { return *(value[i]); }
    };

    template <typename T>
    varying<T*> operator & (const varying<T&> & p) noexcept {
      return varying<T*>(r.value);
    }

    template <typename T>
    varying<T&> operator * (const varying<T*> & p) noexcept {
      return varying<T&>(p.value);
    }

    template <typename T>
    struct varying<bool> {
      uint32_t value;
      varying() noexcept {}
      varying(bool b) noexcept : value(b ? mask::on_mask : 0) {}
      explicit varying(uint32_t value) noexcept : value(value) {}
      varying(const varying & that) noexcept : value(that.value) {}
      varying masked() const noexcept {
        return varying(value & execution_mask.value);
      }
      varying operator != (const varying & rhs) const noexcept { return varying(value ^ rhs.value); }
      varying operator < (const varying & rhs) const noexcept { return varying(~value & rhs.value); }
      varying operator > (const varying & rhs) const noexcept { return varying(value & ~rhs.value); }
      varying operator == (const varying & rhs) const noexcept { return ! (*(this) != rhs); }
      varying operator <= (const varying & rhs) const noexcept { return (!(*this)) || rhs; }
      varying operator >= (const varying & rhs) const noexcept { return (*this) || (!rhs); }
      varying operator || (const varying & rhs) const noexcept { return varying(value | rhs.value); }
      varying operator && (const varying & rhs) const noexcept { return varying(value & rhs.value); }
      varying operator | (const varying & rhs) const noexcept { return varying(value | rhs.value); }
      varying operator & (const varying & rhs) const noexcept { return varying(value & rhs.value); }
      varying operator ! () const noexcept {
        return varying(value ^ mask::on_mask);
      }

      varying & operator = (const varying & rhs) noexcept {
        uint32_t m = execution_mask.value;
        value = (value & ~m) | (rhs & m);
        return *this;
      }

      varying & operator &= (const varying & rhs) noexcept {
        value &= (~execution_mask.value) | rhs;
        return *this;
      }

      varying & operator |= (const varying & rhs) noexcept {
        value |= (rhs & execution_mask.value)
        return *this;
      }

      template <typename F> static varying make(F f) {
        varying result;
        execution_mask.each([&](int i) { if (f(i)) value |= 1 << i; });
        return result;
      }

      // TODO: item_ref and item_ptr like in avx2 for item(i)
    }

    inline mask & mask::operator &= (const varying<bool> & that) noexcept {
      value &= that.value;
    }

    struct execution_mask_scope {
      mask old_execution_mask;
      execution_mask_scope() : old_execution_mask(execution_mask) {}
      execution_mask_scope(varying<bool> cond) : old_execution_mask(execution_mask) { execution_mask &= cond; }
      ~execution_mask_scope() { execution_mask = old_execution_mask; }
      void flip() const noexcept {
        execution_mask.value = ~execution_mask.value & old_execution_mask.value;
      }
    }

    template <typename T> void if_(bool cond, T then_branch) {
      if (cond) then_branch();
    }

    template <typename T> void if_(varying<bool> cond, T then_branch) {
      execution_mask_scope scope(cond);
      if (execution_mask.any()) then_branch();
    }

    template <typename T, typename F> void if_(bool cond, T then_branch, F else_branch) {
      if (cond) then_branch();
      else else_branch();
    }

    template <typename T, typename F> void if_(varying<bool> cond, T then_branch, F else_branch) {
      execution_mask_scope<N> scope(cond);
      if (execution_mask.any()) then_branch();
      scope.flip();
      if (execution_mask.any()) else_branch();
    }

    struct kernel {
      typedef generic::execution_mask execution_mask;
      typedef generic::execution_mask_scope execution_mask_scope;
      template <typename T> using varying = generic::varying<T>;
      template <typename T> using linear = generic::linear<T>;
      template <typename ... Ts> static void if_(Ts ... ts) {
        generic::if_<Ts...>(std::forward(ts)...);
      }
    };
  }
}
