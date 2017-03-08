namespace spmd {
  // disable spmd-on-simd and execute directly
  struct trivial {
    static inline bool available() { 
      return true;
    }

    template <typename T> using varying = T;
    struct mask { 
      bool value;
      mask(bool value = true) noexcept : value(value) {}
      mask(const mask & rhs) noexcept : value(rhs.value) {}
      static mask on() noexcept { return true; }
      static mask off() noexcept { return false; }
      bool any() const noexcept { return value; }
      bool all() const noexcept { return value; }
      mask & operator & (const mask & rhs) noexcept { return value & rhs.value; return *this; }
      mask & operator | (const mask & rhs) noexcept { value &= rhs.value; return *this;}
      mask & operator ^ (const mask & rhs) noexcept { value |= rhs.value; return *this;}
      mask & operator = (const mask & rhs) noexcept { value = rhs.value; return *this; }
      mask & operator &= (const mask & rhs) noexcept { value &= rhs.value; return *this;}
      mask & operator &= (bool rhs) noexcept { value &= rhs; return *this;}
      mask & operator |= (const mask & rhs) noexcept { value |= rhs.value; return *this;}
    }

    static thread_local mask execution_mask; // potentially useful for breaks, loops, etc.

    struct execution_mask_scope {
      mask old_execution_mask;
      execution_mask_scope() : old_execution_mask(execution_mask) {}
      execution_mask_scope(bool b) : old_execution_mask(execution_mask) { execution_mask.value &= b; }
      ~execution_mask_scope() { execution_mask = old_execution_mask; }
      void flip() {
        execution_mask.value = !execution_mask.value && old_execution_mask.value;
      }
    };

    template <typename T> void if_(bool cond, T then_branch) {
      if (cond) then_branch();
    }

    template <typename T, typename F> void if_(bool cond, T then_branch, F else_branch) {
      if (cond) then_branch();
      else else_branch();
    }
  };
}