#include "spmd/avx2.h"
#include "spmd/generic.h"

namespace spmd {
  class unsupported_platform_error : public runtime_error {
    virtual const char * what() const throw() { 
      return "unsupported platform";
    }
  };

  template <typename F, typename T, typename ... Ts> auto dispatch_platform(F fun) {
    return T::available() ? f<T>() : kernel<F, ... Ts>(f)
  };

  [[noreturn]] template <typename F> auto dispatch_platform(F fun) {
    throw unsupported_platform_error();
  }

  // host os specific dispatch pattern
  template <typename F> auto kernel(F fun) {
    return dispatch_platform<F, 
#ifndef __arm__
       avx2,
#endif
       generic<8> // try for comparable width
    >(fun);
  }
};
