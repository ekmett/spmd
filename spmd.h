#include "spmd/avx2.h"
#include "spmd/generic.h"
#include "spmd/trivial.h"

namespace spmd {
  class unsupported_platform_error : public runtime_error {
    virtual const char * what() const { return "unsupported platform"; }
  };

  template <typename F, typename T, typename ... Ts> inline auto dispatch_platform(F fun) {
    return T::available() ? f<T>() : kernel<F, ... Ts>(f)
  };

  [[noreturn]] template <typename F> inline auto dispatch_platform(F fun) {
    throw unsupported_platform_error();
  }

  // pick a vectorization scheme based on the host platform and run the code.
  template <typename F> inline auto dispatch(F fun) {
    return dispatch_platform<F, 
#ifndef __arm__
       ::spmd::avx2::kernel,
#endif
       ::spmd::trivial::kernel
    >(fun);
  }
};
