#include "spmd/avx2.h"
#include "spmd/generic.h"
#include "spmd/trivial.h"
#include <stdexcept>

namespace spmd {
  class unsupported_platform_error : public std::runtime_error {
    virtual const char * what() const throw() { return "unsupported platform"; }
  };

  // pick a vectorization scheme based on the host platform and run the code.
  template <typename F> inline auto dispatch(F fun) -> decltype(fun(::spmd::trivial::kernel())) {
#ifndef __arm__
    if (::spmd::avx2::kernel::available()) return fun(::spmd::avx2::kernel());
#endif
    return fun(::spmd::trivial::kernel());
  }
};
