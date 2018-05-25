#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("geqrf: For batch %lld: Argument %lld has illegal value",
          (long long)i, -info);
    } 
  }
}

}}  // namespace at::native
