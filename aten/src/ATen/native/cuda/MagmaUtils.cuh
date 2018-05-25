#include <magma.h>
#include <magma_types.h>


namespace at { 
namespace native { 

static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of %s (%lld) is too large to fit into a magma_int_t (%llu bytes)",
             varname, (long long)value, sizeof(magma_int_t));
  }
  return result;
}


}} // at::native