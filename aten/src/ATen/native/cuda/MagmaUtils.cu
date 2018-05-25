#include <magma.h>
#include <magma_types.h>


namespace at {
namespace native {

static magma_queue_t createMagmaQueue(const Tensor& tensor) {
  auto& context = tensor.type().get_context();
  magma_queue_t magma_queue;
  magma_queue_create_from_cuda(
      tensor.get_device(),
      context.getCurrentCUDAStream(),
      THCState_getCurrentBlasHandle(context.getTHCState()),
      THCState_getCurrentSparseHandle(context.getTHCState()),
      &magma_queue);
  return magma_queue;
}



}} // at::native