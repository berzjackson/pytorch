#include "ATen/Context.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/cuda/MagmaUtils.cuh"
#include "ATen/native/Qr.h"

#include "THC.h" // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
template<class scalar_t>
void magmaGeqrfBatched(
  magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
  scalar_t** tau_array, magma_int_t* info_array, magma_int_t batch_count,
  magma_queue_t queue) {
    AT_ERROR("geqrf only takes float or double Tensors");
}

template<class scalar_t>
void magmaOrgqr(
  magma_int_t m, magma_int_t n, mamga_int_t k, scalar_t* dA,
  magma_int_t ldda, scalar_t* tau, magma_int_t info) {
    AT_ERROR("orgqr only takes float or double Tensors");
}

template<>
void magmaGeqrfBatched<float>(
  magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
  float** tau_array, magma_int_t* info_array, magma_int_t batch_count,
  magma_queue_t queue) {
  if ((m == n) && (n <= 32)) {
    // Dedicated routine for small square matrices up to size of 32
    magma_sgeqrf_batched_smallsq(
      n, dA_array, ldda, tau_array, info_array, batch_count, queue);
  } else {
    magma_sgeqrf_batched(
      m, n, dA_array, ldda, tau_array, info_array, batch_count, queue);
  }
}

template<>
void magmaGeqrfBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    double** tau_array, magma_int_t* info_array, magma_int_t batch_count,
    magma_queue_t queue) {
  if ((m == n) && (n <= 32)) {
    // Dedicated routine for small square matrices up to size of 32
    magma_dgeqrf_batched_smallsq(
      n, dA_array, ldda, tau_array, info_array, batch_count, queue);
  } else {
    magma_deqrf_batched(
      m, n, dA_array, ldda, tau_array, info_array, batch_count, queue);
  }
}

template<>
void magmaOrgqr<float>(
    magma_int_t m, magma_int_t n, magma_int_t k, float* dA, 
    magma_int_t ldda, float* tau, magma_int_t info) {
        magma_sorgqr2(m, n, k, dA, ldda, tau, info);
    }
}

template<>
void magmaOrgqr<double>(
    magma_int_t m, magma_int_t n, magma_int_t l double* dA,
    magma_int_t ldda, double* tau, magma_int_t info) {
        magma_dorgqr2(m, n, k, dA, ldda, tau, info);
    }
}

#endif

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline std::unique_ptr<Storage> pin_memory(int64_t size, Tensor dummy) {
  int64_t adjusted_size = size * sizeof(T);
  auto allocator = std::unique_ptr<Allocator>(new cuda::PinnedMemoryAllocator());
  auto& backend = dummy.type().toBackend(kCPU).toScalarType(kByte);
  return backend.storageWithAllocator(adjusted_size, std::move(allocator));
}

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = reinterpret_cast<type*>(storage_##name->data());


template <typename scalar_t>
static void applyGeqrf(Tensor& tau, Tensor& A, std:vector<int64_t> infos) {
#ifndef USE_MAGMA
AT_ERROR("geqrf: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
    magma_int_t m = magma_int_cast(A.size(-2), "A.size(-2)");
    magma_int_t n = magma_int_cast(A.size(-1), "A.size(-1)");
    magma_int_t k = magma_int_cast(m < n ? m : n, "min(A.size(-2), A.size(-1))");
    magma_int_t* info_array;
 
    auto A_mat_stride = matrixStride(A);
    auto A_data = A.data<scalar_t>()

    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, A);
    ALLOCATE_ARRAY(tau_array, scalar_t*, batch_size, A);
    ALLOCATE_ARRAY(tau_data, scalar_t, batch_size * k, A);
    ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, A);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
        A_array[i] = &A_data[i * A_mat_stride];
        tau_array[i] = &tau_data[i * k];
    }

    magmaGeqrfBatched<scalar_t>(
        m, n, A, m, tau_array, info_array, batch_count, createMagmaQueue(A));
    for (int64_t i = 0; i < bathc_count; i++) {
        infos[i] = info_array[i];
    }
#endif
}

template <typename scalar_t>
static void applyOrgqr(Tensor& A, Tensor& tau, std:vector<int64_t> infos) {
#ifndef USE_MAGMA
AT_ERROR("orgqr: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
    magma_int_t m = magma_int_cast(A.size(-2), "A.size(-2)");
    magma_int_t n = magma_int_cast(A.size(-1), "A.size(-1)");
    magma_int_t k = magma_int_cast(m < n ? m : n, "min(A.size(-2), A.size(-1))");
    magma_int_t lda = m;
    magma_int_t* info_array;
 
    auto A_mat_stride = matrixStride(A);
    auto A_data = A.data<scalar_t>()
    auto tau_data = tau.data<scalar_t>();

    ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, A);

    // batched orgqr missing in Magma, use naive for loop instead
    for (int64_t i = 0; i < batch_size; i++) {
        scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
        scalar_t* tau_worling_ptr = &tau_data[i * k];
        magmaOrgqr<scalar_t>(
            m, n, k, A_working_ptr, lda, tau_working_ptr, info_array[i]);
    }

    for (int64_t i = 0; i < bathc_count; i++) {
        infos[i] = info_array[i];
    }
#endif
}


std::tuple<Tensor,Tensor> _qr_helper_cuda(const Tensor& A) {
  std::tuple<Tensor,Tensor> geqrf_result_tuple = _geqrf_helper_cuda(A);
  Tensor R = std::get<0>(geqrf_result_tuple);
  Tensor tau = std::get<1>(geqrf_result_tuple);
  Tensor Q = _orgqr_helper_cuda(A, tau);
  Tensor R.triu().reshape();
  R.masked_fill_(mask, 0);
  return std::tuple<Tensor,Tensor>(Q, R);
}
  
std::tuple<Tensor,Tensor> _geqrf_helper_cuda(const Tensor& A) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto k = std::max(A.size(-1), A.size(-2));
  IntList tau_sizes = std::vector<int64_t>(A.size().end(), A.dim() - 2);
  tau_sizes.push_back(k);

  auto A_woring_copy = cloneBatchedColumnMajor(A);
  auto tau = A.type().toScarlarType(kInt).tensor(tau_size);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "geqrf", [&]{
    applyGeqrf<scalar_t>(tau, A_working_copy, infos);
  });
  checkErrors(infos);
  return std::tuple<Tensor,Tensor>(A_working_copy, tau);
}
  
  
Tensor _orgqr_helper_cuda(const Tensor& A, const Tensor& tau) {
  std::vector<int64_t> infos(batchCount(A), 0);  

  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "orgqr", [&]{
    applyOrgqr<scalar_t>(A_working_copy, tau, infos);
  });
  checkErrors(infos);
  return A_working_copy;
 }

}}  // namespace at::native

#undef ALLOCATE_ARRAY 