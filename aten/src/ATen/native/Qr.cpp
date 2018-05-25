#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/Qr.h"

#include "TH.h"  // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgeqrf_(
    int* m, int* n, double* a, int* lda, double* tau, 
    double* work, int* lwork, int* info);
extern "C" void sgeqrf_(
    int* m, int* n, float* a, int* lda, float* tau,
    float* work, int* lwork, int* info);
extern "C" void dorgqr_(
    int *m, int* n, int* k, double* a, int* lda, double* tau,
    double* work, int* lwork, int* info);
extern "C" void sorgqr_(
    int *m, int* n, int* k, float* a, int* lda, float* tau,
    float* work, int* lwork, int* info);
)
#endif

namespace at {
namespace native {

template<class scalar_t>
void lapackGeqrf(
    int m, int n, scalar_t** a, int lda, scalar_t** tau, int* info) {
  AT_ERROR("geqrf only takes float or double Tensors");
}

template<class scalar_t>
void lapackOrgqr(
    int m, int n, int k, scalar_t** a, int lda, scalar_t** tau, int* info) {
  AT_ERROR("orgqr only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGeqrf<float>(
    int m, int n, float* a, int lda, float* tau,
    float* work, int* lwork, int* info)) {
  sgeqrf_(&m, &n, a, &lda, &tau, & work, &lwork, info);
}

template<> void lapackGeqrf<double>(
    int m, int n, double* a, int lda, double* tau,
    double* work, int* lwork, int* info) {
  dgeqrf_(&m, &n, a, &lda, &tau, &work, &lwork, info);
}

template<> void lapackOrgqr<float>(
  int m, int n, int k, float* a, int lda, float* tau,
  float* work, int* lwork, int* info)) {
  sorgqr_(&m, &n, &k, a, &lda, &tau, &work, &lwork, info);
}

template<> void lapackOrgqr<double>(
  int m, int n, int k, double* a, int lda, double* tau,
  double* work, int* lwork, int* info)) {
  dorgqr_(&m, &n, &k, a, &lda, &tau, &work, &lwork, info);
}

#endif

template <typename scalar_t>
static void applyGeqrf(Tensor& tau, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  AT_ERROR("geqrf: LAPACK library not found in compilation");
#endif
  auto batch_size = batchCount(A);
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::max(m, n);
  auto lda = m;

  auto A_mat_stride = matrixStride(A);
  auto A_data = A.data<scalar_t>();
  auto tau_data = tau.data<scalart_t>();

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    int lwork;
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* tau_working_ptr = &tau_data[i * k];
    scalar_t wkopt = 0;

    lapackGeqrf<scalar_t>(m, n, A_working_ptr, lda, 
      tau_working_ptr, &wkopt, -1, &info);
    lwork = (int)wkopt;
    auto work = A.type().toScalarType(kInt).tensor(lwork);
    lapackGeqrf<scalar_t>(m, n, A_working_ptr, lda, 
      tau_working_ptr, work.data<scalar_t>(), lwork, &info);

    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

<template scalar_t>
static void applyOrgqr(Tensor& A, Tensor& tau, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  AT_ERROR("orgqr: LAPACK library not found in compilation");
#endif  
  auto batch_size = batchCount(A);
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::max(m, n);
  auto lda = m;

  auto A_mat_stride = matrixStride(A);
  auto A_data = A.data<scalar_t>();
  auto tau_data = tau.data<scalart_t>();

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    int lwork;
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* tau_working_ptr = &tau_data[i * k];
    scalar_t wkopt = 0;

    lapackOrgqr<scalart_t>(m, n, k, A_working_ptr, lda, 
      tau_working_ptr, &wkopt, -1, &info);
    lwork = (int)wkopt;
    auto work = A.type().toScalarType(kInt).tensor(lwork);
    lapackOrgqr<scalart_t>(m, n, k, A_working_ptr, lda, 
      tau_working_ptr, work.data<scalar_t>(), lwork, &info);

    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

std::tuple<Tensor,Tensor> _qr_helper_cpu(const Tensor& A) {
  std::tuple<Tensor,Tensor> geqrf_result_tuple = _geqrf_helper_cpu(A);
  Tensor R = std::get<0>(geqrf_result_tuple);
  Tensor tau = std::get<1>(geqrf_result_tuple);
  Tensor Q = _orgqr_helper_cpu(A, tau);
  // Tensor R.triu().reshape();
  // R.masked_fill_(mask, 0);
  return std::tuple<Tensor,Tensor>(Q, R);
}

std::tuple<Tensor,Tensor> _geqrf_helper_cpu(const Tensor& A) {
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


Tensor _orgqr_helper_cpu(const Tensor& A, const Tensor& tau) {
  std::vector<int64_t> infos(batchCount(A), 0);  

  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "orgqr", [&]{
    applyOrgqr<scalar_t>(A_working_copy, tau, infos);
  });
  checkErrors(infos);
  return A_working_copy;
}

// Supports arbitrary batch dimensions for self and A
// TODO Comment on the co-exisitance of ATen native and TH implementations

std::tuple<Tensor,Tensor> qr(const Tensor& A) {
  if (A.dim() <= 2) {
    return at::_qr_single(A);
  }
  return A.type()._qr_helper(A);
}

std::tuple<Tensor&,Tensor&> qr_out(Tensor& Q, Tensor& R, const Tensor& A) {
  if (A.dim() > 2) {
    AT_ERROR("torch.qr() with the `out` keyword deos not support batching. "
                "A.dim() (%lld) must be 2", (long long)A.dim());
  }
  return at::_qr_single_out(Q, R, A);
}
 
std::tuple<Tensor,Tensor> geqrf(const Tensor& A) {
  if (A.dim() <= 2) {
    return at::_geqrf_single(A);
  }
  return A.type()._geqrf_helper(A);
}

std::tuple<Tensor&,Tensor&> geqrf_out(
    Tensor& R, Tensor& tau, const Tensor& A) {
  if (A.dim() > 2) {
    AT_ERROR("torch.geqrf() with the `out` keyword does not support batching. "
                  "A.dim() (%lld) must be 2.", (long long)A.dim());
  }
  return at::_geqrf_single_out(R, tau, A);
}

Tensor orgqr(const Tensor& A, const Tensor& tau) {
  if (A.dim() <= 2) {
    return at::_orgqr_single(A, tau);
  }
  return self.type()._orgqr_helper();
}

Tensor& orgqr_out(Tensor&Q, const Tensor& A, const Tensor& tau) {
  if (A.dim() > 2) {
    AT_ERROR("torch.orgqr() with the `out` keyword does not support batching. "
                  "A.dim() (%ldd) must be 2.", (long long)A.dim());
  }
  return at::_orgqr_single_out(Q, A, tau);
} 

}}  // namespace at::native
