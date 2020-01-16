// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Element.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_electric_scalar_1(
    const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar_part,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  *weyl_electric_scalar_part =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        for (size_t l = 0; l < SpatialDim; ++l) {
          get(*weyl_electric_scalar_part) += weyl_electric.get(i, j) *
                                             weyl_electric.get(k, l) *
                                             inverse_spatial_metric.get(i, k) *
                                             inverse_spatial_metric.get(j, l);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_electric_scalar_1(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  Scalar<DataType> weyl_electric_scalar_part{};
  weyl_electric_scalar_1<SpatialDim>(make_not_null(&weyl_electric_scalar_part),
                                     weyl_electric, inverse_spatial_metric);
  return weyl_electric_scalar_part;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_electric_scalar_impl(
    const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar_part,
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        weyl_electric_up_down,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        weyl_electric_up_down->get(j, k) +=
            weyl_electric.get(i, j) * inverse_spatial_metric.get(i, k);
      }
    }
  }
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      if (UNLIKELY(j == 0 and k == 0)) {
        get(*weyl_electric_scalar_part) =
            weyl_electric_up_down->get(j, k) * weyl_electric_up_down->get(j, k);
      } else {
        get(*weyl_electric_scalar_part) +=
            weyl_electric_up_down->get(j, k) * weyl_electric_up_down->get(j, k);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_electric_scalar_2(
    const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar_part,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  *weyl_electric_scalar_part =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  Variables<tmpl::list<Tags::Tempii<0, 3>>> temp{
      get<0, 0>(inverse_spatial_metric).size(), 0.0};
  auto& weyl_electric_up_down = get<Tags::Tempii<0, 3>>(temp);

  weyl_electric_scalar_impl(weyl_electric_scalar_part,
                            make_not_null(&weyl_electric_up_down),
                            weyl_electric, inverse_spatial_metric);
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_electric_scalar_2(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  Scalar<DataType> weyl_electric_scalar_part{};
  weyl_electric_scalar_2<SpatialDim>(make_not_null(&weyl_electric_scalar_part),
                                     weyl_electric, inverse_spatial_metric);
  return weyl_electric_scalar_part;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_electric_scalar_3(
    const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar_part,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  *weyl_electric_scalar_part =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  tnsr::ii<DataVector, 3> weyl_electric_up_down{
      get<0, 0>(inverse_spatial_metric).size(), 0.0};

  weyl_electric_scalar_impl(weyl_electric_scalar_part,
                            make_not_null(&weyl_electric_up_down),
                            weyl_electric, inverse_spatial_metric);
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_electric_scalar_3(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  Scalar<DataType> weyl_electric_scalar_part{};
  weyl_electric_scalar_3<SpatialDim>(make_not_null(&weyl_electric_scalar_part),
                                     weyl_electric, inverse_spatial_metric);
  return weyl_electric_scalar_part;
}

// clang-tidy: don't pass be non-const reference
void bench_1(benchmark::State& state) {  // NOLINT

  constexpr size_t num_points = 1000;
  const tnsr::ii<DataVector, 3, Frame::Inertial> weyl_electric{num_points, 1.0};
  const tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric{
      num_points, 0.4};

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(
        weyl_electric_scalar_1(weyl_electric, inverse_spatial_metric));
  }
}
BENCHMARK(bench_1);  // NOLINT

// clang-tidy: don't pass be non-const reference
void bench_2(benchmark::State& state) {  // NOLINT

  constexpr size_t num_points = 1000;
  const tnsr::ii<DataVector, 3, Frame::Inertial> weyl_electric{num_points, 1.0};
  const tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric{
      num_points, 0.4};

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(
        weyl_electric_scalar_2(weyl_electric, inverse_spatial_metric));
  }
}
BENCHMARK(bench_2);  // NOLINT

// clang-tidy: don't pass be non-const reference
void bench_3(benchmark::State& state) {  // NOLINT

  constexpr size_t num_points = 1000;
  const tnsr::ii<DataVector, 3, Frame::Inertial> weyl_electric{num_points, 1.0};
  const tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric{
      num_points, 0.4};

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(
        weyl_electric_scalar_3(weyl_electric, inverse_spatial_metric));
  }
}
BENCHMARK(bench_3);  // NOLINT
}  // namespace

BENCHMARK_MAIN();
