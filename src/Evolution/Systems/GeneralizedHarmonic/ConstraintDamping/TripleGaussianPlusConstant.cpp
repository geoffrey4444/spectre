// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TripleGaussianPlusConstant.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::ConstraintDamping {

template <size_t VolumeDim, typename Fr>
TripleGaussianPlusConstant<VolumeDim, Fr>::TripleGaussianPlusConstant(
    const double constant, const std::array<double, 3>& amplitudes,
    const std::array<double, 3>& widths,
    const std::array<std::array<double, VolumeDim>, 3>& centers,
    const std::string& function_of_time_name) noexcept
    : constant_(constant),
      amplitudes_(amplitudes),
      inverse_widths_({{1.0 / widths[0], 1.0 / widths[1], 1.0 / widths[2]}}),
      centers_(centers) {
  DampingFunction<VolumeDim, Fr>::function_of_time_for_scaling_name =
      function_of_time_name;
}

template <size_t VolumeDim, typename Fr>
template <typename T>
tnsr::I<T, VolumeDim, Fr>
TripleGaussianPlusConstant<VolumeDim, Fr>::centered_coordinates(
    const tnsr::I<T, VolumeDim, Fr>& x,
    const size_t which_gaussian) const noexcept {
  tnsr::I<T, VolumeDim, Fr> centered_coords = x;
  for (size_t i = 0; i < VolumeDim; ++i) {
    centered_coords.get(i) -= gsl::at(gsl::at(centers_, which_gaussian), i);
  }
  return centered_coords;
}

template <size_t VolumeDim, typename Fr>
template <typename T>
Scalar<T> TripleGaussianPlusConstant<VolumeDim, Fr>::apply_call_operator(
    const tnsr::I<T, VolumeDim, Fr>& centered_coords_0,
    const tnsr::I<T, VolumeDim, Fr>& centered_coords_1,
    const tnsr::I<T, VolumeDim, Fr>& centered_coords_2) const noexcept {
  Scalar<T> result = dot_product(centered_coords_0, centered_coords_0);
  get(result) =
      constant_ +
      amplitudes_[0] *
          exp(-get(result) *
              square(inverse_widths_[0] *
                     DampingFunction<VolumeDim, Fr>::time_dependent_scale));
  get(result) +=
      amplitudes_[1] *
          exp(-get(dot_product(centered_coords_1, centered_coords_1)) *
              square(inverse_widths_[1] *
                     DampingFunction<VolumeDim, Fr>::time_dependent_scale)) +
      amplitudes_[2] *
          exp(-get(dot_product(centered_coords_2, centered_coords_2)) *
              square(inverse_widths_[2] *
                     DampingFunction<VolumeDim, Fr>::time_dependent_scale));
  return result;
}

template <size_t VolumeDim, typename Fr>
Scalar<double> TripleGaussianPlusConstant<VolumeDim, Fr>::operator()(
    const tnsr::I<double, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(centered_coordinates(x, 0),
                             centered_coordinates(x, 1),
                             centered_coordinates(x, 2));
}
template <size_t VolumeDim, typename Fr>
Scalar<DataVector> TripleGaussianPlusConstant<VolumeDim, Fr>::operator()(
    const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept {
  return apply_call_operator(centered_coordinates(x, 0),
                             centered_coordinates(x, 1),
                             centered_coordinates(x, 2));
}

template <size_t VolumeDim, typename Fr>
void TripleGaussianPlusConstant<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | constant_;
  p | amplitudes_;
  p | inverse_widths_;
  p | centers_;
  p | DampingFunction<VolumeDim, Fr>::function_of_time_for_scaling_name;
  p | DampingFunction<VolumeDim, Fr>::time_dependent_scale;
}

template <size_t VolumeDim, typename Fr>
auto TripleGaussianPlusConstant<VolumeDim, Fr>::get_clone() const noexcept
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<TripleGaussianPlusConstant<VolumeDim, Fr>>(*this);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                   \
  template GeneralizedHarmonic::ConstraintDamping::TripleGaussianPlusConstant< \
      DIM(data), FRAME(data)>::                                                \
      TripleGaussianPlusConstant(                                              \
          const double constant, const std::array<double, 3>& amplitudes,      \
          const std::array<double, 3>& widths,                                 \
          const std::array<std::array<double, DIM(data)>, 3>& centers,         \
          const std::string& function_of_time_name) noexcept;                  \
  template void GeneralizedHarmonic::ConstraintDamping::                       \
      TripleGaussianPlusConstant<DIM(data), FRAME(data)>::pup(PUP::er& p);     \
  template auto GeneralizedHarmonic::ConstraintDamping::                       \
      TripleGaussianPlusConstant<DIM(data), FRAME(data)>::get_clone()          \
          const noexcept                                                       \
              ->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                            \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::ConstraintDamping:: \
      TripleGaussianPlusConstant<DIM(data), FRAME(data)>::operator()(   \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x)        \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

/// \endcond
