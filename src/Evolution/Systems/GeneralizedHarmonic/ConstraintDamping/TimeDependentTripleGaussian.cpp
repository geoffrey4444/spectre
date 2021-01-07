// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TimeDependentTripleGaussian.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::ConstraintDamping {

template <size_t VolumeDim, typename Fr>
TimeDependentTripleGaussian<VolumeDim, Fr>::TimeDependentTripleGaussian(
    const double constant, const double amplitude_1, const double width_1,
    const std::array<double, VolumeDim>& center_1, const double amplitude_2,
    const double width_2, const std::array<double, VolumeDim>& center_2,
    const double amplitude_3, const double width_3,
    const std::array<double, VolumeDim>& center_3,
    std::string function_of_time_for_scaling) noexcept
    : constant_(constant),
      amplitude_1_(amplitude_1),
      inverse_width_1_(1.0 / width_1),
      center_1_(center_1),
      amplitude_2_(amplitude_2),
      inverse_width_2_(1.0 / width_2),
      center_2_(center_2),
      amplitude_3_(amplitude_3),
      inverse_width_3_(1.0 / width_3),
      center_3_(center_3),
      function_of_time_for_scaling_(std::move(function_of_time_for_scaling)) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
void TimeDependentTripleGaussian<VolumeDim, Fr>::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x,
    const tnsr::I<T, VolumeDim, Fr>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.at(function_of_time_for_scaling_)
                 ->func(time)[0]
                 .size() == 1,
         "FunctionOfTimeForScaling in TimeDependentTripleGaussian must be a "
         "scalar FunctionOfTime, not "
             << functions_of_time.at(function_of_time_for_scaling_)
                    ->func(time)[0]
                    .size());
  const double function_of_time_value =
      functions_of_time.at(function_of_time_for_scaling_)->func(time)[0][0];

  // Start by setting the result to the constant
  get(*value_at_x) = constant_;

  // Loop over the three Gaussians, adding each to the result
  auto centered_coords = make_with_value<tnsr::I<T, VolumeDim, Fr>>(
      get<0>(x), std::numeric_limits<double>::signaling_NaN());

  const auto add_gauss_to_value_at_x =
      [&value_at_x, &centered_coords, &x, &function_of_time_value](
          const double amplitude, const double inverse_width,
          const std::array<double, VolumeDim>& center) {
        for (size_t i = 0; i < VolumeDim; ++i) {
          centered_coords.get(i) = x.get(i) - gsl::at(center, i);
        }
        get(*value_at_x) +=
            amplitude *
            exp(-get(dot_product(centered_coords, centered_coords)) *
                square(inverse_width * function_of_time_value));
      };
  add_gauss_to_value_at_x(amplitude_1_, inverse_width_1_, center_1_);
  add_gauss_to_value_at_x(amplitude_2_, inverse_width_2_, center_2_);
  add_gauss_to_value_at_x(amplitude_3_, inverse_width_3_, center_3_);
}  // namespace GeneralizedHarmonic::ConstraintDamping

template <size_t VolumeDim, typename Fr>
void TimeDependentTripleGaussian<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<double>*> value_at_x,
    const tnsr::I<double, VolumeDim, Fr>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  apply_call_operator(value_at_x, x, time, functions_of_time);
}
template <size_t VolumeDim, typename Fr>
void TimeDependentTripleGaussian<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<DataVector>*> value_at_x,
    const tnsr::I<DataVector, VolumeDim, Fr>& x, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  destructive_resize_components(value_at_x, get<0>(x).size());
  apply_call_operator(value_at_x, x, time, functions_of_time);
}

template <size_t VolumeDim, typename Fr>
void TimeDependentTripleGaussian<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | constant_;
  p | amplitude_1_;
  p | inverse_width_1_;
  p | center_1_;
  p | amplitude_2_;
  p | inverse_width_2_;
  p | center_2_;
  p | amplitude_3_;
  p | inverse_width_3_;
  p | center_3_;
  p | function_of_time_for_scaling_;
}

template <size_t VolumeDim, typename Fr>
auto TimeDependentTripleGaussian<VolumeDim, Fr>::get_clone() const noexcept
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<TimeDependentTripleGaussian<VolumeDim, Fr>>(*this);
}
}  // namespace GeneralizedHarmonic::ConstraintDamping

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                \
  template GeneralizedHarmonic::ConstraintDamping::                         \
      TimeDependentTripleGaussian<DIM(data), FRAME(data)>::                 \
          TimeDependentTripleGaussian(                                      \
              const double constant, const double amplitude_1,              \
              const double width_1,                                         \
              const std::array<double, DIM(data)>& center_1,                \
              const double amplitude_2, const double width_2,               \
              const std::array<double, DIM(data)>& center_2,                \
              const double amplitude_3, const double width_3,               \
              const std::array<double, DIM(data)>& center_3,                \
              std::string function_of_time_for_scaling) noexcept;           \
  template void GeneralizedHarmonic::ConstraintDamping::                    \
      TimeDependentTripleGaussian<DIM(data), FRAME(data)>::pup(PUP::er& p); \
  template auto GeneralizedHarmonic::ConstraintDamping::                    \
      TimeDependentTripleGaussian<DIM(data), FRAME(data)>::get_clone()      \
          const noexcept                                                    \
              ->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::ConstraintDamping::                   \
      TimeDependentTripleGaussian<DIM(data), FRAME(data)>::operator()(     \
          const gsl::not_null<Scalar<DTYPE(data)>*> value_at_x,            \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x,           \
          const double /*time*/,                                           \
          const std::unordered_map<                                        \
              std::string,                                                 \
              std::unique_ptr<domain::FunctionsOfTime::                    \
                                  FunctionOfTime>>& /*functions_of_time*/) \
          const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE

/// \endcond
