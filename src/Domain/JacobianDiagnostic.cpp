// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/JacobianDiagnostic.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
template <size_t Dim, typename Fr>
void jacobian_diagnostic(
    const gsl::not_null<Scalar<DataVector>*> jacobian_diagnostic,
    const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&
        analytic_jacobian,
    const TensorMetafunctions::prepend_spatial_index<
        tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
        typename Frame::ElementLogical>& numeric_jacobian_transpose) {
  auto jacobian_difference_sum =
      make_with_value<tnsr::i<DataVector, Dim, typename Frame::ElementLogical>>(
          get<0, 0>(analytic_jacobian), 0.0);
  auto numerical_difference_sum =
      make_with_value<tnsr::i<DataVector, Dim, typename Frame::ElementLogical>>(
          get<0, 0>(analytic_jacobian), 0.0);
  constexpr double eps_to_avoid_div_by_zero = 1.e-20;

  // i_hat = logical frame index
  // i = mapped frame index
  for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
    for (size_t i = 0; i < Dim; ++i) {
      jacobian_difference_sum.get(i_hat) -= analytic_jacobian.get(i, i_hat);
      numerical_difference_sum.get(i_hat) +=
          numeric_jacobian_transpose.get(i_hat, i);
    }
  }
  for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {
    // If the numeric and analytic components are both sufficiently small,
    // treat them as equal.
    constexpr double eps_for_ratio = 1.e-10;
    if (max(abs(jacobian_difference_sum.get(i_hat))) < eps_for_ratio and
        max(abs(numerical_difference_sum.get(i_hat))) < eps_for_ratio) {
      jacobian_difference_sum.get(i_hat) = 0.0;
      continue;
    }
    jacobian_difference_sum.get(i_hat) /=
        (numerical_difference_sum.get(i_hat) + eps_to_avoid_div_by_zero);
    jacobian_difference_sum.get(i_hat) += 1.0;
  }

  get(*jacobian_diagnostic) = square(get<0>(jacobian_difference_sum));
  for (size_t i_hat = 1; i_hat < Dim; ++i_hat) {
    jacobian_diagnostic->get() += square(jacobian_difference_sum.get(i_hat));
  }
  get(*jacobian_diagnostic) = sqrt(get(*jacobian_diagnostic));
}
}  // namespace domain

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void domain::jacobian_diagnostic(                                \
      const gsl::not_null<Scalar<DataVector>*> jacobian_diagnostic,         \
      const Jacobian<DataVector, DIM(data), Frame::ElementLogical,          \
                     FRAME(data)>& analytic_jacobian,                       \
      const TensorMetafunctions::prepend_spatial_index<                     \
          tnsr::I<DataVector, DIM(data), FRAME(data)>, DIM(data), UpLo::Lo, \
          typename Frame::ElementLogical>& numeric_jacobian_transpose);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
