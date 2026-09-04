// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Psi0Real.hpp"

#include <cstddef>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {

template <typename Frame>
void psi_0_real(
    const gsl::not_null<Scalar<DataVector>*> psi_0_real_result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<0, 3, Frame>,
                 ::Tags::TempI<1, 3, Frame>, ::Tags::TempI<2, 3, Frame>,
                 ::Tags::TempI<3, 3, Frame>, ::Tags::Tempi<0, 3, Frame>,
                 ::Tags::Tempii<0, 3, Frame>, ::Tags::Tempii<1, 3, Frame>,
                 ::Tags::TempIj<0, 3, Frame>, ::Tags::TempII<0, 3, Frame>>>
      temp_buffer{get<0>(inertial_coords).size()};

  auto& magnitude_cartesian = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_cartesian), inertial_coords,
            spatial_metric);
  auto& r_hat = get<::Tags::TempI<0, 3, Frame>>(temp_buffer);
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < get(magnitude_cartesian).size(); ++i) {
      r_hat.get(j)[i] =
          magnitude_cartesian.get()[i] != 0.0
              ? inertial_coords.get(j)[i] / magnitude_cartesian.get()[i]
              : 0.0;
    }
  }

  auto& lower_r_hat = get<::Tags::Tempi<0, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::i>(make_not_null(&lower_r_hat),
                         r_hat(ti::J) * spatial_metric(ti::i, ti::j));
  auto& projection_tensor = get<::Tags::Tempii<0, 3, Frame>>(temp_buffer);
  transverse_projection_operator(make_not_null(&projection_tensor),
                                 spatial_metric, lower_r_hat);
  auto& inverse_projection_tensor =
      get<::Tags::TempII<0, 3, Frame>>(temp_buffer);
  transverse_projection_operator(make_not_null(&inverse_projection_tensor),
                                 inverse_spatial_metric, r_hat);
  auto& projection_up_lo = get<::Tags::TempIj<0, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::K, ti::i>(
      make_not_null(&projection_up_lo),
      projection_tensor(ti::i, ti::j) * inverse_spatial_metric(ti::K, ti::J));

  auto& u8_minus = get<::Tags::Tempii<1, 3, Frame>>(temp_buffer);
  gr::weyl_propagating(
      make_not_null(&u8_minus), spatial_ricci, extrinsic_curvature,
      inverse_spatial_metric, cov_deriv_extrinsic_curvature, r_hat,
      inverse_projection_tensor, projection_tensor, projection_up_lo, -1.0);

  // Gram-Schmidt x_hat, a unit vector orthogonal to r_hat.
  auto& x_coord = get<::Tags::TempI<1, 3, Frame>>(temp_buffer);
  x_coord.get(0) = 1.0;
  x_coord.get(1) = x_coord.get(2) = 0.0;
  auto& x_component = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&x_component), x_coord, r_hat, spatial_metric);
  auto& x_hat = get<::Tags::TempI<2, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::I>(make_not_null(&x_hat),
                         x_coord(ti::I) - x_component() * r_hat(ti::I));
  auto& magnitude_x = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_x), x_hat, spatial_metric);
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < get(magnitude_x).size(); ++i) {
      x_hat.get(j)[i] = magnitude_x.get()[i] != 0.0
                            ? x_hat.get(j)[i] / magnitude_x.get()[i]
                            : 0.0;
    }
  }

  // Gram-Schmidt y_hat, a unit vector orthogonal to r_hat and x_hat.
  auto& y_coord = get<::Tags::TempI<1, 3, Frame>>(temp_buffer);
  y_coord.get(1) = 1.0;
  y_coord.get(0) = y_coord.get(2) = 0.0;
  auto& y_component = get<::Tags::TempScalar<0>>(temp_buffer);
  dot_product(make_not_null(&y_component), y_coord, r_hat, spatial_metric);
  auto& y_hat = get<::Tags::TempI<3, 3, Frame>>(temp_buffer);
  tenex::evaluate<ti::I>(make_not_null(&y_hat),
                         y_coord(ti::I) - y_component() * r_hat(ti::I));
  dot_product(make_not_null(&y_component), y_coord, x_hat, spatial_metric);
  tenex::evaluate<ti::I>(make_not_null(&y_hat),
                         y_hat(ti::I) - y_component() * x_hat(ti::I));
  auto& magnitude_y = get<::Tags::TempScalar<0>>(temp_buffer);
  magnitude(make_not_null(&magnitude_y), y_hat, spatial_metric);
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < get(magnitude_y).size(); ++i) {
      y_hat.get(j)[i] = magnitude_y.get()[i] != 0.0
                            ? y_hat.get(j)[i] / magnitude_y.get()[i]
                            : 0.0;
    }
  }

  tenex::evaluate(psi_0_real_result, -0.5 * u8_minus(ti::i, ti::j) *
                                         (x_hat(ti::I) * x_hat(ti::J) -
                                          y_hat(ti::I) * y_hat(ti::J)));
}

template <typename Frame>
Scalar<DataVector> psi_0_real(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  auto result = make_with_value<Scalar<DataVector>>(
      get<0, 0>(inverse_spatial_metric),
      std::numeric_limits<double>::signaling_NaN());
  psi_0_real(make_not_null(&result), spatial_ricci, extrinsic_curvature,
             cov_deriv_extrinsic_curvature, spatial_metric,
             inverse_spatial_metric, inertial_coords);
  return result;
}
}  // namespace gr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template Scalar<DataVector> gr::psi_0_real(                             \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);        \
  template void gr::psi_0_real(                                           \
      const gsl::not_null<Scalar<DataVector>*> psi_0_real_result,         \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
