// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Psi0Real.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

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
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempI<0, 3, Frame>, ::Tags::TempI<1, 3, Frame>,
                       ::Tags::TempI<2, 3, Frame>, ::Tags::TempI<3, 3, Frame>,
                       ::Tags::Tempi<0, 3, Frame>, ::Tags::Tempii<0, 3, Frame>,
                       ::Tags::Tempii<1, 3, Frame>, ::Tags::TempIj<0, 3, Frame>,
                       ::Tags::TempII<0, 3, Frame>>>
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

  // Cross products construct the projected coordinate direction without
  // subtracting nearly parallel vectors near the x axis or the xy plane.
  auto& coordinate_direction = get<::Tags::TempI<1, 3, Frame>>(temp_buffer);
  get<0>(coordinate_direction) = 1.0;
  get<1>(coordinate_direction) = get<2>(coordinate_direction) = 0.0;
  auto& metric_determinant = get<::Tags::TempScalar<3>>(temp_buffer);
  determinant(make_not_null(&metric_determinant), spatial_metric);
  auto& second_polarization = get<::Tags::TempI<3, 3, Frame>>(temp_buffer);
  second_polarization = cross_product(
      r_hat, coordinate_direction, inverse_spatial_metric, metric_determinant);
  auto& magnitude_projected_x = get<::Tags::TempScalar<1>>(temp_buffer);
  magnitude(make_not_null(&magnitude_projected_x), second_polarization,
            spatial_metric);

  // Use the projected y direction where the projected x direction vanishes.
  get<0>(coordinate_direction) = 0.0;
  get<1>(coordinate_direction) = 1.0;
  auto& first_polarization = get<::Tags::TempI<2, 3, Frame>>(temp_buffer);
  first_polarization = cross_product(
      r_hat, coordinate_direction, inverse_spatial_metric, metric_determinant);
  auto& magnitude_projected_y = get<::Tags::TempScalar<2>>(temp_buffer);
  magnitude(make_not_null(&magnitude_projected_y), first_polarization,
            spatial_metric);
  for (size_t p = 0; p < get(magnitude_cartesian).size(); ++p) {
    if (get(magnitude_cartesian)[p] == 0.0) {
      continue;
    }
    const double minimum_magnitude = 100.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     sqrt(get<0, 0>(spatial_metric)[p]);
    for (size_t j = 0; j < 3; ++j) {
      if (get(magnitude_projected_x)[p] > minimum_magnitude) {
        second_polarization.get(j)[p] /= get(magnitude_projected_x)[p];
      } else {
        second_polarization.get(j)[p] =
            first_polarization.get(j)[p] / get(magnitude_projected_y)[p];
      }
    }
  }
  first_polarization = cross_product(
      second_polarization, r_hat, inverse_spatial_metric, metric_determinant);

  for (size_t p = 0; p < get(magnitude_cartesian).size(); ++p) {
    if (get(magnitude_cartesian)[p] == 0.0) {
      // The origin has no radial tetrad. Retain the Cartesian extension used
      // by Psi4 for compatible volume visualizations.
      const double metric_xx = get<0, 0>(spatial_metric)[p];
      const double metric_xy = get<0, 1>(spatial_metric)[p];
      const double magnitude_y =
          sqrt(get<1, 1>(spatial_metric)[p] - square(metric_xy) / metric_xx);
      get<0>(first_polarization)[p] = 1.0 / sqrt(metric_xx);
      get<1>(first_polarization)[p] = get<2>(first_polarization)[p] = 0.0;
      get<0>(second_polarization)[p] = -metric_xy / (metric_xx * magnitude_y);
      get<1>(second_polarization)[p] = 1.0 / magnitude_y;
      get<2>(second_polarization)[p] = 0.0;
    }
  }

  tenex::evaluate(
      psi_0_real_result,
      -0.5 * u8_minus(ti::i, ti::j) *
          (first_polarization(ti::I) * first_polarization(ti::J) -
           second_polarization(ti::I) * second_polarization(ti::J)));
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
