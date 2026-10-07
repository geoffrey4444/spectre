// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/RadialProjectionLocalSource.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.tpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
Variables<typename System<3>::variables_tag::tags_list>
radial_projection_local_source(
    const Variables<typename System<3>::variables_tag::tags_list>& state,
    const Mesh<3>& mesh, const double time,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const gauges::GaugeCondition& gauge) {
  if (dynamic_cast<const gauges::DampedHarmonic*>(&gauge) == nullptr) {
    ERROR(
        "Experimental radial local-source projection currently requires the "
        "non-rollon DampedHarmonic gauge.");
  }
  using Metric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = gh::Tags::Pi<DataVector, 3>;
  using Phi = gh::Tags::Phi<DataVector, 3>;
  using TD = System<3>::compute_volume_time_derivative_terms;
  const size_t points = state.number_of_grid_points();
  if (mesh.number_of_grid_points() != points) {
    ERROR("Local-source mesh and state sizes disagree.");
  }
  Variables<typename System<3>::variables_tag::tags_list> source(points, 0.0);
  Variables<typename TD::temporary_tags> temporary(points);
  tnsr::iaa<DataVector, 3> dmetric(points, 0.0), dpi(points, 0.0);
  tnsr::ijaa<DataVector, 3> dphi(points, 0.0);
  Scalar<DataVector> gamma0(points, 0.0), gamma1(points, -1.0),
      gamma2(points, 0.0);
  [&]<class... Temp>(tmpl::list<Temp...>) {
    TD::apply(make_not_null(&get<Metric>(source)),
              make_not_null(&get<Pi>(source)), make_not_null(&get<Phi>(source)),
              make_not_null(&get<Temp>(temporary))..., dmetric, dpi, dphi,
              get<Metric>(state), get<Pi>(state), get<Phi>(state), gamma0,
              gamma1, gamma2, gauge, mesh, time, coordinates, inverse_jacobian,
              mesh_velocity);
  }(typename TD::temporary_tags{});
  // Generic grid advection is zero in this source-isolation call because
  // its external derivatives are zero. Its original coarse contribution
  // remains in the full RHS to which the source difference will be added.
  return source;
}
}  // namespace gh
