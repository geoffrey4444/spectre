// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/SpECMultiPenalty.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace gh::BoundaryCorrections {
namespace {
struct ShellInterpolation {
  ShellInterpolation(const ylm::Spherepack& source,
                     const ylm::Spherepack& target)
      : spectral(source.set_up_interpolation_info(target.theta_phi_points())) {
    const auto& source_theta = source.theta_points();
    const auto& target_theta = target.theta_points();
    const size_t source_n_phi = source.phi_points().size();
    const size_t target_n_phi = target.phi_points().size();
    const size_t common_n_phi = std::gcd(source_n_phi, target_n_phi);
    // Both uniform azimuthal grids start at zero. Their intersection is
    // another uniform grid with gcd(N_phi_source, N_phi_target) points.
    for (size_t s = 0; s < source_theta.size(); ++s) {
      for (size_t t = 0; t < target_theta.size(); ++t) {
        if (std::abs(source_theta[s] - target_theta[t]) < 1.e-13) {
          for (size_t p = 0; p < common_n_phi; ++p) {
            coincident_points.emplace_back(
                s + source_theta.size() * (p * (source_n_phi / common_n_phi)),
                t + target_theta.size() * (p * (target_n_phi / common_n_phi)));
          }
        }
      }
    }
  }

  ylm::Spherepack::InterpolationInfo<DataVector> spectral;
  std::vector<std::pair<size_t, size_t>> coincident_points;
};
}  // namespace

SpECMultiPenalty::SpECMultiPenalty(const size_t modes)
    : incoming_filter_modes_(modes) {}
SpECMultiPenalty::SpECMultiPenalty(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}
PUP::able::PUP_ID SpECMultiPenalty::my_PUP_ID = 0;  // NOLINT
void SpECMultiPenalty::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | incoming_filter_modes_;
}
std::unique_ptr<evolution::BoundaryCorrection> SpECMultiPenalty::get_clone()
    const {
  return std::make_unique<SpECMultiPenalty>(*this);
}

DataVector SpECMultiPenalty::package_native_face(
    const Metric::type& metric, const Pi::type& pi, const Phi::type& phi,
    const Gamma1::type& gamma1, const Gamma2::type& gamma2,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const Normal::type& normal, const Magnitude::type& magnitude,
    const std::optional<Velocity::type>& mesh_velocity,
    const Mesh<3>& volume_mesh, const Direction<3>& direction,
    const Jac::type& jac, const InvJac::type& inv_jac) const {
  if (direction.dimension() != 0 or
      volume_mesh.basis(0) != Spectral::Basis::Chebyshev or
      volume_mesh.quadrature(0) != Spectral::Quadrature::GaussLobatto or
      volume_mesh.basis(1) != Spectral::Basis::SphericalHarmonic or
      volume_mesh.basis(2) != Spectral::Basis::SphericalHarmonic or
      volume_mesh.quadrature(1) != Spectral::Quadrature::Gauss or
      volume_mesh.quadrature(2) != Spectral::Quadrature::Equiangular or
      volume_mesh.extents(2) != 2 * volume_mesh.extents(1) - 1) {
    ERROR(
        "SpECMultiPenalty requires radial faces of full Chebyshev-Lobatto "
        "shells with the standard (L+1) by (2L+1) Spherepack angular grid.");
  }
  DataVector data(get(lapse).size() *
                  Variables<NativeTags>::number_of_independent_components);
  Variables<NativeTags> native(data.data(), data.size());
  get<Metric>(native) = metric;
  get<Pi>(native) = pi;
  get<Phi>(native) = phi;
  get<Gamma1>(native) = gamma1;
  get<Gamma2>(native) = gamma2;
  get<Normal>(native) = normal;
  get<Magnitude>(native) = magnitude;
  get<gr::Tags::Lapse<DataVector>>(native) = lapse;
  get<gr::Tags::Shift<DataVector, 3>>(native) = shift;
  if (mesh_velocity.has_value()) {
    get<Velocity>(native) = *mesh_velocity;
  } else {
    for (auto& component : get<Velocity>(native)) {
      component = 0.;
    }
  }
  ::dg::project_tensor_to_boundary(make_not_null(&get<Jac>(native)), jac,
                                   volume_mesh, direction);
  ::dg::project_tensor_to_boundary(make_not_null(&get<InvJac>(native)), inv_jac,
                                   volume_mesh, direction);
  return data;
}

Variables<SpECMultiPenalty::Fields> SpECMultiPenalty::native_face_terms(
    const DataVector& local, const DataVector& donor, const Mesh<2>& face,
    const Mesh<2>& donor_face, const size_t radial_points) const {
  const size_t n = face.number_of_grid_points();
  const size_t donor_points = donor_face.number_of_grid_points();
  ASSERT(local.size() ==
                 n * Variables<NativeTags>::number_of_independent_components and
             donor.size() >=
                 donor_points *
                     Variables<Fields>::number_of_independent_components,
         "Invalid native GH trace size");
  const Variables<NativeTags> interior(const_cast<double*>(local.data()),
                                       local.size());
  Variables<Fields> received(n);
  if (face == donor_face) {
    std::copy_n(donor.data(), received.size(), received.data());
  } else {
    const auto& source = ylm::get_spherepack_cache(donor_face.extents(0) - 1);
    const auto& target = ylm::get_spherepack_cache(face.extents(0) - 1);
    // LTS can couple the same two grids many times. Only the donor values
    // change between those evaluations.
    static thread_local std::map<std::pair<size_t, size_t>, ShellInterpolation>
        interpolations;
    const auto& interpolation =
        interpolations
            .try_emplace(std::pair{source.l_max(), target.l_max()}, source,
                         target)
            .first->second;
    for (size_t i = 0; i < received.number_of_independent_components; ++i) {
      DataVector result(received.data() + i * n, n);
      const double* donor_component = donor.data() + i * donor_points;
      source.interpolate(make_not_null(&result), make_not_null(donor_component),
                         interpolation.spectral);
      // SpEC's SpectralInterpolator copies coincident nodes before filtering.
      // Projection onto spherical harmonics can change underresolved nodal
      // data.
      for (const auto& [from, to] : interpolation.coincident_points) {
        result[to] = donor_component[from];
      }
    }
    if (incoming_filter_modes_ != 0) {
      // The filter owns mutable tensor-harmonic matrices and scratch space.
      // Separate them by receiving resolution and thread.
      static thread_local std::map<std::pair<size_t, size_t>,
                                   Filters::SphericalShell<Fields>>
          filters;
      auto& filter =
          filters
              .try_emplace(std::pair{target.l_max(), incoming_filter_modes_},
                           incoming_filter_modes_, std::nullopt, std::nullopt,
                           true, std::nullopt, false, false, std::nullopt,
                           std::nullopt)
              .first->second;
      filter.apply_on_boundary(make_not_null(&received), face,
                               get<InvJac>(interior), get<Jac>(interior));
    }
  }

  const auto inverse_metric =
      determinant_and_inverse(gr::spatial_metric(get<Metric>(interior))).second;
  const auto& gamma2 = get<Gamma2>(interior);
  const auto& normal = get<Normal>(interior);
  // Transform both traces separately to preserve SpEC's subtraction order.
  auto chars =
      gh::characteristic_fields(gamma2, inverse_metric, get<Metric>(interior),
                                get<Pi>(interior), get<Phi>(interior), normal);
  chars -=
      gh::characteristic_fields(gamma2, inverse_metric, get<Metric>(received),
                                get<Pi>(received), get<Phi>(received), normal);
  const auto speeds = gh::characteristic_speeds(
      get<Gamma1>(interior), get<gr::Tags::Lapse<DataVector>>(interior),
      get<gr::Tags::Shift<DataVector, 3>>(interior), normal,
      std::optional{get<Velocity>(interior)});
  const double coefficient =
      0.5 * square(static_cast<double>(radial_points - 1));
  auto apply_speed = [&](auto& field, const DataVector& speed) {
    for (auto& component : field) {
      for (size_t i = 0; i < n; ++i) {
        component[i] *= std::min(speed[i], 0.) *
                        get(get<Magnitude>(interior))[i] * coefficient;
      }
    }
  };
  auto& g = get<gh::Tags::VSpacetimeMetric<DataVector, 3>>(chars);
  auto& v0 = get<gh::Tags::VZero<DataVector, 3>>(chars);
  auto& vp = get<gh::Tags::VPlus<DataVector, 3>>(chars);
  auto& vm = get<gh::Tags::VMinus<DataVector, 3>>(chars);
  apply_speed(g, speeds[0]);
  apply_speed(v0, speeds[1]);
  apply_speed(vp, speeds[2]);
  apply_speed(vm, speeds[3]);
  return gh::evolved_fields_from_characteristic_fields(gamma2, g, v0, vp, vm,
                                                       normal);
}
}  // namespace gh::BoundaryCorrections
