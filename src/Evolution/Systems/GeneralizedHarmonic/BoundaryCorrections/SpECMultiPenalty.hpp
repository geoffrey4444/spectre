// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/AveragedUpwindPenalty.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"

namespace gh::BoundaryCorrections {
/// SpEC's forward-in-time MultiPenalty on aligned, full Chebyshev-Lobatto
/// spherical shells. The donor's spherical harmonic expansion is evaluated at
/// the receiver's nodes before the receiver forms the characteristic jump.
/// For each incoming field the RHS gains
/// \f$\tfrac12 (N_r-1)^2 \lvert d\xi\rvert_{\gamma^{-1}}\,\lambda (u_{\rm
/// local}-u_{\rm received})\f$, where the normal magnitude is computed with the
/// receiver's spatial metric. The optional filter acts on the interpolated
/// fundamental fields on an unequal grid. External boundaries use the usual GH
/// boundary treatment.
class SpECMultiPenalty final : public evolution::BoundaryCorrection {
 public:
  struct IncomingFilterModes {
    using type = size_t;
    static constexpr Options::String help =
        "Number of highest tensor-harmonic modes to remove from the received "
        "fields on unequal angular grids. Zero disables incoming filtering.";
  };
  using options = tmpl::list<IncomingFilterModes>;
  static constexpr Options::String help =
      "Receiving-shell characteristic penalty with spectral point "
      "interpolation.";
  SpECMultiPenalty() = default;
  explicit SpECMultiPenalty(size_t incoming_filter_modes);
  explicit SpECMultiPenalty(CkMigrateMessage* msg);
  WRAPPED_PUPable_decl_template(SpECMultiPenalty);  // NOLINT
  void pup(PUP::er& p) override;                    // NOLINT
  std::unique_ptr<evolution::BoundaryCorrection> get_clone() const override;
  bool uses_native_face_data() const override { return true; }

  using ExteriorCorrection = AveragedUpwindPenalty<3>;
  using dg_package_field_tags = ExteriorCorrection::dg_package_field_tags;
  using dg_package_data_temporary_tags =
      ExteriorCorrection::dg_package_data_temporary_tags;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using Jac = domain::Tags::Jacobian<3, Frame::Grid, Frame::Inertial>;
  using InvJac = domain::Tags::InverseJacobian<3, Frame::Grid, Frame::Inertial>;
  using dg_package_data_volume_tags = tmpl::list<Jac, InvJac>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  using Metric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = gh::Tags::Pi<DataVector, 3>;
  using Phi = gh::Tags::Phi<DataVector, 3>;
  using Fields = tmpl::list<Metric, Pi, Phi>;
  using Gamma1 = gh::Tags::ConstraintGamma1;
  using Gamma2 = gh::Tags::ConstraintGamma2;
  using Normal = evolution::dg::Tags::NormalCovector<3>;
  using Magnitude = evolution::dg::Tags::MagnitudeOfNormal;
  using Velocity = ExteriorCorrection::MeshVelocity;
  using NativeTags =
      tmpl::append<Fields,
                   tmpl::list<Gamma1, Gamma2, Normal, Magnitude, Velocity,
                              gr::Tags::Lapse<DataVector>,
                              gr::Tags::Shift<DataVector, 3>, Jac, InvJac>>;

  DataVector package_native_face(
      const Metric::type& metric, const Pi::type& pi, const Phi::type& phi,
      const Gamma1::type& gamma1, const Gamma2::type& gamma2,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const Normal::type& normal, const Magnitude::type& magnitude,
      const std::optional<Velocity::type>& mesh_velocity,
      const Mesh<3>& volume_mesh, const Direction<3>& direction,
      const Jac::type& jac, const InvJac::type& inv_jac) const;

  Variables<Fields> native_face_terms(const DataVector& local,
                                      const DataVector& donor,
                                      const Mesh<2>& face,
                                      const Mesh<2>& donor_face,
                                      size_t radial_points) const;

 private:
  size_t incoming_filter_modes_{0};
};
}  // namespace gh::BoundaryCorrections
