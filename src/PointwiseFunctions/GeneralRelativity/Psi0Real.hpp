// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the real part of the Newman-Penrose quantity \f$\Psi_0\f$
 * using \f$\mathrm{Re}(\Psi_0) =
 * -\frac{1}{2} U^{8-}_{ij}(\hat{x}^i \hat{x}^j -
 * \hat{y}^i \hat{y}^j)\f$.
 *
 * The first polarization vector \f$\hat{x}^i\f$ is constructed by projecting
 * the coordinate \f$x\f$ direction orthogonal to the radial direction, with
 * the coordinate \f$y\f$ direction used as a fallback near the \f$x\f$ axis.
 * The second polarization vector \f$\hat{y}^i\f$ is the coordinate \f$y\f$
 * direction Gram-Schmidt orthonormalized against the radial direction and
 * \f$\hat{x}^i\f$. Where those directions become linearly dependent, the
 * metric cross product of the radial direction and \f$\hat{x}^i\f$ is used as
 * a fallback. At the origin, where the radial direction is undefined, the
 * historical Cartesian \f$x\f$-\f$y\f$ basis is used.
 */
template <typename Frame>
void psi_0_real(
    gsl::not_null<Scalar<DataVector>*> psi_0_real_result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);

template <typename Frame>
Scalar<DataVector> psi_0_real(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);
/// @}

namespace Tags {
/// Computes the real part of the Newman-Penrose quantity \f$\Psi_0\f$ using
/// \f$\mathrm{Re}(\Psi_0) =
/// -\frac{1}{2} U^{8-}_{ij}(\hat{x}^i \hat{x}^j -
/// \hat{y}^i \hat{y}^j)\f$.
///
/// Can be retrieved using `gr::Tags::Psi0Real`
template <typename Frame>
struct Psi0RealCompute : Psi0Real<DataVector>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialRicci<DataVector, 3, Frame>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>,
      ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>,
                    tmpl::size_t<3>, Frame>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>,
      domain::Tags::Coordinates<3, Frame>>;

  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::ijj<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::II<DataVector, 3, Frame>&,
      const tnsr::I<DataVector, 3, Frame>&)>(&psi_0_real<Frame>);
  using base = Psi0Real<DataVector>;
};
}  // namespace Tags
}  // namespace gr
