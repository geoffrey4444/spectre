// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

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
 * \brief Computes the real part of the vacuum Newman-Penrose scalar
 * \f$\Psi_0\f$
 * using \f$\mathrm{Re}(\Psi_0) =
 * -\frac{1}{2} U^{8-}_{ij}(\hat{x}^i \hat{x}^j -
 * \hat{y}^i \hat{y}^j)\f$.
 *
 * The unit radial vector is the metric-normalized `inertial_coords`. The
 * first polarization vector \f$\hat{x}^i\f$ is the normalized projection of
 * the coordinate \f$x\f$ direction orthogonal to it, with the coordinate
 * \f$y\f$ direction used as a fallback near the \f$x\f$ axis. Metric cross
 * products construct this projection and the second polarization vector
 * \f$\hat{y}^i\f$ without subtracting nearly parallel vectors. The sign of
 * the second vector cancels in the real part.
 *
 * This coordinate-dependent basis need not be continuous across its
 * coordinate singularities. At the origin there is no radial null tetrad;
 * the metric-orthonormalized Cartesian \f$x\f$-\f$y\f$ basis supplies an
 * arbitrary finite extension compatible with `gr::psi_4`.
 *
 * \note This uses the vacuum expression for `gr::weyl_propagating`. The
 * derivative of extrinsic curvature must be its spatial covariant
 * derivative. Matter corrections are not included. This local,
 * tetrad-dependent quantity is not a gravitational-wave energy or flux.
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
/// Computes the real part of the vacuum Newman-Penrose scalar \f$\Psi_0\f$
/// using
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
      gr::Tags::CovariantDerivativeOfExtrinsicCurvature<DataVector, 3, Frame>,
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
