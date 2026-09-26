// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Newman Penrose quantity \f$\Psi_4\f$ using the characteristic
 * field U\f$^{8+}\f$ and complex vector \f$\bar{m}^i\f$.
 *
 * \details Computes \f$\Psi_4\f$ as: \f$\Psi_4 =
 * -\frac{1}{2} U^{8+}_{ij}\bar{m}^i\bar{m}^j\f$ with the characteristic field
 * \f$U^{8+} = (P^{(a}_i P^{b)}_j - \frac{1}{2}P_{ij}P^{ab})
 * (E_{ab} - \epsilon_a^{cd}n_dB_{cb})\f$,
 * and \f$\bar{m}^i\f$ = \f$(\hat{x}^i-i\hat{y}^i)\f$. The unit radial
 * vector is the metric-normalized `inertial_coords`. The first polarization
 * \f$\hat{x}^i\f$ is the normalized projection of the coordinate \f$x\f$
 * direction orthogonal to it, with the coordinate \f$y\f$ direction used
 * as a fallback near the \f$x\f$ axis. Metric cross products construct both
 * transverse vectors without subtracting nearly parallel vectors.
 *
 * Away from the coordinate singularities, the second polarization has the
 * orientation of the historical Gram-Schmidt projection of the coordinate
 * \f$y\f$ direction. For the projected \f$x\f$ basis this is
 * \f$\operatorname{sgn}(z)(\hat{r}\times\hat{x})^i\f$. On the
 * \f$xy\f$ plane, and when the first-vector fallback is used, choose
 * \f$(\hat{r}\times\hat{x})^i\f$. This coordinate-dependent convention
 * need not be continuous across its singularities. At the origin there is
 * no radial null tetrad; retain the historical metric-orthonormal Cartesian
 * \f$x\f$-\f$y\f$ extension for compatibility.
 *
 * \note This uses the vacuum expression for `gr::weyl_propagating` and
 * requires the spatial covariant derivative of extrinsic curvature.
 *
 */
template <typename Frame>
void psi_4(gsl::not_null<Scalar<ComplexDataVector>*> psi_4_result,
           const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
           const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
           const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
           const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
           const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
           const tnsr::I<DataVector, 3, Frame>& inertial_coords);

template <typename Frame>
Scalar<ComplexDataVector> psi_4(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);
/// @}

}  // namespace gr
