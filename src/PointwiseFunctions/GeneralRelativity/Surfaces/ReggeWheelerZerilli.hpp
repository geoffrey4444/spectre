// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {

/*!
 * \ingroup SurfacesGroup
 * \brief Finite-radius Regge-Wheeler-Zerilli master functions and strain.
 *
 * \details The stored strain quantity is \f$r h_{\ell m}\f$, matching the
 * finite-radius quantity that SpEC writes to `rh_FiniteRadii_CodeUnits.h5`.
 */
struct ReggeWheelerZerilli {
  ReggeWheelerZerilli() = default;
  explicit ReggeWheelerZerilli(size_t l_max);

  ComplexModalVector phi_plus{};
  ComplexModalVector phi_minus{};
  ComplexModalVector r_times_strain{};
};

/*!
 * \ingroup SurfacesGroup
 * \brief Proper-metric metadata on a finite extraction sphere.
 */
struct ExtractionSphereMetadata {
  double average_lapse{0.0};
  double areal_radius{0.0};
};

/*!
 * \ingroup SurfacesGroup
 * \brief Compute flat-background finite-radius RWZ master functions and
 * \f$r h_{\ell m}\f$ from gauge-invariant mode amplitudes.
 *
 * \details The amplitudes follow the same conventions used by the standard
 * SpEC BBH RWZ extraction path. Inputs and outputs are stored in Goldberg
 * ordering, with storage size `square(l_max + 1)`. The overloads below either
 * write into `rwz_quantities` or return the result by value.
 */
void regge_wheeler_zerilli_moncrief(
    gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, size_t l_max, double extraction_radius);
ReggeWheelerZerilli regge_wheeler_zerilli_moncrief(
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, size_t l_max, double extraction_radius);

/*!
 * \ingroup SurfacesGroup
 * \brief Compute finite-radius RWZ quantities from GH spacetime variables on an
 * extraction sphere.
 *
 * \details The routine interprets the supplied GH variables as a perturbation
 * of flat space written on a coordinate sphere of radius `extraction_radius`
 * centered at `center`, decomposes the perturbation into tensor spherical
 * harmonics, and evaluates the Moncrief gauge invariants. The overloads below
 * either write into `rwz_quantities` or return the result by value.
 */
void regge_wheeler_zerilli_moncrief_from_gh_vars(
    gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    double extraction_radius);
ReggeWheelerZerilli regge_wheeler_zerilli_moncrief_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    double extraction_radius);

/*!
 * \ingroup SurfacesGroup
 * \brief Compute finite-radius \f$r \Psi_4^{\ell m}\f$ from 3+1 quantities on
 * an extraction sphere.
 *
 * \details This follows the same high-level path as SpEC finite-radius
 * extraction: form the propagating Weyl characteristic field \f$U^{8+}\f$ on
 * the extraction sphere, decompose it into tensor spherical harmonics, select
 * the spin-weight \f$-2\f$ component, and scale by the coordinate radius.
 * Inputs and outputs use Goldberg ordering, with storage size
 * `square(l_max + 1)`. The overloads below either write into
 * `r_times_psi_4` or return the result by value.
 */
void psi_4_modes_from_tensors(
    gsl::not_null<ComplexModalVector*> r_times_psi_4,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>&
        cov_deriv_extrinsic_curvature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    double extraction_radius);
ComplexModalVector psi_4_modes_from_tensors(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>&
        cov_deriv_extrinsic_curvature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    double extraction_radius);

/*!
 * \ingroup SurfacesGroup
 * \brief Compute average lapse and areal radius on an extraction sphere from
 * the GH spacetime metric.
 *
 * \details The overloads below either write into `metadata` or return the
 * result by value.
 */
void extraction_sphere_metadata_from_gh_vars(
    gsl::not_null<ExtractionSphereMetadata*> metadata,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper);
ExtractionSphereMetadata extraction_sphere_metadata_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper);

}  // namespace gr::surfaces
