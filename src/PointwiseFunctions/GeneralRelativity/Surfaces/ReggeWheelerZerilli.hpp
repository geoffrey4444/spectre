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
 * ordering, with storage size `square(l_max + 1)`.
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

void extraction_sphere_metadata_from_gh_vars(
    gsl::not_null<ExtractionSphereMetadata*> metadata,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper);

ExtractionSphereMetadata extraction_sphere_metadata_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper);

}  // namespace gr::surfaces
