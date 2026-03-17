// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
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

}  // namespace gr::surfaces
