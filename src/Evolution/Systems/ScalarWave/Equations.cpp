// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Equations.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"   // IWYU pragma: keep
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
// Doxygen is not good at templates and so we have to hide the definition.
/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
    const Scalar<DataVector>& gamma2) noexcept {
  get(*dt_psi) = -get(pi);
  get(*dt_pi) = -get<0, 0>(d_phi);
  for (size_t d = 1; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = -d_pi.get(d) + get(gamma2) * (d_psi.get(d) - phi.get(d));
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
    const Scalar<DataVector>& pi) noexcept {
  destructive_resize_components(pi_normal_dot_flux, get(pi).size());
  destructive_resize_components(phi_normal_dot_flux, get(pi).size());
  destructive_resize_components(psi_normal_dot_flux, get(pi).size());
  get(*pi_normal_dot_flux) = 0.0;
  get(*psi_normal_dot_flux) = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    phi_normal_dot_flux->get(i) = 0.0;
  }
}

template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_v_psi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_v_zero,
    const gsl::not_null<Scalar<DataVector>*> packaged_v_plus,
    const gsl::not_null<Scalar<DataVector>*> packaged_v_minus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_n_times_v_plus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_n_times_v_minus,
    const gsl::not_null<Scalar<DataVector>*> packaged_gamma2_v_psi,
    const gsl::not_null<std::array<DataVector, 4>*> packaged_char_speeds,

    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const std::array<DataVector, 4>& char_speeds,
    const Scalar<DataVector>& constraint_gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Computes the contribution to the numerical flux from one side of the
  // interface.
  //
  // Note: when PenaltyFlux::operator() is called, an Element passes in its own
  // packaged data to fill the interior fields, and its neighbor's packaged data
  // to fill the exterior fields. This introduces a sign flip for each normal
  // used in computing the exterior fields.
  get(*packaged_v_psi) = char_speeds[0] * get(v_psi);
  *packaged_v_zero = v_zero;
  for (size_t i = 0; i < Dim; ++i) {
    packaged_v_zero->get(i) *= char_speeds[1];
  }
  get(*packaged_v_plus) = char_speeds[2] * get(v_plus);
  get(*packaged_v_minus) = char_speeds[3] * get(v_minus);
  for (size_t d = 0; d < Dim; ++d) {
    packaged_n_times_v_plus->get(d) =
        get(*packaged_v_plus) * interface_unit_normal.get(d);
    packaged_n_times_v_minus->get(d) =
        get(*packaged_v_minus) * interface_unit_normal.get(d);
  }
  *packaged_char_speeds = char_speeds;
  get(*packaged_gamma2_v_psi) = get(constraint_gamma2) * get(*packaged_v_psi);
}

template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,

    const Scalar<DataVector>& v_psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_int,
    const Scalar<DataVector>& v_plus_int, const Scalar<DataVector>& v_minus_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_v_plus_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_v_minus_int,
    const Scalar<DataVector>& constraint_gamma2_v_psi_int,
    const std::array<DataVector, 4>& char_speeds_int,

    const Scalar<DataVector>& v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero_ext,
    const Scalar<DataVector>& v_plus_ext, const Scalar<DataVector>& v_minus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_times_v_plus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_times_v_minus_ext,
    const Scalar<DataVector>& constraint_gamma2_v_psi_ext,
    const std::array<DataVector, 4>& char_speeds_ext) const noexcept {
  const DataVector weighted_lambda_psi_int = step_function(-char_speeds_int[0]);
  const DataVector weighted_lambda_psi_ext = -step_function(char_speeds_ext[0]);

  const DataVector weighted_lambda_zero_int =
      step_function(-char_speeds_int[1]);
  const DataVector weighted_lambda_zero_ext =
      -step_function(char_speeds_ext[1]);

  const DataVector weighted_lambda_plus_int =
      step_function(-char_speeds_int[2]);
  const DataVector weighted_lambda_plus_ext =
      -step_function(char_speeds_ext[2]);

  const DataVector weighted_lambda_minus_int =
      step_function(-char_speeds_int[3]);
  const DataVector weighted_lambda_minus_ext =
      -step_function(char_speeds_ext[3]);

  // D_psi = Theta(-lambda_psi^{ext}) lambda_psi^{ext} v_psi^{ext}
  //       - Theta(-lambda_psi^{int}) lambda_psi^{int} v_psi^{int}
  // where the unit normals on both sides point in the same direction, out
  // of the current element. Since lambda_psi from the neighbor is computing
  // with the normal vector pointing into the current element in the code,
  // we need to swap the sign of lambda_psi^{ext}. Theta is the heaviside step
  // function with Theta(0) = 0.
  psi_boundary_correction->get() = weighted_lambda_psi_ext * get(v_psi_ext) -
                                   weighted_lambda_psi_int * get(v_psi_int);

  get(*pi_boundary_correction) =
      0.5 * (weighted_lambda_plus_ext * get(v_plus_ext) +
             weighted_lambda_minus_ext * get(v_minus_ext)) +
      weighted_lambda_psi_ext * get(constraint_gamma2_v_psi_ext)

      - 0.5 * (weighted_lambda_plus_int * get(v_plus_int) +
               weighted_lambda_minus_int * get(v_minus_int)) -
      weighted_lambda_psi_int * get(constraint_gamma2_v_psi_int);

  for (size_t d = 0; d < Dim; ++d) {
    // Overall minus sign on ext because of normal vector is opposite direction.
    phi_boundary_correction->get(d) =
        -0.5 *
            (weighted_lambda_minus_ext * minus_normal_times_v_minus_ext.get(d) -
             weighted_lambda_plus_ext * minus_normal_times_v_plus_ext.get(d)) +
        weighted_lambda_zero_ext * v_zero_ext.get(d)

        - 0.5 * (weighted_lambda_plus_int * normal_times_v_plus_int.get(d) -
                 weighted_lambda_minus_int * normal_times_v_minus_int.get(d)) -
        weighted_lambda_zero_int * v_zero_int.get(d);
  }
}
/// \endcond
}  // namespace ScalarWave

// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using derivative_tags = typename ScalarWave::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename ScalarWave::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class ScalarWave::ComputeDuDt<DIM(data)>;                         \
  template class ScalarWave::ComputeNormalDotFluxes<DIM(data)>;              \
  template class ScalarWave::UpwindPenaltyCorrection<DIM(data)>;             \
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
