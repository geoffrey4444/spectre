// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename TagsList>
class Variables;

// IWYU pragma: no_forward_declare DataVector
// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> dt_phi,
    const tnsr::aa<DataVector, Dim>& spacetime_metric,
    const tnsr::aa<DataVector, Dim>& pi, const tnsr::iaa<DataVector, Dim>& phi,
    const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim>& d_pi,
    const tnsr::ijaa<DataVector, Dim>& d_phi, const Scalar<DataVector>& gamma0,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::a<DataVector, Dim>& gauge_function,
    const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::AA<DataVector, Dim>& inverse_spacetime_metric,
    const tnsr::a<DataVector, Dim>& trace_christoffel,
    const tnsr::abb<DataVector, Dim>& christoffel_first_kind,
    const tnsr::Abb<DataVector, Dim>& christoffel_second_kind,
    const tnsr::A<DataVector, Dim>& normal_spacetime_vector,
    const tnsr::a<DataVector, Dim>& normal_spacetime_one_form) {
  const size_t n_pts = shift.begin()->size();

  // Scalar: TempScalar<0> = gamma12
  //         TempScalar<1> = pi_contract_two_normal_spacetime_vectors
  //         TempScalar<2> = normal_dot_one_index_constraint
  //         TempScalar<3> = gamma1p1
  // a: Tempa<0> = pi_dot_normal_spacetime_vector
  //    Tempa<1> = phi_contract_two_normal_spacetime_vectors
  //    Tempa<2> = one_index_constraint
  // aa: shift_dot_three_index_constraint
  // ia: phi_dot_normal_spacetime_vector
  // aB: pi_2_up
  // iaa: three_index_constraint
  // Iaa: phi_1_up
  // abC: phi_3_up
  //      christoffel_first_kind_3_up
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>,
                       ::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>,
                       ::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::TempaB<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::TempIaa<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::TempabC<0, Dim, Frame::Inertial, DataVector>,
                       ::Tags::TempabC<1, Dim, Frame::Inertial, DataVector>>>
      buffer(n_pts);

  get(get<::Tags::TempScalar<0>>(buffer)) = gamma1.get() * gamma2.get();
  const DataVector& gamma12 = get(get<::Tags::TempScalar<0>>(buffer));

  tnsr::Iaa<DataVector, Dim>& phi_1_up =
      get<::Tags::TempIaa<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_1_up.get(m, mu, nu) =
            inverse_spatial_metric.get(m, 0) * phi.get(0, mu, nu);
        for (size_t n = 1; n < Dim; ++n) {
          phi_1_up.get(m, mu, nu) +=
              inverse_spatial_metric.get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  tnsr::abC<DataVector, Dim>& phi_3_up =
      get<::Tags::TempabC<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        phi_3_up.get(m, nu, alpha) =
            inverse_spacetime_metric.get(alpha, 0) * phi.get(m, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          phi_3_up.get(m, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  tnsr::aB<DataVector, Dim>& pi_2_up =
      get<::Tags::TempaB<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      pi_2_up.get(nu, alpha) =
          inverse_spacetime_metric.get(alpha, 0) * pi.get(nu, 0);
      for (size_t beta = 1; beta < Dim + 1; ++beta) {
        pi_2_up.get(nu, alpha) +=
            inverse_spacetime_metric.get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  tnsr::abC<DataVector, Dim>& christoffel_first_kind_3_up =
      get<::Tags::TempabC<1, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        christoffel_first_kind_3_up.get(mu, nu, alpha) =
            inverse_spacetime_metric.get(alpha, 0) *
            christoffel_first_kind.get(mu, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up.get(mu, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) *
              christoffel_first_kind.get(mu, nu, beta);
        }
      }
    }
  }

  tnsr::a<DataVector, Dim>& pi_dot_normal_spacetime_vector =
      get<::Tags::Tempa<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_dot_normal_spacetime_vector.get(mu) =
        get<0>(normal_spacetime_vector) * pi.get(0, mu);
    for (size_t nu = 1; nu < Dim + 1; ++nu) {
      pi_dot_normal_spacetime_vector.get(mu) +=
          normal_spacetime_vector.get(nu) * pi.get(nu, mu);
    }
  }

  DataVector& pi_contract_two_normal_spacetime_vectors =
      get(get<::Tags::TempScalar<1>>(buffer));
  pi_contract_two_normal_spacetime_vectors =
      get<0>(normal_spacetime_vector) * get<0>(pi_dot_normal_spacetime_vector);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    pi_contract_two_normal_spacetime_vectors +=
        normal_spacetime_vector.get(mu) *
        pi_dot_normal_spacetime_vector.get(mu);
  }

  tnsr::ia<DataVector, Dim>& phi_dot_normal_spacetime_vector =
      get<::Tags::Tempia<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      phi_dot_normal_spacetime_vector.get(n, nu) =
          get<0>(normal_spacetime_vector) * phi.get(n, 0, nu);
      for (size_t mu = 1; mu < Dim + 1; ++mu) {
        phi_dot_normal_spacetime_vector.get(n, nu) +=
            normal_spacetime_vector.get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim>& phi_contract_two_normal_spacetime_vectors =
      get<::Tags::Tempa<1, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    phi_contract_two_normal_spacetime_vectors.get(n) =
        get<0>(normal_spacetime_vector) *
        phi_dot_normal_spacetime_vector.get(n, 0);
    for (size_t mu = 1; mu < Dim + 1; ++mu) {
      phi_contract_two_normal_spacetime_vectors.get(n) +=
          normal_spacetime_vector.get(mu) *
          phi_dot_normal_spacetime_vector.get(n, mu);
    }
  }

  tnsr::iaa<DataVector, Dim>& three_index_constraint =
      get<::Tags::Tempiaa<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        const size_t storage_index =
            three_index_constraint.get_storage_index(n, mu, nu);
        three_index_constraint[storage_index] =
            d_spacetime_metric[storage_index] - phi[storage_index];
      }
    }
  }

  tnsr::a<DataVector, Dim>& one_index_constraint =
      get<::Tags::Tempa<2, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    one_index_constraint.get(nu) =
        gauge_function.get(nu) + trace_christoffel.get(nu);
  }

  DataVector& normal_dot_one_index_constraint =
      get(get<::Tags::TempScalar<2>>(buffer));
  normal_dot_one_index_constraint =
      get<0>(normal_spacetime_vector) * get<0>(one_index_constraint);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    normal_dot_one_index_constraint +=
        normal_spacetime_vector.get(mu) * one_index_constraint.get(mu);
  }

  get(get<::Tags::TempScalar<3>>(buffer)) = 1.0 + gamma1.get();
  const DataVector& gamma1p1 = get(get<::Tags::TempScalar<3>>(buffer));

  tnsr::aa<DataVector, Dim>& shift_dot_three_index_constraint =
      get<::Tags::Tempaa<0, Dim, Frame::Inertial, DataVector>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      shift_dot_three_index_constraint.get(mu, nu) =
          get<0>(shift) * three_index_constraint.get(0, mu, nu);
      for (size_t m = 1; m < Dim; ++m) {
        shift_dot_three_index_constraint.get(mu, nu) +=
            shift.get(m) * three_index_constraint.get(m, mu, nu);
      }
    }
  }

  // Here are the actual equations

  // Equation for dt_spacetime_metric
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      const size_t storage_index_mu_nu =
          dt_spacetime_metric->get_storage_index(mu, nu);
      (*dt_spacetime_metric)[storage_index_mu_nu] =
          -lapse.get() * pi[storage_index_mu_nu];
      (*dt_spacetime_metric)[storage_index_mu_nu] +=
          gamma1p1 * shift_dot_three_index_constraint[storage_index_mu_nu];
      for (size_t m = 0; m < Dim; ++m) {
        (*dt_spacetime_metric)[storage_index_mu_nu] +=
            shift.get(m) * phi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_pi
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      const size_t storage_index_mu_nu = dt_pi->get_storage_index(mu, nu);
      (*dt_pi)[storage_index_mu_nu] =
          -spacetime_deriv_gauge_function.get(mu, nu) -
          spacetime_deriv_gauge_function.get(nu, mu) -
          0.5 * pi_contract_two_normal_spacetime_vectors *
              pi[storage_index_mu_nu] +
          gamma0.get() * (normal_spacetime_one_form.get(mu) *
                              one_index_constraint.get(nu) +
                          normal_spacetime_one_form.get(nu) *
                              one_index_constraint.get(mu)) -
          gamma0.get() * spacetime_metric[storage_index_mu_nu] *
              normal_dot_one_index_constraint;

      for (size_t delta = 0; delta < Dim + 1; ++delta) {
        (*dt_pi)[storage_index_mu_nu] +=
            2 * christoffel_second_kind.get(delta, mu, nu) *
                gauge_function.get(delta) -
            2 * pi.get(mu, delta) * pi_2_up.get(nu, delta);

        for (size_t n = 0; n < Dim; ++n) {
          (*dt_pi)[storage_index_mu_nu] +=
              2 * phi_1_up.get(n, mu, delta) * phi_3_up.get(n, nu, delta);
        }

        for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
          (*dt_pi)[storage_index_mu_nu] -=
              2. * christoffel_first_kind_3_up.get(mu, alpha, delta) *
              christoffel_first_kind_3_up.get(nu, delta, alpha);
        }
      }

      for (size_t m = 0; m < Dim; ++m) {
        (*dt_pi)[storage_index_mu_nu] -=
            pi_dot_normal_spacetime_vector.get(m + 1) * phi_1_up.get(m, mu, nu);

        for (size_t n = 0; n < Dim; ++n) {
          (*dt_pi)[storage_index_mu_nu] -=
              inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
        }
      }

      (*dt_pi)[storage_index_mu_nu] *= lapse.get();

      (*dt_pi)[storage_index_mu_nu] +=
          gamma12 * shift_dot_three_index_constraint[storage_index_mu_nu];

      for (size_t m = 0; m < Dim; ++m) {
        // DualFrame term
        (*dt_pi)[storage_index_mu_nu] += shift.get(m) * d_pi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_phi
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        const size_t storage_index_i_mu_nu =
            dt_phi->get_storage_index(i, mu, nu);
        (*dt_phi)[storage_index_i_mu_nu] =
            0.5 * pi.get(mu, nu) *
                phi_contract_two_normal_spacetime_vectors.get(i) -
            d_pi.get(i, mu, nu) +
            gamma2.get() * three_index_constraint[storage_index_i_mu_nu];
        for (size_t n = 0; n < Dim; ++n) {
          (*dt_phi)[storage_index_i_mu_nu] +=
              phi_dot_normal_spacetime_vector.get(i, n + 1) *
              phi_1_up.get(n, mu, nu);
        }

        (*dt_phi)[storage_index_i_mu_nu] *= lapse.get();
        for (size_t m = 0; m < Dim; ++m) {
          (*dt_phi)[storage_index_i_mu_nu] +=
              shift.get(m) * d_phi.get(m, i, mu, nu);
        }
      }
    }
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*>
        spacetime_metric_normal_dot_flux,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
    const tnsr::aa<DataVector, Dim>& spacetime_metric) noexcept {
  destructive_resize_components(pi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(phi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(spacetime_metric_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  for (size_t storage_index = 0; storage_index < pi_normal_dot_flux->size();
       ++storage_index) {
    (*pi_normal_dot_flux)[storage_index] = 0.0;
    (*spacetime_metric_normal_dot_flux)[storage_index] = 0.0;
  }

  for (size_t storage_index = 0; storage_index < phi_normal_dot_flux->size();
       ++storage_index) {
    (*phi_normal_dot_flux)[storage_index] = 0.0;
  }
}
/// \endcond
}  // namespace GeneralizedHarmonic

// Explicit instantiations of structs defined in `Equations.cpp` as well as of
// `partial_derivatives` function for use in the computation of spatial
// derivatives of `gradients_tags`, and of the initial gauge source function
// (needed in `Initialize.hpp`).
/// \cond
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

using derivative_frame = Frame::Inertial;

template <size_t Dim>
using derivative_tags_initial_gauge =
    tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, derivative_frame>>;

template <size_t Dim>
using variables_tags_initial_gauge =
    tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, derivative_frame>>;

template <size_t Dim>
using derivative_tags =
    typename GeneralizedHarmonic::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename GeneralizedHarmonic::System<Dim>::variables_tag::tags_list;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                 \
  template struct GeneralizedHarmonic::ComputeDuDt<DIM(data)>;               \
  template struct GeneralizedHarmonic::ComputeNormalDotFluxes<DIM(data)>;    \
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;   \
  template Variables<db::wrap_tags_in<                                       \
      ::Tags::deriv, derivative_tags_initial_gauge<DIM(data)>,               \
      tmpl::size_t<DIM(data)>, derivative_frame>>                            \
  partial_derivatives<derivative_tags_initial_gauge<DIM(data)>,              \
                      variables_tags_initial_gauge<DIM(data)>, DIM(data),    \
                      derivative_frame>(                                     \
      const Variables<variables_tags_initial_gauge<DIM(data)>>& u,           \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
/// \endcond
