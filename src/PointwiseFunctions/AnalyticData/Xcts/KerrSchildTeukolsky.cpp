// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/KerrSchildTeukolsky.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::AnalyticData::KerrSchildTeukolsky_detail {

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  const auto& kerr_schild_spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, Dim>>(kerr_schild_vars.get());
  const auto& teukolsky_spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>>(
          teukolsky_vars.get());
  tenex::evaluate<ti::i, ti::j>(conformal_metric,
                                kerr_schild_spatial_metric(ti::i, ti::j) +
                                    teukolsky_spatial_metric(ti::i, ti::j));
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    if (not(this->mesh.has_value() and this->inv_jacobian.has_value())) {
      ERROR("Need a mesh and a Jacobian for numeric differentiation.");
    }
    const auto& conformal_metric = cache->get_var(
        *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric,
                       this->mesh->get(), this->inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "The derivative of the perturbed conformal metric is computed "
        "numerically and requires DataVector grid data.");
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, Dim>>(kerr_schild_vars.get());
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(
          kerr_schild_vars.get());
  trace(trace_extrinsic_curvature, extrinsic_curvature, inv_spatial_metric);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  *lapse_times_conformal_factor_minus_one =
      get<gr::Tags::Lapse<DataType>>(kerr_schild_vars.get());
  get(*lapse_times_conformal_factor_minus_one) -= 1.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& dt_teukolsky_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>>>(
          teukolsky_vars.get());
  const auto& conformal_metric = cache->get_var(
      *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});

  const auto trace_dt_metric = trace(dt_teukolsky_metric, inv_conformal_metric);

  tenex::evaluate<ti::I, ti::J>(
      longitudinal_shift_background_minus_dt_conformal_metric,
      -inv_conformal_metric(ti::I, ti::K) * inv_conformal_metric(ti::J, ti::L) *
          (dt_teukolsky_metric(ti::k, ti::l) -
           conformal_metric(ti::k, ti::l) * trace_dt_metric() / 3.));
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  *shift_excess = get<gr::Tags::Shift<DataType, Dim>>(kerr_schild_vars.get());
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*energy_density) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*stress_trace) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  std::fill(momentum_density->begin(), momentum_density->end(), 0.);
}

template class KerrSchildTeukolskyVariables<double>;
template class KerrSchildTeukolskyVariables<DataVector>;

}  // namespace Xcts::AnalyticData::KerrSchildTeukolsky_detail

namespace Xcts::AnalyticData {

KerrSchildTeukolsky::KerrSchildTeukolsky(
    gr::Solutions::KerrSchild kerr_schild,
    const gr::Solutions::TeukolskyWave& teukolsky_wave,
    const Options::Context& context)
    : kerr_schild_(std::move(kerr_schild)),
      teukolsky_wave_(teukolsky_wave.with_minkowski_background(false)) {
  if (not kerr_schild_.zero_velocity()) {
    const auto& velocity = kerr_schild_.boost_velocity();
    PARSE_ERROR(context,
                "KerrSchildTeukolsky requires a stationary Kerr-Schild "
                "background, but the velocity is ["
                    << velocity[0] << ", " << velocity[1] << ", " << velocity[2]
                    << "].");
  }
}

void KerrSchildTeukolsky::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | kerr_schild_;
  p | teukolsky_wave_;
}

bool operator==(const KerrSchildTeukolsky& lhs,
                const KerrSchildTeukolsky& rhs) {
  return lhs.kerr_schild_ == rhs.kerr_schild_ and
         lhs.teukolsky_wave_ == rhs.teukolsky_wave_;
}

bool operator!=(const KerrSchildTeukolsky& lhs,
                const KerrSchildTeukolsky& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID KerrSchildTeukolsky::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    double, typename Xcts::AnalyticData::KerrSchildTeukolsky_detail::
                KerrSchildTeukolskyVariables<double>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::KerrSchildTeukolsky_detail::
                    KerrSchildTeukolskyVariables<DataVector>::Cache>;
