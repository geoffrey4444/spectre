// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace sr {
template <size_t SpatialDim, typename Frame>
tnsr::Ab<double, SpatialDim, Frame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame>& velocity) noexcept {
  auto boost_matrix = make_with_value<tnsr::Ab<double, SpatialDim, Frame>>(
      get<0>(velocity), std::numeric_limits<double>::signaling_NaN());
  lorentz_boost_matrix<SpatialDim, Frame>(&boost_matrix, velocity);
  return boost_matrix;
}

template <size_t SpatialDim, typename Frame>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::Ab<double, SpatialDim, Frame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame>& velocity) noexcept {
  const double velocity_squared{get(dot_product(velocity, velocity))};
  const double lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};

  // For the spatial-spatial terms of the boost matrix, we need to compute
  // a prefactor, which is essentially kinetic energy per mass per velocity
  // squared. Specifically, the prefactor is
  //
  // kinetic_energy_per_v_squared = (lorentz_factor-1.0)/velocity^2
  //
  // We would like to avoid large numerical errors (or even FPEs) when
  // the velocity is zero or close to zero. The idea is to use a truncated
  // Taylor series, expanding in v, for velocities close enough to zero
  // that the truncation error in the series is less than roundoff error
  // of the full expression.
  //
  // Let kinetic_energy_per_v_squared = d, velocity = v. Then, series expanding
  // about small v gives
  //
  // d = [1/\sqrt(1-v^2) - 1] / v^2 ~ 1/2 + 3/8 v^2 + 5/16 v^4 + O(v^6)
  //
  // If you truncate the series as 1/2 + 3/8 v^2, the small-velocity
  // error is approximately
  //
  // eps_{small velocity} ~ 5/16 v^4
  //
  // Compare this to roundoff error in evaluating d directly. Let v->v+dv,
  // where dv represents roundoff error of v. Then
  //
  // eps_{roundoff} ~ (d with v->v+dv) - d
  //
  // Plotting both errors shows that these errors are comparable
  // (i.e., eps_{roundoff} ~ eps_{small velocity}) for
  //
  // v^2 ~ 10^-5.
  //
  // For v^2 >~ 10^{-5}, there is more error from the small-velocity
  // approximation than from roundoff. For v^2 <~ 10^{-5}, there is more
  // error from roundoff than from the small-velocity approximation.
  //
  // So, if v^2 < 10^{-5}, use d ~ 1/2 + (3/8) v^2. Otherwise, evaluate
  // the exact expression.

  double kinetic_energy_per_v_squared{0.5 + 0.375 * velocity_squared};
  if (velocity_squared > 1.0e-5) {
    kinetic_energy_per_v_squared = (lorentz_factor - 1.0) / velocity_squared;
  }

  get<0, 0>(*boost_matrix) = lorentz_factor;
  for (size_t i = 0; i < SpatialDim; ++i) {
    (*boost_matrix).get(0, i + 1) = velocity.get(i) * lorentz_factor;
    (*boost_matrix).get(i + 1, 0) = velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < SpatialDim; ++j) {
      (*boost_matrix).get(i + 1, j + 1) =
          velocity.get(i) * velocity.get(j) * kinetic_energy_per_v_squared;
    }
    (*boost_matrix).get(i + 1, i + 1) += 1.0;
  }
}
}  // namespace sr

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::Ab<double, DIM(data), FRAME(data)> sr::lorentz_boost_matrix( \
      const tnsr::I<double, DIM(data), FRAME(data)>& velocity) noexcept;      \
  template void sr::lorentz_boost_matrix(                                     \
      gsl::not_null<tnsr::Ab<double, DIM(data), FRAME(data)>*> boost_matrix,  \
      const tnsr::I<double, DIM(data), FRAME(data)>& velocity) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
