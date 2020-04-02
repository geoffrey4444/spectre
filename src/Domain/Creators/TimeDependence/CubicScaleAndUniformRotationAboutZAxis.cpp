// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/Composition.hpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
  template class Composition<CubicScale<1>, UniformRotationAboutZAxis<1>>;
  template class Composition<CubicScale<2>, UniformRotationAboutZAxis<2>>;
  template class Composition<CubicScale<3>, UniformRotationAboutZAxis<3>>;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
