// Distributed under the MIT License.
// See LICENSE.txt for details.
#include "CubicScaleAndUniformRotationAboutZAxis.hpp"
#include "Domain/Creators/TimeDependence/Composition.tpp"

namespace domain {
namespace creators {
namespace time_dependence {
template class Composition<
    TimeDependenceCompositionTag<CubicScale<1>>,
    TimeDependenceCompositionTag<UniformRotationAboutZAxis<1>>>;
template class Composition<
    TimeDependenceCompositionTag<CubicScale<2>>,
    TimeDependenceCompositionTag<UniformRotationAboutZAxis<2>>>;
template class Composition<
    TimeDependenceCompositionTag<CubicScale<3>>,
    TimeDependenceCompositionTag<UniformRotationAboutZAxis<3>>>;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
