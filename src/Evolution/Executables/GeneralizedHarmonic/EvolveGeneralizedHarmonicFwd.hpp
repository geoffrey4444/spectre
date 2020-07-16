// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace GeneralizedHarmonic {
template <size_t Dim>
struct System;
namespace Solutions {
template <typename GrSolution>
struct WrappedGr;
}  // namespace Solutions
}  // namespace GeneralizedHarmonic
namespace gr {
namespace Solutions {
struct KerrSchild;
template <size_t VolumeDim>
struct Minkowski;
}  // namespace Solutions
}  // namespace gr
namespace evolution {
template <typename System>
struct NumericInitialData;
}  // namespace evolution

template <typename InitialData, typename BoundaryConditions>
struct EvolutionMetavars;
/// \endcond
