// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Basis.hpp"

template <size_t Dim>
class Domain;

namespace evolution::dg {
/// The interval basis used for both element placement and DG initialization.
template <typename Metavariables, size_t Dim>
Spectral::Basis initial_i1_basis(const Domain<Dim>& domain) {
  if constexpr (requires { Metavariables::initial_i1_basis(domain); }) {
    return Metavariables::initial_i1_basis(domain);
  } else {
    return Spectral::Basis::Legendre;
  }
}
}  // namespace evolution::dg
