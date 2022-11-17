// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/Spectral/Python/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Python/Spectral.hpp"

PYBIND11_MODULE(_PySpectral, m) {  // NOLINT
  Spectral::py_bindings::bind_basis(m);
  Spectral::py_bindings::bind_quadrature(m);
  Spectral::py_bindings::bind_nodal_to_modal_matrix(m);
  Spectral::py_bindings::bind_modal_to_nodal_matrix(m);
  Spectral::py_bindings::bind_collocation_points(m);
  py_bindings::bind_mesh(m);
}
