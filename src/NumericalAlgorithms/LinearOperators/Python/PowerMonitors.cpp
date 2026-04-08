// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Python/PowerMonitors.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace PowerMonitors::py_bindings {

namespace {
template <typename T>
py::dict shell_power_monitor_buffer_to_dict(const T& tensor,
                                            const Mesh<3>& mesh) {
  const auto buffer = shell_power_monitor_buffer(tensor, mesh);
  py::dict result{};
  result["radial_sums"] = buffer.radial_sums;
  result["angular_sums"] = buffer.angular_sums;
  result["radial_counts"] = buffer.radial_counts;
  result["angular_counts"] = buffer.angular_counts;
  return result;
}

template <typename T>
py::dict shell_power_monitors_to_dict(const T& tensor, const Mesh<3>& mesh) {
  const auto monitors = shell_power_monitors(tensor, mesh);
  py::dict result{};
  result["radial"] = monitors.radial;
  result["angular"] = monitors.angular;
  return result;
}

template <size_t Dim>
void bind_power_monitors_impl(py::module& m) {  // NOLINT
  m.def("power_monitors",
        py::overload_cast<const DataVector&, const Mesh<Dim>&>(
            &power_monitors<DataVector, Dim>),
        py::arg("data_vector"), py::arg("mesh"));
  m.def("relative_truncation_error",
        py::overload_cast<const DataVector&, const Mesh<Dim>&>(
            &relative_truncation_error<DataVector, Dim>),
        py::arg("tensor_component"), py::arg("mesh"));
  m.def("absolute_truncation_error",
        py::overload_cast<const DataVector&, const Mesh<Dim>&>(
            &absolute_truncation_error<DataVector, Dim>),
        py::arg("tensor_component"), py::arg("mesh"));
}
}  // namespace

void bind_power_monitors(py::module& m) {
  bind_power_monitors_impl<1>(m);
  bind_power_monitors_impl<2>(m);
  bind_power_monitors_impl<3>(m);
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<DataVector>, py::arg("data_vector"),
        py::arg("mesh"));
  m.def("shell_power_monitors", &shell_power_monitors_to_dict<DataVector>,
        py::arg("data_vector"), py::arg("mesh"));
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<Scalar<DataVector>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitors",
        &shell_power_monitors_to_dict<Scalar<DataVector>>, py::arg("tensor"),
        py::arg("mesh"));
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<tnsr::i<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitors",
        &shell_power_monitors_to_dict<tnsr::i<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<tnsr::ii<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitors",
        &shell_power_monitors_to_dict<tnsr::ii<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<tnsr::ij<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitors",
        &shell_power_monitors_to_dict<tnsr::ij<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitor_buffer",
        &shell_power_monitor_buffer_to_dict<tnsr::ijj<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def("shell_power_monitors",
        &shell_power_monitors_to_dict<tnsr::ijj<DataVector, 3>>,
        py::arg("tensor"), py::arg("mesh"));
  m.def(
      "convergence_rate_and_number_of_pile_up_modes",
      [](const DataVector& power_monitor,
         const size_t number_of_filtered_modes) {
        py::dict result{};
        const ConvergenceInfo info =
            convergence_rate_and_number_of_pile_up_modes(
                power_monitor, number_of_filtered_modes);
        result["convergence_rate"] = info.convergence_rate;
        result["number_of_pile_up_modes"] = info.number_of_pile_up_modes;
        return result;
      },
      py::arg("power_monitor"), py::arg("number_of_filtered_modes") = 0);
}

}  // namespace PowerMonitors::py_bindings
