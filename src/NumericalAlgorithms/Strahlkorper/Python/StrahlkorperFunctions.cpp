// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Strahlkorper/Python/StrahlkorperFunctions.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Strahlkorper/ChangeCenterOfStrahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
namespace {
template <typename Frame>
void bind_strahlkorper_functions_impl(pybind11::module& m) {  // NOLINT
  using Strahlkorper = ylm::Strahlkorper<Frame>;
  m.def("cartesian_coords",
        py::overload_cast<const Strahlkorper&>(&ylm::cartesian_coords<Frame>),
        py::arg("strahlkorper"));
  m.def("power_monitor",
        py::overload_cast<const Strahlkorper&>(&ylm::power_monitor<Frame>),
        py::arg("strahlkorper"));
  m.def(
      "change_expansion_center_of_strahlkorper",
      [](Strahlkorper strahlkorper, const std::array<double, 3>& center) {
        ylm::change_expansion_center_of_strahlkorper(
            make_not_null(&strahlkorper), center);
        return strahlkorper;
      },
      py::arg("strahlkorper"), py::arg("center"),
      "Return a copy of the surface expanded about the requested center.");
  m.def(
      "time_deriv_of_strahlkorper",
      [](const std::deque<std::pair<double, Strahlkorper>>& history) {
        if (history.empty() or history.size() > 4) {
          throw py::value_error("Expected one to four surfaces in history.");
        }
        for (size_t i = 0; i < history.size(); ++i) {
          if (not std::isfinite(history[i].first) or
              (i > 0 and history[i].first >= history[i - 1].first)) {
            throw py::value_error(
                "Surface history must have finite times, newest first, "
                "with no repeated times.");
          }
          if (history[i].second.expansion_center() !=
              history.front().second.expansion_center()) {
            throw py::value_error(
                "Surface history must have a fixed expansion center.");
          }
        }
        auto result = history.front().second;
        // The C++ helper handles changes in l_max, assuming m_max == l_max.
        // Normalize the history to that convention and let it restrict the
        // derivative back to the newest surface's original (l_max, m_max).
        auto full_history = history;
        for (auto& [time, surface] : full_history) {
          if (surface.m_max() != surface.l_max()) {
            surface = Strahlkorper{surface.l_max(), surface.l_max(), surface};
          }
        }
        ylm::time_deriv_of_strahlkorper(make_not_null(&result), full_history);
        return result;
      },
      py::arg("history"),
      "Differentiate one to four (time, surface) pairs, newest first. "
      "Uses the newest surface's resolution and a fixed expansion center. "
      "A single surface returns zero, following the C++ helper convention; "
      "it does not establish that the surface is stationary.");
}
}  // namespace

void bind_strahlkorper_functions(pybind11::module& m) {  // NOLINT
  bind_strahlkorper_functions_impl<Frame::Grid>(m);
  bind_strahlkorper_functions_impl<Frame::Inertial>(m);
  bind_strahlkorper_functions_impl<Frame::Distorted>(m);
}
}  // namespace ylm::py_bindings
