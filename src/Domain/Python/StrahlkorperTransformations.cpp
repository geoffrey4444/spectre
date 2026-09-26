// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/StrahlkorperTransformations.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"

namespace py = pybind11;

namespace domain::py_bindings {
namespace {

using PyFunctionsOfTimeMap =
    std::unordered_map<std::string,
                       const domain::FunctionsOfTime::FunctionOfTime&>;

// Transform functions-of-time map to unique_ptrs because pybind11
// can't handle them easily as function arguments (it's hard to
// transfer ownership of a Python object to C++)
domain::FunctionsOfTimeMap transform_functions_of_time(
    const std::optional<PyFunctionsOfTimeMap>& functions_of_time) {
  domain::FunctionsOfTimeMap functions_of_time_ptrs{};
  if (functions_of_time.has_value()) {
    for (const auto& [name, fot] : *functions_of_time) {
      functions_of_time_ptrs[name] = fot.get_clone();
    }
  }
  return functions_of_time_ptrs;
}

template <typename SrcFrame, typename DestFrame>
void bind_strahlkorper_transformations_impl(py::module& m) {  // NOLINT
  m.def(
      "strahlkorper_in_inertial_frame",
      [](const ylm::Strahlkorper<SrcFrame>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<PyFunctionsOfTimeMap>& functions_of_time,
         const std::optional<double>& time) {
        ylm::Strahlkorper<DestFrame> result{};
        strahlkorper_in_different_frame<SrcFrame, DestFrame>(
            make_not_null(&result), strahlkorper, domain,
            transform_functions_of_time(functions_of_time),
            time.value_or(std::numeric_limits<double>::signaling_NaN()));
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt);
  m.def(
      "strahlkorper_in_inertial_frame_aligned",
      [](const ylm::Strahlkorper<SrcFrame>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<PyFunctionsOfTimeMap>& functions_of_time,
         const std::optional<double>& time) {
        ylm::Strahlkorper<DestFrame> result{};
        strahlkorper_in_different_frame_aligned<SrcFrame, DestFrame>(
            make_not_null(&result), strahlkorper, domain,
            transform_functions_of_time(functions_of_time),
            time.value_or(std::numeric_limits<double>::signaling_NaN()));
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt);
}
}  // namespace

void bind_strahlkorper_transformations(py::module& m) {  // NOLINT
  // Only instantiating for Grid->Inertial because the Py functions are
  // currently named like that
  bind_strahlkorper_transformations_impl<Frame::Grid, Frame::Inertial>(m);
  m.def(
      "strahlkorper_in_distorted_frame",
      [](const ylm::Strahlkorper<Frame::Inertial>& strahlkorper,
         const Domain<3>& domain,
         const std::optional<PyFunctionsOfTimeMap>& functions_of_time,
         const std::optional<double>& time) {
        // Stationary domains have coincident coordinate frames and no
        // time-dependent maps to pass to the general transformation helper.
        if (not domain.is_time_dependent()) {
          return ylm::Strahlkorper<Frame::Distorted>{strahlkorper};
        }
        if (not time.has_value() or not std::isfinite(*time)) {
          throw py::value_error(
              "A finite time is required for a time-dependent domain.");
        }
        const auto functions_of_time_ptrs =
            transform_functions_of_time(functions_of_time);
        const auto check_functions_of_time = [&functions_of_time_ptrs,
                                              &time](const auto& map) {
          for (const auto& name : map.function_of_time_names()) {
            const auto fot = functions_of_time_ptrs.find(name);
            if (fot == functions_of_time_ptrs.end()) {
              throw py::value_error("Function of time '" + name +
                                    "' is missing.");
            }
            const auto bounds = fot->second->time_bounds();
            if (not(bounds[0] <= *time and *time <= bounds[1])) {
              throw py::value_error("Function of time '" + name +
                                    "' does not cover the requested time "
                                    "(outside its time bounds).");
            }
          }
        };
        // The block search may evaluate any moving block's inverse map.
        // Validate all required functions before entering the map routines.
        for (const auto& block : domain.blocks()) {
          if (block.is_time_dependent()) {
            check_functions_of_time(block.moving_mesh_grid_to_inertial_map());
            if (block.has_distorted_frame()) {
              check_functions_of_time(
                  block.moving_mesh_grid_to_distorted_map());
              check_functions_of_time(
                  block.moving_mesh_distorted_to_inertial_map());
            }
          }
        }
        const auto block_coords = block_logical_coordinates(
            domain, ylm::cartesian_coords(strahlkorper), *time,
            functions_of_time_ptrs);
        for (const auto& block_coord : block_coords) {
          if (not block_coord.has_value()) {
            throw py::value_error("Surface lies outside the domain.");
          }
          if (not domain.blocks()[block_coord->id.get_index()]
                      .has_distorted_frame()) {
            throw py::value_error(
                "Surface lies outside the distorted frame region.");
          }
        }
        ylm::Strahlkorper<Frame::Distorted> result{};
        strahlkorper_in_different_frame<Frame::Inertial, Frame::Distorted>(
            make_not_null(&result), strahlkorper, domain,
            functions_of_time_ptrs, *time);
        return result;
      },
      py::arg("strahlkorper"), py::arg("domain"),
      py::arg("functions_of_time") = std::nullopt,
      py::arg("time") = std::nullopt,
      "Transform an Inertial surface to the Distorted frame. Stationary "
      "domains have coincident frames. For time-dependent domains every "
      "surface point must lie in a block with a Distorted frame.");
}

}  // namespace domain::py_bindings
