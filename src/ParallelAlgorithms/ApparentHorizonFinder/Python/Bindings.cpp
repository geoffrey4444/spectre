// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Strahlkorper/ChangeCenterOfStrahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/PostprocessRescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/RescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/SampleRescaledSurfaceCharSpeeds.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.SphericalHarmonics");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Domain");
  py::module_::import("spectre.Spectral");
  py::module_::import("spectre.Strahlkorper");
  py::enum_<FastFlow::FlowType>(m, "FlowType")
      .value("Jacobi", FastFlow::FlowType::Jacobi)
      .value("Curvature", FastFlow::FlowType::Curvature)
      .value("Fast", FastFlow::FlowType::Fast);
  py::enum_<FastFlow::Status>(m, "Status")
      .value("SuccessfulIteration", FastFlow::Status::SuccessfulIteration)
      .value("AbsTol", FastFlow::Status::AbsTol)
      .value("TruncationTol", FastFlow::Status::TruncationTol)
      .value("MaxIts", FastFlow::Status::MaxIts)
      .value("NegativeRadius", FastFlow::Status::NegativeRadius)
      .value("DivergenceError", FastFlow::Status::DivergenceError)
      .value("InterpolationFailure", FastFlow::Status::InterpolationFailure);
  py::class_<FastFlow::IterInfo>(m, "IterInfo")
      .def_readonly("iteration", &FastFlow::IterInfo::iteration)
      .def_readonly("r_min", &FastFlow::IterInfo::r_min)
      .def_readonly("r_max", &FastFlow::IterInfo::r_max)
      .def_readonly("min_residual", &FastFlow::IterInfo::min_residual)
      .def_readonly("max_residual", &FastFlow::IterInfo::max_residual)
      .def_readonly("residual_ylm", &FastFlow::IterInfo::residual_ylm)
      .def_readonly("residual_mesh", &FastFlow::IterInfo::residual_mesh);
  py::class_<FastFlow>(m, "FastFlow")
      .def(py::init<FastFlow::FlowType, double, double, double, double, double,
                    size_t, size_t>(),
           py::arg("flow_type"), py::arg("alpha"), py::arg("beta"),
           py::arg("abs_tol"), py::arg("truncation_tol"),
           py::arg("divergence_tol"), py::arg("divergence_iter"),
           py::arg("max_its"))
      .def(
          "iterate_horizon_finder",
          [](FastFlow& fast_flow,
             ylm::Strahlkorper<Frame::Inertial>& current_strahlkorper,
             const tnsr::II<DataVector, 3>& upper_spatial_metric,
             const tnsr::ii<DataVector, 3>& extrinsic_curvature,
             const tnsr::Ijj<DataVector, 3>& christoffel_2nd_kind) {
            return fast_flow.iterate_horizon_finder<Frame::Inertial>(
                make_not_null(&current_strahlkorper), upper_spatial_metric,
                extrinsic_curvature, christoffel_2nd_kind);
          },
          py::arg("current_strahlkorper"), py::arg("upper_spatial_metric"),
          py::arg("extrinsic_curvature"), py::arg("christoffel_2nd_kind"))
      .def("current_l_mesh", &FastFlow::current_l_mesh<Frame::Inertial>)
      .def("reset_for_next_find", &FastFlow::reset_for_next_find);
  m.def("rescaled_surface_factors", &ah::rescaled_surface_factors,
        py::arg("horizon_radius"), py::arg("excision_radius"),
        py::arg("number_of_surfaces") = 10, py::arg("relative_margin") = 1.e-7);
  m.def("rescaled_surface_char_speed_extrema",
        &ah::rescaled_surface_char_speed_extrema, py::arg("surface"),
        py::arg("time_deriv_surface"), py::arg("lapse"), py::arg("shift"),
        py::arg("inverse_spatial_metric"));
  m.def(
      "prepare_horizon_for_char_speeds",
      [](ylm::Strahlkorper<Frame::Distorted> horizon, const Domain<3>& domain,
         const std::string& excision_sphere) {
        const auto sphere = domain.excision_spheres().find(excision_sphere);
        if (sphere == domain.excision_spheres().end()) {
          throw std::invalid_argument("Unknown excision sphere '" +
                                      excision_sphere + "'.");
        }
        const auto& center = sphere->second.center();
        const std::array new_center{get<0>(center), get<1>(center),
                                    get<2>(center)};
        if (horizon.expansion_center() != new_center) {
          ylm::change_expansion_center_of_strahlkorper(make_not_null(&horizon),
                                                       new_center);
        }
        return horizon;
      },
      py::arg("horizon"), py::arg("domain"), py::arg("excision_sphere"));
  m.def(
      "sample_rescaled_surface_char_speeds",
      [](const ylm::Strahlkorper<Frame::Distorted>& horizon,
         const std::optional<ylm::Strahlkorper<Frame::Distorted>>&
             time_deriv_horizon,
         const std::vector<ElementId<3>>& element_ids,
         const std::vector<Mesh<3>>& meshes,
         const std::vector<tnsr::aa<DataVector, 3>>& spacetime_metrics,
         const Domain<3>& domain,
         const std::unordered_map<
             std::string, const domain::FunctionsOfTime::FunctionOfTime&>&
             functions_of_time,
         const double time, const std::string& excision_sphere,
         const size_t number_of_surfaces, const double relative_excision_margin,
         const std::optional<std::vector<std::string>>& blocks) {
        domain::FunctionsOfTimeMap cloned_functions{};
        for (const auto& [name, function] : functions_of_time) {
          cloned_functions[name] = function.get_clone();
        }
        return std::make_pair(
            ah::rescaled_surface_char_speed_legend(number_of_surfaces),
            ah::postprocess_rescaled_surface_char_speeds(
                horizon, time_deriv_horizon, element_ids, meshes,
                spacetime_metrics, domain, cloned_functions, time,
                ah::RescaledSurfaceCharSpeedOptions{excision_sphere,
                                                    number_of_surfaces,
                                                    relative_excision_margin},
                blocks));
      },
      py::arg("horizon"), py::arg("time_deriv_horizon"), py::arg("element_ids"),
      py::arg("meshes"), py::arg("spacetime_metrics"), py::arg("domain"),
      py::arg("functions_of_time"), py::arg("time"), py::arg("excision_sphere"),
      py::arg("number_of_surfaces") = 10,
      py::arg("relative_excision_margin") = 1.e-7,
      py::arg("blocks") = std::nullopt);
}
