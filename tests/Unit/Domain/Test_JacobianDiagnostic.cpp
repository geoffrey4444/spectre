// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/JacobianDiagnostic.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"

// IWYU pragma: no_include "Utilities/Array.hpp"

namespace {
template <size_t Dim>
ElementMap<Dim, Frame::Grid> jac_diag_map_that_fits();

template <>
ElementMap<1, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Affine map{-1.0, 1.0, 0.4, 5.5};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<2, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 2;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Rotation<2> map{M_PI_4};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<3, Frame::Grid> jac_diag_map_that_fits() {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Rotation<3> map{M_PI_4, M_PI_2, M_PI_2};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <size_t Dim>
ElementMap<Dim, Frame::Grid> jac_diag_generic_map();

template <>
ElementMap<1, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  const domain::CoordinateMaps::Equiangular map{-1.0, 1.0, 0.4, 5.5};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<2, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 2;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  using Equiangular = domain::CoordinateMaps::Equiangular;
  using Equiangular2D =
      domain::CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  const Equiangular2D map{{-1.0, 1.0, 0.4, 5.5}, {-1.0, 1.0, -4.4, -0.2}};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <>
ElementMap<3, Frame::Grid> jac_diag_generic_map() {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(0, 0)}});
  const ElementId<dim> element_id(0, segment_ids);
  using Equiangular = domain::CoordinateMaps::Equiangular;
  using Equiangular3D =
      domain::CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular,
                                             Equiangular>;
  const Equiangular3D map{
      {-1.0, 1.0, 0.4, 5.5}, {-1.0, 1.0, -4.4, -0.2}, {-1.0, 1.0, -5.0, 3.0}};
  return ElementMap<dim, Frame::Grid>{
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(map)};
}

template <size_t Dim, bool UseGenericMap>
void test_jacobian_diagnostic_databox() {
  if constexpr (UseGenericMap) {
    TestHelpers::db::test_compute_tag<domain::Tags::JacobianDiagnosticCompute<
        domain::Tags::ElementMap<Dim>,
        domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
        domain::Tags::MappedCoordinates<
            domain::Tags::ElementMap<Dim>,
            domain::Tags::Coordinates<Dim, Frame::ElementLogical>>>>(
        "JacobianDiagnostic");
  }

  auto setup_databox = [](const size_t points_per_dimension) {
    std::array<size_t, Dim> extents{};
    for (size_t i = 0; i < Dim; ++i) {
      extents[i] = points_per_dimension;
    }
    Mesh<Dim> mesh{extents, Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    auto logical_coords = logical_coordinates(mesh);
    auto map = UseGenericMap ? jac_diag_generic_map<Dim>()
                             : jac_diag_map_that_fits<Dim>();

    return db::create<
        tmpl::list<domain::Tags::ElementMap<Dim, Frame::Grid>,
                   domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                   domain::Tags::Mesh<Dim>>,
        db::AddComputeTags<
            domain::Tags::MappedCoordinates<
                domain::Tags::ElementMap<Dim, Frame::Grid>,
                domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
            domain::Tags::JacobianDiagnosticCompute<
                domain::Tags::ElementMap<Dim, Frame::Grid>,
                domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                domain::Tags::MappedCoordinates<
                    domain::Tags::ElementMap<Dim, Frame::Grid>,
                    domain::Tags::Coordinates<Dim, Frame::ElementLogical>>>>>(
        std::move(map), std::move(logical_coords), std::move(mesh));
  };

  const auto box = setup_databox(5);
  const DataVector jac_diag =
      get(db::get<domain::Tags::JacobianDiagnostic>(box));
  const DataVector expected_jac_diag =
      get(make_with_value<Scalar<DataVector>>(jac_diag, 0.0));

  if constexpr (UseGenericMap) {
    auto box_high = setup_databox(7);
    const DataVector jac_diag_high =
        get(db::get<domain::Tags::JacobianDiagnostic>(box_high));
    // Check that higher resolution leads to significantly smaller diagnostic
    CHECK(max(abs(jac_diag_high)) < 0.125 * max(abs(jac_diag)));
  } else {
    // Map fits in the resolution provided, so the diagnostic should be roundoff
    CHECK_ITERABLE_APPROX(jac_diag, expected_jac_diag);
  }
}

template <size_t Dim, typename Fr>
void test_jacobian_diagnostic_random() {
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          const gsl::not_null<Scalar<DataVector>*>,
          const Jacobian<DataVector, Dim, typename Frame::ElementLogical, Fr>&,
          const TensorMetafunctions::prepend_spatial_index<
              tnsr::I<DataVector, Dim, Fr>, Dim, UpLo::Lo,
              typename Frame::ElementLogical>&)>(
          &domain::jacobian_diagnostic<Dim, Fr>),
      "Test_JacobianDiagnostic", {"jacobian_diagnostic"}, {{{-1.0, 1.0}}},
      DataVector(5));
}
}  // namespace
SPECTRE_TEST_CASE("Unit.Domain.JacobianDiagnostic", "[Domain][Unit]") {
  TestHelpers::db::test_simple_tag<domain::Tags::JacobianDiagnostic>(
      "JacobianDiagnostic");

  pypp::SetupLocalPythonEnvironment local_python_env{"Domain/Python/"};

  test_jacobian_diagnostic_databox<1, false>();
  test_jacobian_diagnostic_databox<2, false>();
  test_jacobian_diagnostic_databox<3, false>();
  test_jacobian_diagnostic_databox<1, true>();
  test_jacobian_diagnostic_databox<2, true>();
  test_jacobian_diagnostic_databox<3, true>();

  test_jacobian_diagnostic_random<1, Frame::Grid>();
  test_jacobian_diagnostic_random<2, Frame::Grid>();
  test_jacobian_diagnostic_random<3, Frame::Grid>();
  test_jacobian_diagnostic_random<1, Frame::Inertial>();
  test_jacobian_diagnostic_random<2, Frame::Inertial>();
  test_jacobian_diagnostic_random<3, Frame::Inertial>();
}
