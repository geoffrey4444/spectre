// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

struct TimeIndependentMetavariables {
  static constexpr bool enable_time_dependence{false};
};

struct TimeDependentMetavariables {
  static constexpr bool enable_time_dependence{true};
};

namespace {
template <typename... FuncsOfTime>
void test_binary_compact_object_construction(
    const domain::creators::BinaryCompactObject& binary_compact_object,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {},
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {}) {
  const auto domain = binary_compact_object.create_domain();
  test_initial_domain(domain,
                      binary_compact_object.initial_refinement_levels());
  test_physical_separation(binary_compact_object.create_domain().blocks(), time,
                           functions_of_time);

  TestHelpers::domain::creators::test_functions_of_time(
      binary_compact_object, expected_functions_of_time);
}

void test_connectivity() {
  // ObjectA:
  constexpr double inner_radius_objectA = 0.5;
  constexpr double outer_radius_objectA = 1.0;
  constexpr double xcoord_objectA = -3.0;

  // ObjectB:
  constexpr double inner_radius_objectB = 0.3;
  constexpr double outer_radius_objectB = 1.0;
  constexpr double xcoord_objectB = 3.0;

  // Enveloping Cube:
  constexpr double radius_enveloping_cube = 25.5;
  constexpr double radius_enveloping_sphere = 32.4;

  // Misc.:
  constexpr size_t refinement = 1;
  constexpr size_t grid_points = 3;
  constexpr bool use_projective_map = true;

  // Options for outer sphere
  constexpr size_t addition_to_outer_layer_radial_refinement_level = 3;

  for (const bool excise_interiorA : {true, false}) {
    for (const bool excise_interiorB : {true, false}) {
      for (const bool use_equiangular_map : {true, false}) {
        for (const bool use_logarithmic_map_outer_spherical_shell :
             {true, false}) {
          const domain::creators::BinaryCompactObject binary_compact_object{
              inner_radius_objectA,
              outer_radius_objectA,
              xcoord_objectA,
              excise_interiorA,
              inner_radius_objectB,
              outer_radius_objectB,
              xcoord_objectB,
              excise_interiorB,
              radius_enveloping_cube,
              radius_enveloping_sphere,
              refinement,
              grid_points,
              use_equiangular_map,
              use_projective_map,
              use_logarithmic_map_outer_spherical_shell,
              addition_to_outer_layer_radial_refinement_level,
          };
          test_binary_compact_object_construction(binary_compact_object);

          // Also check whether the radius of the inner boundary of Layer 5 is
          // chosen correctly.
          // Compute the radius of a point in the grid frame on this boundary.
          // Block 44 is one block whose -zeta face is on this boundary.
          const auto map{binary_compact_object.create_domain()
                             .blocks()[44]
                             .stationary_map()
                             .get_clone()};
          tnsr::I<double, 3, Frame::Logical> logical_point(
              std::array<double, 3>{{0.0, 0.0, -1.0}});
          const double layer_5_inner_radius = get(
              magnitude(std::move(map)->operator()(logical_point)));
          // The number of radial divisions in layers 4 and 5, excluding those
          // resulting from InitialRefinement > 0.
          const auto radial_divisions_in_outer_layers = static_cast<double>(
              pow(2, addition_to_outer_layer_radial_refinement_level) + 1);
          if (use_logarithmic_map_outer_spherical_shell) {
            CHECK(layer_5_inner_radius / radius_enveloping_cube ==
                  approx(pow(radius_enveloping_sphere / radius_enveloping_cube,
                             1.0 / radial_divisions_in_outer_layers)));
          } else {
            CHECK(layer_5_inner_radius - radius_enveloping_cube ==
                  approx((radius_enveloping_sphere - radius_enveloping_cube) /
                         radial_divisions_in_outer_layers));
          }
        }
      }
  }
}
}
void test_bbh_time_dependent_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeDependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n"
      "    InitialTime: 0.0 \n"
      "    InitialExpirationDeltaT: Auto \n"
      "    ExpansionMap: \n"
      "      ExpansionMapOuterBoundary: 25.0 \n"
      "      InitialExpansion: [1.0, 1.0] \n"
      "      InitialExpansionVelocity: [-0.1, -0.1] \n"
      "      InitialExpansionAcceleration: [-0.01, -0.01] \n"
      "      ExpansionFunctionOfTimeNames: ['ExpansionFactor', "
      " 'ExpansionFactor'] \n"
      "    SizeMap: \n"
      "      InitialSizeMapValues: [0.0, 0.0]\n"
      "      InitialSizeMapVelocities: [-0.1, -0.1]\n"
      "      InitialSizeMapAccelerations: [-0.01, -0.01]\n"
      "      SizeMapFunctionOfTimeNames: ['LambdaFactorA0', "
      " 'LambdaFactorB0']\n");
  const std::array<double, 4> times_to_check{{0.0, 4.4, 7.8}};

  std::array<DataVector, 4> expansion_factor_coefs{
      {{1.0}, {-0.1}, {-0.01}, {0.0}}};
  std::array<DataVector, 4> size_map_coefs{{{0.0}, {-0.1}, {-0.01}, {0.0}}};
  const std::tuple<
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>>
      expected_functions_of_time = std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "ExpansionFactor"s,
              {0.0, expansion_factor_coefs,
               std::numeric_limits<double>::max()}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "LambdaFactorA0"s,
              {0.0, size_map_coefs, std::numeric_limits<double>::max()}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "LambdaFactorB0"s,
              {0.0, size_map_coefs, std::numeric_limits<double>::max()}});

  for (const double time : times_to_check) {
    test_binary_compact_object_construction(
        dynamic_cast<const domain::creators::BinaryCompactObject&>(
            *binary_compact_object),
        time, binary_compact_object->functions_of_time(),
        expected_functions_of_time);
  }
}  // namespace

void test_bbh_equiangular_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_2_outer_radial_refinements_linear_map_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 2\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 2\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_3_outer_radial_refinements_log_map_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 3\n"
      "    UseLogarithmicMapObjectA: true\n"
      "    AdditionToObjectARadialRefinementLevel: 3\n"
      "    UseLogarithmicMapObjectB: true\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_equidistant_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: false\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bns_equiangular_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: false\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: false\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bns_equidistant_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: false\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: false\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: false\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bhns_equiangular_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: false\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bhns_equidistant_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: true\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: false\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: false\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_nsbh_equiangular_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: false\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: true\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_nsbh_equidistant_factory() {
  const auto binary_compact_object = TestHelpers::test_factory_creation<
      DomainCreator<3>,
      TestHelpers::TestCreationOpt<std::unique_ptr<DomainCreator<3>>>,
      TimeIndependentMetavariables>(
      "  BinaryCompactObject:\n"
      "    InnerRadiusObjectA: 0.2\n"
      "    OuterRadiusObjectA: 1.0\n"
      "    XCoordObjectA: -2.0\n"
      "    ExciseInteriorA: false\n"
      "    InnerRadiusObjectB: 1.0\n"
      "    OuterRadiusObjectB: 2.0\n"
      "    XCoordObjectB: 3.0\n"
      "    ExciseInteriorB: true\n"
      "    RadiusOuterCube: 22.0\n"
      "    RadiusOuterSphere: 25.0\n"
      "    InitialRefinement: 1\n"
      "    InitialGridPoints: 3\n"
      "    UseEquiangularMap: false\n"
      "    UseProjectiveMap: true\n"
      "    UseLogarithmicMapOuterSphericalShell: false\n"
      "    AdditionToOuterLayerRadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectA: false\n"
      "    AdditionToObjectARadialRefinementLevel: 0\n"
      "    UseLogarithmicMapObjectB: false\n"
      "    AdditionToObjectBRadialRefinementLevel: 0\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}
}  // namespace

// [[Timeout, 6]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.FactoryTests",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory();
  test_bbh_2_outer_radial_refinements_linear_map_factory();
  test_bbh_3_outer_radial_refinements_log_map_factory();
  test_bbh_time_dependent_factory();
  test_bbh_equiangular_factory();
  test_bbh_equidistant_factory();
  test_bns_equiangular_factory();
  test_bns_equidistant_factory();
  test_bhns_equiangular_factory();
  test_bhns_equidistant_factory();
  test_nsbh_equiangular_factory();
  test_nsbh_equidistant_factory();
}

// [[OutputRegex, The radius for the enveloping cube is too small! The Frustums
// will be malformed.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options1",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 0.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -7.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 0.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 8.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
}
// [[OutputRegex, ObjectA's inner radius must be less than its outer radius.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options2",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 1.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -1.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 0.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 1.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
}
// [[OutputRegex, ObjectB's inner radius must be less than its outer radius.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options3",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 0.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -1.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 3.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 1.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
}
