// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
void check_aa_orthogonality(
    const tnsr::aa<DataType, SpatialDim, Frame::Inertial>& projection_aa,
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const DataType& zero) {
  tnsr::a<DataType, SpatialDim, Frame::Inertial> right_contraction{
      get_size(get<0, 0>(projection_aa))};
  tnsr::a<DataType, SpatialDim, Frame::Inertial> left_contraction{
      get_size(get<0, 0>(projection_aa))};
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    right_contraction.get(a) = zero;
    left_contraction.get(a) = zero;
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      right_contraction.get(a) +=
          projection_aa.get(a, b) * spacetime_normal_vector.get(b);
      left_contraction.get(b) +=
          projection_aa.get(a, b) * spacetime_normal_vector.get(a);
    }
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    CHECK(max(abs(right_contraction.get(a) - zero)) < 1.e-12);
    CHECK(max(abs(left_contraction.get(a) - zero)) < 1.e-12);
  }
}

template <size_t SpatialDim, typename DataType>
void check_ab_orthogonality(
    const tnsr::Ab<DataType, SpatialDim, Frame::Inertial>& projection_ab,
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const DataType& zero) {
  tnsr::a<DataType, SpatialDim, Frame::Inertial> lower_contraction{
      get_size(get<0, 0>(projection_ab))};
  tnsr::A<DataType, SpatialDim, Frame::Inertial> upper_contraction{
      get_size(get<0, 0>(projection_ab))};
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    lower_contraction.get(a) = zero;
    upper_contraction.get(a) = zero;
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      lower_contraction.get(b) +=
          projection_ab.get(a, b) * spacetime_normal_one_form.get(a);
      upper_contraction.get(a) +=
          projection_ab.get(a, b) * spacetime_normal_vector.get(b);
    }
  }
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    CHECK(max(abs(lower_contraction.get(a) - zero)) < 1.e-12);
    CHECK(max(abs(upper_contraction.get(a) - zero)) < 1.e-12);
  }
}

template <size_t SpatialDim, typename DataType>
void test_projection_operator(const DataType& used_for_size) {
  {
    tnsr::II<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ij<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "transverse_projection_operator_mixed_from_spatial_input",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::AA<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::AA<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::aa<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::aa<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ab<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "projection_operator_transverse_to_interface_mixed", {{{-1., 1.}}},
        used_for_size);
  }

  const auto zero = make_with_value<DataType>(used_for_size, 0.);
  const auto data_size = get_size(used_for_size);

  const auto lapse = make_with_value<Scalar<DataType>>(used_for_size, 1.3);
  tnsr::I<DataType, SpatialDim, Frame::Inertial> shift{data_size};
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift.get(i) = make_with_value<DataType>(
        used_for_size, 0.1 * (static_cast<double>(i) + 1.0));
  }

  tnsr::ii<DataType, SpatialDim, Frame::Inertial> spatial_metric{data_size};
  for (size_t i = 0; i < SpatialDim; ++i) {
    spatial_metric.get(i, i) =
        make_with_value<DataType>(used_for_size, 2. + static_cast<double>(i));
    for (size_t j = i + 1; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) = make_with_value<DataType>(
          used_for_size,
          0.1 * (static_cast<double>(i) + static_cast<double>(j) + 1.0));
    }
  }

  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataType, SpatialDim, Frame::Inertial>(
          lapse);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);

  tnsr::i<DataType, SpatialDim, Frame::Inertial> interface_unit_normal_one_form(
      data_size);
  tnsr::I<DataType, SpatialDim, Frame::Inertial> interface_unit_normal_vector(
      data_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    interface_unit_normal_one_form.get(i) = make_with_value<DataType>(
        used_for_size, 0.2 * (static_cast<double>(i) + 1.0));
    interface_unit_normal_vector.get(i) = interface_unit_normal_one_form.get(i);
  }

  const auto projection_aa = gr::transverse_projection_operator(
      spacetime_metric, spacetime_normal_one_form,
      interface_unit_normal_one_form, shift);
  check_aa_orthogonality(projection_aa, spacetime_normal_vector, zero);

  tnsr::aa<DataType, SpatialDim, Frame::Inertial> projection_aa_not_null(
      data_size);
  gr::transverse_projection_operator(
      make_not_null(&projection_aa_not_null), spacetime_metric,
      spacetime_normal_one_form, interface_unit_normal_one_form, shift);
  check_aa_orthogonality(projection_aa_not_null, spacetime_normal_vector, zero);

  const auto projection_ab = gr::transverse_projection_operator(
      spacetime_normal_vector, spacetime_normal_one_form,
      interface_unit_normal_vector, interface_unit_normal_one_form, shift);
  check_ab_orthogonality(projection_ab, spacetime_normal_one_form,
                         spacetime_normal_vector, zero);

  tnsr::Ab<DataType, SpatialDim, Frame::Inertial> projection_ab_not_null(
      data_size);
  gr::transverse_projection_operator(
      make_not_null(&projection_ab_not_null), spacetime_normal_vector,
      spacetime_normal_one_form, interface_unit_normal_vector,
      interface_unit_normal_one_form, shift);
  check_ab_orthogonality(projection_ab_not_null, spacetime_normal_one_form,
                         spacetime_normal_vector, zero);
}
}  // namespace

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using frame = Frame::Inertial;
constexpr size_t SpatialDim = 3;

// Test projection operators by comparing to values from SpEC
void test_spatial_projection_tensors_3D(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const Direction<SpatialDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, SpatialDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        get<0>(tmp)[i * SpatialDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * SpatialDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();

  // 1. Projection IJ
  auto local_inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_vector =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      local_inverse_spatial_metric.get(0, j)[i] = 41.;
      local_inverse_spatial_metric.get(1, j)[i] = 43.;
      local_inverse_spatial_metric.get(2, j)[i] = 47.;
    }
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_vector) = -1.;
  get<1>(local_unit_interface_normal_vector) = 1.;
  get<2>(local_unit_interface_normal_vector) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_IJ), local_inverse_spatial_metric,
      local_unit_interface_normal_vector);

  // Initialize with values from SpEC
  auto spec_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {40., 42., 42., 42., 42., 42., 42., 42., 46.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_IJ.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_IJ,
                        spec_spatial_projection_IJ);

  // 2. Projection ij
  auto local_spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_one_form =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < SpatialDim; ++i) {
    local_spatial_metric.get(0, i) = 263.;
    local_spatial_metric.get(1, i) = 269.;
    local_spatial_metric.get(2, i) = 271.;
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_one_form) = -1.;
  get<1>(local_unit_interface_normal_one_form) = 1.;
  get<2>(local_unit_interface_normal_one_form) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_ij), local_spatial_metric,
      local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {262., 264., 264., 264., 268., 268., 264., 268., 270.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_ij,
                        spec_spatial_projection_ij);

  // 3. Projection Ij
  auto local_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_Ij),
      local_unit_interface_normal_vector, local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {0., 1., 1., 1., 0., -1., 1., -1., 0.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        spec_spatial_projection_Ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_APPROX(local_spatial_projection_Ij,
                        spec_spatial_projection_Ij);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.ProjectionOps",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_projection_operator, (1, 2, 3));

  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_spatial_projection_tensors_3D(grid_size, lower_bound, upper_bound);
}
