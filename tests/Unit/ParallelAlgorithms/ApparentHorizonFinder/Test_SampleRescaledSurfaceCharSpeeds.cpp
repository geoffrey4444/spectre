// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/SampleRescaledSurfaceCharSpeeds.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void test_sampling(const bool time_dependent) {
  CAPTURE(time_dependent);
  using Fr = Frame::Distorted;
  const domain::creators::Sphere creator{
      1.6,
      2.4,
      domain::creators::Sphere::Excision{nullptr},
      0_st,
      4_st,
      true,
      std::nullopt,
      std::vector<double>{1.8}};
  auto domain = creator.create_domain();
  std::unordered_set<std::string> blocks{};
  for (size_t id = 0; id < domain.blocks().size(); ++id) {
    if (time_dependent) {
      const domain::CoordinateMaps::Identity<3> identity{};
      domain.inject_time_dependent_map_for_block(
          id,
          domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              identity),
          domain::make_coordinate_map_base<Frame::Grid, Fr>(identity),
          domain::make_coordinate_map_base<Fr, Frame::Inertial>(identity));
    }
    blocks.insert(domain.blocks()[id].name());
  }
  const ylm::Strahlkorper<Fr> horizon{4_st, 2.0, std::array{0.0, 0.0, 0.0}};
  auto dt_horizon = horizon;
  dt_horizon.coefficients() = 0.0;
  const ah::RescaledSurfaceCharSpeedOptions options{"ExcisionSphere", 3, 1.e-7};
  ah::Storage::RescaledSurfaceCharSpeeds data{};
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              {}, 1.0);
  REQUIRE(data.radius_factors.size() == 3);
  CHECK_ITERABLE_APPROX(data.radius_factors,
                        (std::vector<double>{1.0, 0.95000002, 0.80000008}));
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::Valid);

  std::unordered_map<ElementId<3>, ah::Storage::VolumeVariables<Fr>> volumes{};
  CHECK_FALSE(ah::sample_rescaled_surface_char_speeds(
      make_not_null(&data), volumes, domain, {}, 1.0, blocks));

  const auto add_volume = [&volumes](const size_t block_id) {
    auto& volume = volumes[ElementId<3>{block_id}];
    volume.mesh = Mesh<3>{4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const auto size = volume.mesh.number_of_grid_points();
    volume.vars_to_interpolate_to_target.initialize(size, 0.0);
    auto& inverse_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>>(
            volume.vars_to_interpolate_to_target);
    get<0, 0>(inverse_metric) = 1.0;
    get<1, 1>(inverse_metric) = 1.0;
    get<2, 2>(inverse_metric) = 1.0;
    volume.rescaled_surface_vars.emplace(size, 0.0);
    get(get<gr::Tags::Lapse<DataVector>>(*volume.rescaled_surface_vars)) = 1.0;
  };

  // The horizon data arrive before the inner shell needed by the last surface.
  for (size_t id = 6; id < domain.blocks().size(); ++id) {
    add_volume(id);
  }
  CHECK_FALSE(ah::sample_rescaled_surface_char_speeds(
      make_not_null(&data), volumes, domain, {}, 1.0, blocks));
  CHECK(data.next_surface == 2);
  data = serialize_and_deserialize(data);
  CHECK(data.next_surface == 2);
  CHECK(data.horizon == horizon);
  for (size_t id = 0; id < 6; ++id) {
    add_volume(id);
  }
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), volumes,
                                                domain, {}, 1.0, blocks));
  CHECK(data.next_surface == 3);
  CHECK_ITERABLE_APPROX(data.min_speeds,
                        (std::vector<double>{-1.0, -1.0, -1.0}));
  CHECK_ITERABLE_APPROX(data.max_speeds, data.min_speeds);
  CHECK(data.horizon == horizon);

  CHECK(ah::rescaled_surface_char_speed_legend(3) ==
        (std::vector<std::string>{"Time", "Status", "RadiusFactor_0",
                                  "MinCharSpeed_0", "MaxCharSpeed_0",
                                  "RadiusFactor_1", "MinCharSpeed_1",
                                  "MaxCharSpeed_1", "RadiusFactor_2",
                                  "MinCharSpeed_2", "MaxCharSpeed_2"}));
  CHECK_ITERABLE_APPROX(
      ah::rescaled_surface_char_speed_row(1.0, data),
      (std::vector<double>{1.0, 0.0, 1.0, -1.0, -1.0, 0.95000002, -1.0, -1.0,
                           0.80000008, -1.0, -1.0}));

  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, false, options,
                                              domain, {}, 1.0);
  CHECK(data.status ==
        ah::Storage::RescaledSurfaceStatus::MissingTimeDerivative);
  CHECK_ITERABLE_APPROX(data.radius_factors,
                        (std::vector<double>{1.0, 0.95000002, 0.80000008}));
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), volumes,
                                                domain, {}, 1.0, blocks));
  REQUIRE(data.min_speeds.size() == 3);
  CHECK(std::isnan(data.min_speeds[0]));

  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              {}, 1.0);
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), volumes,
                                                domain, {}, 1.0, {}));
  CHECK(data.status ==
        ah::Storage::RescaledSurfaceStatus::MissingBlockCoverage);

  // Validate every surface before waiting, including a missing inner shell.
  std::unordered_set<std::string> outer_blocks{};
  for (size_t id = 6; id < domain.blocks().size(); ++id) {
    outer_blocks.insert(domain.blocks()[id].name());
  }
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              {}, 1.0);
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), {},
                                                domain, {}, 1.0, outer_blocks));
  CHECK(data.status ==
        ah::Storage::RescaledSurfaceStatus::MissingBlockCoverage);

  const ylm::Strahlkorper<Fr> outside_horizon{4_st, 3.0,
                                              std::array{0.0, 0.0, 0.0}};
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data),
                                              outside_horizon, dt_horizon, true,
                                              options, domain, {}, 1.0);
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), {},
                                                domain, {}, 1.0, blocks));
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::OutsideDomain);

  for (auto& [element_id, volume] : volumes) {
    (void)element_id;
    get(get<gr::Tags::Lapse<DataVector>>(*volume.rescaled_surface_vars)) =
        std::numeric_limits<double>::quiet_NaN();
  }
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              {}, 1.0);
  CHECK(ah::sample_rescaled_surface_char_speeds(make_not_null(&data), volumes,
                                                domain, {}, 1.0, blocks));
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::NonfiniteSpeed);

  const ah::RescaledSurfaceCharSpeedOptions bad_options{"Absent", 3, 1.e-7};
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, bad_options,
                                              domain, {}, 1.0);
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::InvalidGeometry);

  const ylm::Strahlkorper<Fr> displaced_horizon{4_st, 2.0,
                                                std::array{0.1, 0.0, 0.0}};
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data),
                                              displaced_horizon, dt_horizon,
                                              true, options, domain, {}, 1.0);
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::InvalidGeometry);

  auto nonfinite_horizon = horizon;
  nonfinite_horizon.coefficients()[0] =
      std::numeric_limits<double>::quiet_NaN();
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data),
                                              nonfinite_horizon, dt_horizon,
                                              true, options, domain, {}, 1.0);
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::InvalidGeometry);

  // A radial map changes the excision radius used to choose the factors.
  using Scale = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
  using FunctionOfTime = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time["Scale"] = std::make_unique<FunctionOfTime>(
      0.0, std::array{DataVector{1.1}, DataVector{0.0}, DataVector{0.0}}, 2.0);
  domain = creator.create_domain();
  for (size_t id = 0; id < domain.blocks().size(); ++id) {
    const Scale scale{100.0, "Scale", "Scale"};
    domain.inject_time_dependent_map_for_block(
        id,
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(scale),
        domain::make_coordinate_map_base<Frame::Grid, Fr>(scale),
        domain::make_coordinate_map_base<Fr, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{}));
  }
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              functions_of_time, 1.0);
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::Valid);
  CHECK_ITERABLE_APPROX(data.radius_factors,
                        (std::vector<double>{1.0, 0.970000022, 0.880000088}));

  // A translation changes the angular rays and is outside the supported maps.
  using Translation = domain::CoordinateMaps::TimeDependent::Translation<3>;
  functions_of_time["Translation"] = std::make_unique<FunctionOfTime>(
      0.0,
      std::array{DataVector{0.1, 0.0, 0.0}, DataVector(3, 0.0),
                 DataVector(3, 0.0)},
      2.0);
  domain = creator.create_domain();
  for (size_t id = 0; id < domain.blocks().size(); ++id) {
    const Translation translation{"Translation"};
    domain.inject_time_dependent_map_for_block(
        id,
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            translation),
        domain::make_coordinate_map_base<Frame::Grid, Fr>(translation),
        domain::make_coordinate_map_base<Fr, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{}));
  }
  ah::initialize_rescaled_surface_char_speeds(make_not_null(&data), horizon,
                                              dt_horizon, true, options, domain,
                                              functions_of_time, 1.0);
  CHECK(data.status == ah::Storage::RescaledSurfaceStatus::InvalidGeometry);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.SampleRescaledSurfaceCharSpeeds",
                  "[Unit][ApparentHorizonFinder]") {
  test_sampling(false);
  test_sampling(true);
}
