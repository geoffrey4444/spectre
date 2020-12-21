// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TimeDependentTripleGaussian.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_triple_gaussian_random(const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      GeneralizedHarmonic::ConstraintDamping::TimeDependentTripleGaussian<
          VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double constant = real_dis(gen);

  const double amplitude_1{positive_dis(gen)};
  const double amplitude_2{positive_dis(gen)};
  const double amplitude_3{positive_dis(gen)};

  const double width_1{positive_dis(gen) + 0.5};
  const double width_2{positive_dis(gen) + 0.5};
  const double width_3{positive_dis(gen) + 0.5};

  // Generate the center
  std::array<double, VolumeDim> center_1{};
  std::array<double, VolumeDim> center_2{};
  std::array<double, VolumeDim> center_3{};
  for (size_t i = 0; i < VolumeDim; ++i) {
    gsl::at(center_1, i) = real_dis(gen);
    gsl::at(center_2, i) = real_dis(gen);
    gsl::at(center_3, i) = real_dis(gen);
  }

  // Name of FunctionOfTime to read
  const std::string function_of_time_for_scaling{"ExpansionFactor"s};

  GeneralizedHarmonic::ConstraintDamping::TimeDependentTripleGaussian<VolumeDim,
                                                                      Fr>
      triple_gauss{constant,
                   amplitude_1,
                   width_1,
                   center_1,
                   amplitude_2,
                   width_2,
                   center_2,
                   amplitude_3,
                   width_3,
                   center_3,
                   function_of_time_for_scaling};

  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      std::move(triple_gauss), "time_dependent_triple_gaussian", used_for_size,
      {{{-1.0, 1.0}}}, function_of_time_for_scaling, constant, amplitude_1,
      width_1, center_1, amplitude_2, width_2, center_2, amplitude_3, width_3,
      center_3);

  std::unique_ptr<GeneralizedHarmonic::ConstraintDamping::
                      TimeDependentTripleGaussian<VolumeDim, Fr>>
      triple_gauss_unique_ptr =
          std::make_unique<GeneralizedHarmonic::ConstraintDamping::
                               TimeDependentTripleGaussian<VolumeDim, Fr>>(
              constant, amplitude_1, width_1, center_1, amplitude_2, width_2,
              center_2, amplitude_3, width_3, center_3,
              function_of_time_for_scaling);

  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      std::move(triple_gauss_unique_ptr->get_clone()),
      "time_dependent_triple_gaussian", used_for_size, {{{-1.0, 1.0}}},
      function_of_time_for_scaling, constant, amplitude_1, width_1, center_1,
      amplitude_2, width_2, center_2, amplitude_3, width_3, center_3);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintDamp.TimeDep3Gauss",
    "[PointwiseFunctions][Unit]") {
  const DataVector dv{5};

  pypp::SetupLocalPythonEnvironment{
      "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python"};

  using VolumeDims = tmpl::integral_list<size_t, 1, 2, 3>;
  using Frames = tmpl::list<Frame::Grid, Frame::Inertial>;

  tmpl::for_each<VolumeDims>([&dv](auto dim_v) {
    using VolumeDim = typename decltype(dim_v)::type;
    tmpl::for_each<Frames>([&dv](auto frame_v) {
      using Fr = typename decltype(frame_v)::type;
      test_triple_gaussian_random<VolumeDim::value, DataVector, Fr>(dv);
      test_triple_gaussian_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });

  TestHelpers::test_factory_creation<GeneralizedHarmonic::ConstraintDamping::
                                         DampingFunction<1, Frame::Inertial>>(
      "TimeDependentTripleGaussian:\n"
      "  Constant: 4.0\n"
      "  Amplitude1: 3.0\n"
      "  Width1: 2.0\n"
      "  Center1: [-9.0]\n"
      "  Amplitude2: 4.0\n"
      "  Width2: 1.5\n"
      "  Center2: [8.0]\n"
      "  Amplitude3: 5.0\n"
      "  Width3: 1.0\n"
      "  Center3: [7.0]\n"
      "  FunctionOfTimeForScaling: ExpansionFactor");

  const double constant_3d{5.0};
  const double amplitude_1_3d{4.0};
  const double width_1_3d{1.5};
  const std::array<double, 3> center_1_3d{{1.1, -2.2, 3.3}};
  const double amplitude_2_3d{3.0};
  const double width_2_3d{2.0};
  const std::array<double, 3> center_2_3d{{4.4, -5.5, 6.6}};
  const double amplitude_3_3d{5.0};
  const double width_3_3d{1.0};
  const std::array<double, 3> center_3_3d{{7.7, -8.8, 9.9}};

  // Name of FunctionOfTime to read
  const std::string function_of_time_for_scaling{"ExpansionFactor"s};
  const GeneralizedHarmonic::ConstraintDamping::TimeDependentTripleGaussian<
      3, Frame::Inertial>
      triple_gauss_3d{constant_3d,
                      amplitude_1_3d,
                      width_1_3d,
                      center_1_3d,
                      amplitude_2_3d,
                      width_2_3d,
                      center_2_3d,
                      amplitude_3_3d,
                      width_3_3d,
                      center_3_3d,
                      function_of_time_for_scaling};
  const auto created_triple_gauss = TestHelpers::test_creation<
      GeneralizedHarmonic::ConstraintDamping::TimeDependentTripleGaussian<
          3, Frame::Inertial>>(
      "Constant: 5.0\n"
      "Amplitude1: 4.0\n"
      "Width1: 1.5\n"
      "Center1: [1.1, -2.2, 3.3]\n"
      "Amplitude2: 3.0\n"
      "Width2: 2.0\n"
      "Center2: [4.4, -5.5, 6.6]\n"
      "Amplitude3: 5.0\n"
      "Width3: 1.0\n"
      "Center3: [7.7, -8.8, 9.9]\n"
      "FunctionOfTimeForScaling: ExpansionFactor");
  CHECK(created_triple_gauss == triple_gauss_3d);
  const auto created_triple_gauss_gh_damping_function =
      TestHelpers::test_factory_creation<
          GeneralizedHarmonic::ConstraintDamping::DampingFunction<
              3, Frame::Inertial>>(
          "TimeDependentTripleGaussian:\n"
          "  Constant: 5.0\n"
          "  Amplitude1: 4.0\n"
          "  Width1: 1.5\n"
          "  Center1: [1.1, -2.2, 3.3]\n"
          "  Amplitude2: 3.0\n"
          "  Width2: 2.0\n"
          "  Center2: [4.4, -5.5, 6.6]\n"
          "  Amplitude3: 5.0\n"
          "  Width3: 1.0\n"
          "  Center3: [7.7, -8.8, 9.9]\n"
          "  FunctionOfTimeForScaling: ExpansionFactor");

  test_serialization(triple_gauss_3d);
}
