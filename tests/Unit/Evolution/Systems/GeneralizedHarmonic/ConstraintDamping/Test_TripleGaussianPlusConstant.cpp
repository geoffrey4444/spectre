// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TripleGaussianPlusConstant.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t VolumeDim, typename DataType, typename Fr>
void test_triple_gaussian_plus_constant_random(
    const DataType& used_for_size) noexcept {
  Parallel::register_derived_classes_with_charm<
      GeneralizedHarmonic::ConstraintDamping::TripleGaussianPlusConstant<
          VolumeDim, Fr>>();

  // Generate the amplitude and width
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> positive_dis(0, 1);

  const double constant = real_dis(gen);
  const std::array<double, 3> amplitudes{
      {positive_dis(gen), positive_dis(gen), positive_dis(gen)}};
  // If the width is too small then the terms in the second derivative
  // can become very large and fail the test due to rounding errors.
  const std::array<double, 3> widths{{positive_dis(gen) + 0.5,
                                      positive_dis(gen) + 0.5,
                                      positive_dis(gen) + 0.5}};

  // Generate the centers
  std::array<std::array<double, VolumeDim>, 3> centers{};
  for (size_t which_gaussian = 0; which_gaussian < 3; ++which_gaussian) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      gsl::at(gsl::at(centers, which_gaussian), i) = real_dis(gen);
    }
  }

  // The name of a FunctionOfTime that could be used to update
  // the time-dependent Gaussian widths. Since we don't actually have a
  // functions_of_time set up for this test, this name could be anything
  const std::string function_of_time_name{"ExpansionFactor"};

  GeneralizedHarmonic::ConstraintDamping::TripleGaussianPlusConstant<VolumeDim,
                                                                     Fr>
      triple_gauss_plus_const{constant, amplitudes, widths, centers,
                              function_of_time_name};

  // Note: pass each center separately, because pypp::check_with_random_values
  // only accepts 1D arrays as parameter inputs
  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      triple_gauss_plus_const, "triple_gaussian_plus_constant", used_for_size,
      {{{-1.0, 1.0}}}, constant, amplitudes, widths, centers[0], centers[1],
      centers[2]);

  // Check that the call operator did not modify function_of_time_name
  CHECK(triple_gauss_plus_const.function_of_time_for_scaling_name ==
        function_of_time_name);

  CHECK(triple_gauss_plus_const.is_time_dependent == true);

  std::unique_ptr<GeneralizedHarmonic::ConstraintDamping::
                      TripleGaussianPlusConstant<VolumeDim, Fr>>
      triple_gauss_plus_const_unique_ptr =
          std::make_unique<GeneralizedHarmonic::ConstraintDamping::
                               TripleGaussianPlusConstant<VolumeDim, Fr>>(
              constant, amplitudes, widths, centers, function_of_time_name);

  TestHelpers::GeneralizedHarmonic::ConstraintDamping::check(
      triple_gauss_plus_const_unique_ptr->get_clone(),
      "triple_gaussian_plus_constant", used_for_size, {{{-1.0, 1.0}}}, constant,
      amplitudes, widths, centers[0], centers[1], centers[2]);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintDamp.TripleGauss",
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
      test_triple_gaussian_plus_constant_random<VolumeDim::value, DataVector,
                                                Fr>(dv);
      test_triple_gaussian_plus_constant_random<VolumeDim::value, double, Fr>(
          std::numeric_limits<double>::signaling_NaN());
    });
  });

  TestHelpers::test_factory_creation<GeneralizedHarmonic::ConstraintDamping::
                                         DampingFunction<1, Frame::Inertial>>(
      "TripleGaussianPlusConstant:\n"
      "  Constant: 4.0\n"
      "  Amplitudes: [3.0, 2.2, 1.1]\n"
      "  Widths: [2.0, 3.0, 4.0]\n"
      "  Centers: [[-9.0], [0.0], [9.0]]\n"
      "  FunctionOfTimeName: ExpansionFactor");

  const double constant_3d{5.0};
  const std::array<double, 3> amplitudes_3d{{4.0, 2.0, 5.0}};
  const std::array<double, 3> widths_3d{{2.5, 2.0, 1.5}};
  const std::array<std::array<double, 3>, 3> centers_3d{
      {{1.1, -2.2, 3.3}, {-3.2, 2.1, -1.0}, {0.1, 0.0, -0.1}}};
  const std::string function_of_time_name{"ExpansionFactor"};
  const GeneralizedHarmonic::ConstraintDamping::TripleGaussianPlusConstant<
      3, Frame::Inertial>
      triple_gauss_plus_const_3d{constant_3d, amplitudes_3d, widths_3d,
                                 centers_3d, function_of_time_name};
  const auto created_triple_gauss_plus_const = TestHelpers::test_creation<
      GeneralizedHarmonic::ConstraintDamping::TripleGaussianPlusConstant<
          3, Frame::Inertial>>(
      "Constant: 5.0\n"
      "Amplitudes: [4.0, 2.0, 5.0]\n"
      "Widths: [2.5, 2.0, 1.5]\n"
      "Centers: [[1.1, -2.2, 3.3], [-3.2, 2.1, -1.0], [0.1, 0.0, -0.1]]\n"
      "FunctionOfTimeName: ExpansionFactor");
  CHECK(created_triple_gauss_plus_const == triple_gauss_plus_const_3d);
  const auto created_gauss_gh_damping_function =
      TestHelpers::test_factory_creation<
          GeneralizedHarmonic::ConstraintDamping::DampingFunction<
              3, Frame::Inertial>>(
          "TripleGaussianPlusConstant:\n"
          "  Constant: 5.0\n"
          "  Amplitudes: [4.0, 2.0, 5.0]\n"
          "  Widths: [2.5, 2.0, 1.5]\n"
          "  Centers: [[1.1, -2.2, 3.3], [-3.2, 2.1, -1.0], [0.1, 0.0, -0.1]]\n"
          "  FunctionOfTimeName: ExpansionFactor");

  test_serialization(triple_gauss_plus_const_3d);
}
