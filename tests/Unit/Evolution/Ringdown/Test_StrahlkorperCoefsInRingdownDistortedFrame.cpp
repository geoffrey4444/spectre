// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "Evolution/Ringdown/StrahlkorperCoefsInRingdownDistortedFrame.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Ringdown.SCoefsRDis", "[Unit][Evolution]") {
  const std::string& path_to_horizons_h5{
      "/Users/geoffrey/Downloads/spectre_vis_of_the_day/010224/"
      "l10lessvolendearlier/GhBinaryBlackHoleReductionData.h5"};
  const std::string surface_subfile_name{"ObservationAhC_Ylm"};
  std::vector<double> ahc_times{};
  for (size_t i = 0; i < 81; ++i) {
    ahc_times.push_back(4899.0 + 0.01 * i);
  }
  const size_t requested_number_of_times_from_end{7};
  const double match_time{4899.719999995199};
  const double settling_timescale{10.0};
  const std::array<double, 3> exp_func_and_2_derivs{{1.0, 0.0, 0.0}};
  const std::array<double, 3> exp_outer_bdry_func_and_2_derivs{
      {0.9951007901801149, -1.0001041026560264e-06, 4.2480033515241445e-14}};
  const std::array<std::array<double, 4>, 3> rot_func_and_2_derivs{
      {{0.30691045008109474, 1.1423081819002098e-05, 2.1790394436228194e-06,
        0.9517383965648255},
       {-0.12879009360095658, -3.221829327813639e-07, -1.3467985732678257e-06,
        0.04153139743267536},
       {-0.019929314042624748, -1.1750772017891358e-07, -2.3064243188540884e-07,
        -0.012813637112797401}}};
  evolution::Ringdown::strahlkorper_coefs_in_ringdown_distorted_frame(
      path_to_horizons_h5, surface_subfile_name, ahc_times,
      requested_number_of_times_from_end, match_time, settling_timescale,
      exp_func_and_2_derivs, exp_outer_bdry_func_and_2_derivs,
      rot_func_and_2_derivs);
}
