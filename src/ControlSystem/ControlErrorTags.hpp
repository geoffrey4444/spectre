// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace domain::OptionTags {
template <size_t>
struct DomainCreator;
}

namespace Tags {
struct ExcisionXLocationA : db::SimpleTag {
  using type = double;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  template <typename Metavariables>
  static auto create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) noexcept {
    double x_loc{};
    // The x location of the excision center only makes sense if the
    // creator is a BBH domain, so we must cast first and throw
    // if the domain is not a BBH domain.
    try {
      const auto bco =
          dynamic_cast<domain::creators::BinaryCompactObject*>(
            domain_creator.get());
      x_loc = bco->xcoord_object_a();
    } catch (const std::bad_cast& e) {
      ERROR(
          "Can only get the x location of the excision surface with a "
          "BinaryCompactObject domain");
    }
    return x_loc;
  }
};

struct ExcisionXLocationB : db::SimpleTag {
  using type = double;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  template <typename Metavariables>
  static auto create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) noexcept {
    double x_loc{};
    // The x location of the excision center only makes sense if the
    // creator is a BBH domain, so we must cast first and throw
    // if the domain is not a BBH domain.
    try {
      const auto bco =
          dynamic_cast<domain::creators::BinaryCompactObject*>(
            domain_creator.get());
      x_loc = bco->xcoord_object_b();
    } catch (const std::bad_cast& e) {
      ERROR(
          "Can only get the x location of the excision surface with a "
          "BinaryCompactxObject domain");
    }
    return x_loc;
  }
};
}  // namespace Tags

namespace ControlSystem {

/// Struct to define the control system structure for the expansion map.
/// Defines the name of the map
template <typename Metavariables>
struct Expansion {
  // Identifies name in the FofT list
  static std::string name() { return "ExpansionFactor"; };

  using return_type = std::map<::TimeStepId, DataVector>;

  using argument_tags =
      tmpl::list<Tags::HorizonCenter<typename Metavariables::AhA>,
                 Tags::HorizonCenter<typename Metavariables::AhB>,
                 Tags::ExcisionXLocationA, Tags::ExcisionXLocationB,
                 ::domain::Tags::FunctionsOfTime>;

  // Since the data from the horizons might arrive at different times, the point
  // here is to calculate the control error for as many times as horizon centers
  // has been calculated, and to return the error at each time: hence the map
  // from TimeStepId's to what _should_ be the return type of a DataVector.
  static std::map<::TimeStepId, DataVector> apply(
      const std::map<::TimeStepId, std::array<double, 3>>& horizon_centers_a,
      const std::map<::TimeStepId, std::array<double, 3>>& horizon_centers_b,
      const double x_loc_excision_a, const double x_loc_excision_b,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time
  ) {
    std::map<::TimeStepId, DataVector> ret{};
    Parallel::printf("%lf %lf\n", x_loc_excision_a, x_loc_excision_b);
    // Iterate over the horizon centers while both exist for
    // a particular time.
    for (auto it_a = horizon_centers_a.begin(),
              it_b = horizon_centers_b.begin();
         it_a != horizon_centers_a.end() and it_b != horizon_centers_b.end();
         it_a++, it_b++) {
      ASSERT(it_a->first == it_b->first,
             "Inconsistent horizon data " << *it_a << " " << *it_b);
      const ::TimeStepId& time_id = it_a->first;

      const double x_loc_horizon_a = (it_a->second)[0];
      const double x_loc_horizon_b = (it_b->second)[0];
      Parallel::printf("%lf %lf\n", x_loc_horizon_a, x_loc_horizon_b);
      const double dx0 = (x_loc_horizon_a - x_loc_horizon_b) /
                         (x_loc_excision_a - x_loc_excision_b);

      ret[time_id] = DataVector{functions_of_time.at("ExpansionFactor")
                                    ->func(time_id.substep_time().value())[0] *
                                (dx0 - 1.)};
    }
    return ret;
  };
};
}  // namespace ControlSystem
