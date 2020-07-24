// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"

namespace observers {
/// Passed to `RegisterWithObservers` action to register observer event.
template <typename ObservationValueTag, typename ObsType>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTagsList, ObservationValueTag>> =
                nullptr>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) noexcept {
    return {
        observers::TypeOfObservation::ReductionAndVolume,
        observers::ObservationId{
            static_cast<double>(db::get<ObservationValueTag>(box)), ObsType{}}};
  }

  template <
      typename ParallelComponent, typename DbTagsList, typename ArrayIndex,
      Requires<not tmpl::list_contains_v<DbTagsList, ObservationValueTag>> =
          nullptr>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    ERROR("Observation id not found in box");
  }
};
}  // namespace observers
