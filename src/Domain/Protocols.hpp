// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace domain::protocols {
/*
 * \brief Indicates the `ConformingType` represents compile-time choices for a
 * Domain.
 *
 * Requires that the class has this member function:
 * - `enable_time_dependent_maps`: A boolean that chooses whether to include a
 * domain's time-dependent maps.
 *
 * Here is an example of a class that conforms to this protocol:
 *
 * \snippet Domain/Test_Protocols.cpp domain_metavariables_example
 *
 */
struct Metavariables {
  template <typename ConformingType>
  struct test {
    using enable_time_dependent_maps_type = const bool;
    using enable_time_dependent_maps_return_type =
        decltype(ConformingType::enable_time_dependent_maps);
    static_assert(std::is_same_v<enable_time_dependent_maps_type,
                                 enable_time_dependent_maps_return_type>,
                  "The metavariable 'enable_time_dependent_maps' should be a "
                  "static constexpr bool.");
  };
};
}  // namespace domain::protocols
