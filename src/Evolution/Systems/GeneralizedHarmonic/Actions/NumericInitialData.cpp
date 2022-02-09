// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Actions/NumericInitialData.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace GeneralizedHarmonic {

std::ostream& operator<<(std::ostream& os,
                         const NumericInitialDataVariables& value) {
  switch (value) {
    case NumericInitialDataVariables::GeneralizedHarmonic:
      return os << "GeneralizedHarmonic";
    case NumericInitialDataVariables::Adm:
      return os << "Adm";
    default:
      ERROR("Unknown GeneralizedHarmonic::NumericInitialDataVariables");
  }
}

}  // namespace GeneralizedHarmonic

template <>
GeneralizedHarmonic::NumericInitialDataVariables
Options::create_from_yaml<GeneralizedHarmonic::NumericInitialDataVariables>::
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    create<void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("GeneralizedHarmonic" == type_read) {
    return GeneralizedHarmonic::NumericInitialDataVariables::
        GeneralizedHarmonic;
  } else if ("Adm" == type_read) {
    return GeneralizedHarmonic::NumericInitialDataVariables::Adm;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to GeneralizedHarmonic::NumericInitialDataVariables. "
                     "Must be one of 'GeneralizedHarmonic' or 'Adm'.");
}
