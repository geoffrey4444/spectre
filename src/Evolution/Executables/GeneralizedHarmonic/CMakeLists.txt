# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    DiscontinuousGalerkin
    DomainCreators
    GeneralizedHarmonic
    Informer
    LinearOperators
    MathFunctions
    Time
    Utilities
    WaveEquation
    ${SPECTRE_LIBRARIES}
  )

add_spectre_executable(
  EvolveGeneralizedHarmonicPeriodic
  EvolveGeneralizedHarmonicPeriodic
  Evolution/Executables/GeneralizedHarmonic
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
)
