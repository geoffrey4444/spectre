// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps::TimeDependent {
/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
class SphericalCompression;
/// \endcond

/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Time-dependent compression of a finite 3D spherical volume.
 *
 * \details Let \f$\xi^i\f$ be the unmapped coordinates, and let \f$\rho\f$ be
 * the Euclidean radius corresponding to these coordinates with respect to
 * some center \f$C^i\f$. This map is the
 * identity everywhere except in a spherical region \f$\rho \leq
 * \rho_{\rm max}\f$, where instead the map applies compression that is
 * spherically symmetric about the center \f$C^i\f$. The amount of
 * compression decreases linearly from a maximum at
 * \f$\rho \leq \rho_{\rm min}\f$ to zero at \f$\rho = \rho_{\rm max}\f$.
 * A scalar domain::FunctionsOfTime::FunctionOfTime \f$\lambda_{00}(t)\f$
 * controls the amount of compression.
 *
 * \note The mapped coordinates are a continuous function of the unmapped
 * coordinates, but the Jacobians are not continuous at \f$\rho_{\rm min}\f$
 * and \f$\rho_{\rm max}\f$. Therefore, \f$\rho_{\rm min}\f$ and \f$\rho_{\rm
 * max}\f$ should both be surfaces corresponding to cell boundaries.
 *
 * \note Currently, this map only performs a compression. A generalization of
 * this map could also change the region's shape as well as its size, by
 * including more terms than the spherically symmetric term included here.
 *
 * ### Mapped coordinates
 *
 * The mapped coordinates
 * \f$x^i\f$ are related to the unmapped coordinates \f$\xi^i\f$
 * as follows:
 * \f{align}{
 * x^i &= \xi^i - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho^i}{\rho_{\rm min}}, \mbox{ } \rho < \rho_{\rm min}, \\
 * x^i &= \xi^i - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}} \rho^i,
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max}, \\
 * x^i &= \xi^i, \mbox{ } \rho_{\rm max} < \rho,
 * \f}
 * where \f$\rho^i = \xi^i - C^i\f$ is the Euclidean radial position vector in
 * the unmapped coordinates with respect to the center \f$C^i\f$, \f$\rho =
 * \sqrt{\delta_{kl}\left(\xi^k - C^l\right)\left(\xi^l - C^l\right)}\f$ is the
 * Euclidean magnitude of \f$\rho^i\f$, and \f$\rho^j = \delta_{ij} \rho^i\f$.
 *
 * ### Frame velocity
 *
 * The frame velocity \f$v^i \equiv dx^i/dt\f$ is then
 * \f{align}{
 * v^i &= - \frac{\lambda_{00}^\prime(t)}{\sqrt{4\pi}}
 * \frac{\rho^i}{\rho_{\rm min}}, \mbox{ } \rho < \rho_{\rm min}, \\
 * v^i &= - \frac{\lambda_{00}^\prime(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}} \rho^i,
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max}, \\
 * v^i &= 0, \mbox{ } \rho_{\rm max} < \rho.
 * \f}
 *
 * ### Jacobian
 *
 * Differentiating the equations for \f$x^i\f$ gives the Jacobian
 * \f$\partial x^i / \partial \xi^j\f$. Using the result
 * \f{align}{
 * \frac{\partial \rho^i}{\partial \xi^j} &= \frac{\partial}{\partial \xi^j}
 * \left(\xi^i - C^i\right) = \frac{\partial \xi^i}{\partial \xi^j}
 * = \delta^i_{j}
 * \f}
 * and taking the derivatives yields
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j \left(1
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm min}}\right),
 * \mbox{ } \rho < \rho_{\rm min},\\
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j
 * \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * - \rho^i \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\partial}{\partial \xi^j}\left(
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right),
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max},\\
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j,
 * \rho_{\rm max} < \rho.
 * \f}
 * Inserting
 * \f{align}{
 * \frac{\partial}{\partial \xi^j}\left(
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * &= \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}
 * \frac{\partial}{\partial \xi^j}\left(\frac{1}{\rho}\right)
 * = - \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}} \frac{1}{\rho^2}
 * \frac{\partial \rho}{\partial \xi^j}
 * \f}
 * and
 * \f{align}{
 * \frac{\partial \rho}{\partial \xi^j} &= \frac{\rho_j}{\rho}.
 * \f}
 * into the Jacobian yields
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j \left(1
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \frac{1}{\rho_{\rm min}}\right),
 * \mbox{ } \rho < \rho_{\rm min},\\
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j
 * \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * + \rho^i \rho_j \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}\frac{1}{\rho^3},
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max},\\
 * \frac{\partial x^i}{\partial \xi^j} &= \delta^i_j,
 * \rho_{\rm max} < \rho.
 * \f}
 *
 * ### Inverse Jacobian
 *
 * This map finds the inverse Jacobian by first finding the Jacobian and then
 * numerically inverting it.
 *
 * ### Inverse map
 *
 * The map is invertible if \f$\rho_{\rm min} - \rho_{\rm max} < \lambda_{00}(t)
 * / \sqrt{4\pi} < \rho_{\rm min}\f$. In this case, the
 * inverse mapping can be derived as follows. Let \f$r^i \equiv x^i - C^i\f$.
 * In terms of \f$r^i\f$, the map is
 * \f{align}{
 * r^i &= \rho^i \left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}\right), \mbox{ } \rho < \rho_{\rm min}, \\
 * r^i &= \rho^i\left(1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} / \rho - 1}{\rho_{\rm max} - \rho_{\rm min}}\right),
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max}, \\
 * r^i &= \rho^i, \mbox{ } \rho_{\rm max} < \rho.
 * \f}
 *
 * Taking the Euclidean magnitude of both sides and simplifying yields
 * \f{align}{
 * \frac{r}{\rho} &= 1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}, \mbox{ } \rho < \rho_{\rm min}, \\
 * \frac{r}{\rho} &= 1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}/\rho - 1}{\rho_{\rm max} - \rho_{\rm min}},
 * \mbox{ } \rho_{\rm min} \leq \rho < \rho_{\rm max}, \\
 * \frac{r}{\rho} &= 1, \mbox{ } \rho_{\rm max} < \rho,
 * \f}
 * which implies
 * \f{align}{
 * r^i = \rho^i \frac{r}{\rho} \Rightarrow \rho^i = r^i \frac{\rho}{r}.
 * \f}
 *
 * Inserting \f$\rho_{min}\f$ or \f$\rho_{max}\f$ then gives the corresponding
 * bounds in the mapped coordinates:
 * \f{align}{
 * r_{\rm min} &= \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * r_{\rm max} &= \rho_{\rm max}.
 * \f}
 *
 * In the regime \f$\rho_{\rm min} \leq \rho < \rho_{\rm max}\f$, rearranging
 * yields a linear relationship between \f$\rho\f$ and \f$r\f$, which
 * can then be solved for \f$\rho(r)\f$:
 * \f{align}{
 * r &= \rho - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max} - \rho}{\rho_{\rm max} - \rho_{\rm min}}\\
 * \Rightarrow r &= \rho \left(1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}\right)
 * - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{\rho_{\rm max}}{\rho_{\rm max} - \rho_{\rm min}}.
 * \f}
 * Solving this linear equation for \f$\rho\f$ yields
 * \f{align}{
 * \rho &= \frac{r+\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}}{1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}}.
 * \f}
 *
 * Inserting the expressions for \f$\rho\f$ into the equation
 * \f{align}{
 * \rho^i = r^i \frac{\rho}{r}
 * \f}
 * then gives
 * \f{align}{
 * \rho^i &= \frac{r^i}{1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}},
 * \mbox{ } r < \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * \rho^i &= r^i
 * \frac{1+\frac{1}{r}\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}}{1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}},
 * \mbox{ }\rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \leq r
 * \leq \rho_{\rm max},\\
 * \rho^i &= r^i, \mbox{ }\rho_{\rm max} < r.
 * \f}
 * Finally, inserting \f$\rho^i = \xi^i - C^i\f$ yields the inverse map:
 * \f{align}{
 * \xi^i &= \frac{r^i}{1 - \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm min}}} + C^i,
 * \mbox{ } r < \rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}},\\
 * \xi^i &= r^i
 * \frac{1+\frac{1}{r}\frac{\lambda_{00}(t)}{\sqrt{4\pi}}\frac{\rho_{\rm
 * max}}{\rho_{\rm max}-\rho_{\rm min}}}{1 + \frac{\lambda_{00}(t)}{\sqrt{4\pi}}
 * \frac{1}{\rho_{\rm max} - \rho_{\rm min}}} + C^i,
 * \mbox{ }\rho_{\rm min} - \frac{\lambda_{00}(t)}{\sqrt{4\pi}} \leq r
 * \leq \rho_{\rm max},\\
 * \xi^i &= r^i + C^i = x^i, \mbox{ }\rho_{\rm max} < r.
 * \f}
 *
 */
template <>
class SphericalCompression<3> {
 public:
  static constexpr size_t dim = 3;

  explicit SphericalCompression(std::string function_of_time_name,
                                double min_radius, double max_radius,
                                const std::array<double, 3>& center) noexcept;
  SphericalCompression() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> frame_velocity(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  friend bool operator==(const SphericalCompression<3>& lhs,
                         const SphericalCompression<3>& rhs) noexcept;
  std::string f_of_t_name_;
  double min_radius_ = std::numeric_limits<double>::signaling_NaN();
  double max_radius_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_;
};

bool operator!=(const SphericalCompression<3>& lhs,
                const SphericalCompression<3>& rhs) noexcept;

}  // namespace domain::CoordinateMaps::TimeDependent
