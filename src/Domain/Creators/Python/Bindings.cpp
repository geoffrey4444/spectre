// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Domain/Creators/Python/Brick.hpp"
#include "Domain/Creators/Python/Cylinder.hpp"
#include "Domain/Creators/Python/DomainCreator.hpp"
#include "Domain/Creators/Python/Interval.hpp"
#include "Domain/Creators/Python/Rectangle.hpp"
#include "Domain/Creators/Python/Shell.hpp"
#include "Domain/Creators/Python/Sphere.hpp"

namespace domain::creators {

PYBIND11_MODULE(_PyDomainCreators, m) {  // NOLINT
  // Order is important: The base class `DomainCreator` needs to have its
  // bindings set up before the derived classes
  py_bindings::bind_domain_creator(m);
  py_bindings::bind_brick(m);
  py_bindings::bind_cylinder(m);
  py_bindings::bind_interval(m);
  py_bindings::bind_rectangle(m);
  py_bindings::bind_shell(m);
  py_bindings::bind_sphere(m);
}

}  // namespace domain::creators
