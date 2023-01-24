# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import unittest


def jacobian_diagnostic(analytic_jacobian, numeric_jacobian_transpose):
    analytic_sum = analytic_jacobian.sum(axis=0)
    numeric_sum = numeric_jacobian_transpose.sum(axis=1)
    for i, x in enumerate(analytic_sum):
        if abs(x) < 1.e-10 and abs(numeric_sum[i]) < 1.e-10:
            analytic_sum[i] = 0.0
            numeric_sum[i] = 0.0
    jac_diff_sum = 1.0 - analytic_sum / (numeric_sum + 1.e-20)
    jac_diag = np.sqrt(np.dot(jac_diff_sum, jac_diff_sum))
    return jac_diag


if __name__ == '__main__':
    unittest.main(verbosity=2)
