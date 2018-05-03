// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function generalized_eigenvalue.

#pragma once

/// \cond
class DataVector;
class Matrix;
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Solve the generalized eigenvalue problem for two matrices.
 *
 * This function uses the lapack routine dggev to solve the
 * generalized eigenvalue problem \f$A v_a =\lambda_a B v_a \f$
 * for the generalized eigenvalues \f$lambda_a\f$ and corresponding
 * eigenvectors \f$v_a\f$.
 * `A` and `B` are each a `Matrix`; they correspond to square
 * matrices \f$A\f$ and \f$B\f$ that are the same order \f$N\f$.
 * `eigenvalues_real_part` is a `DataVector` of size \f$N\f$ that
 * will store the real parts of the eigenvalues,
 * `eigenvalues_imaginary_part` is a `DataVector` of size \f$N\f$
 * that will store the imaginary parts of the eigenvalues.
 * Complex eigenvalues always form complex conjugate pairs, and
 * the \f$j\f$ and \f$j+1\f$ eigenvalues will have the forms
 * \f$a+ib\f$ and \f$a-ib\f$, respectively. The eigenvectors
 * are returned as the columns of a square `Matrix` of order $N$
 * called `eigenvectors`. If eigenvalue j is real, then column j of `v` is
 * the corresponding eigenvector. If eigenvalue j and j+1 are
 * complex-conjugate pairs, then the eigenvector for
 * eigenvalue j is (column j) + i (column j+1), and the
 * eigenvector for eigenvalue j+1 is (column j) - i (column j+1).
 */
void generalized_eigenvalue(const Matrix& A, const Matrix& B,
                            DataVector& eigenvalues_real_part,
                            DataVector& eigenvalues_imaginary_part,
                            Matrix& eigenvectors) noexcept;
