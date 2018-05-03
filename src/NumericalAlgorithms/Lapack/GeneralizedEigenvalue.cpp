// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Lapack/GeneralizedEigenvalue.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"

// LAPACK routine to do the generalized eigenvalue problem
extern "C" {
extern void dggev_(char*, char*, int*, double*, int*, double*, int*, double*,
                   double*, double*, double*, int*, double*, int*, double*,
                   int*, int*);
}

void generalized_eigenvalue(const Matrix& A, const Matrix& B,
                            DataVector& eigenvalues_real_part,
                            DataVector& eigenvalues_imaginary_part,
                            Matrix& eigenvectors) noexcept {
  // Sanity checks on the sizes of the vectors and matrices
  const size_t number_of_rows = A.rows();
  ASSERT(number_of_rows == A.columns(), "Matrix A should be square");
  ASSERT(number_of_rows == B.rows() && number_of_rows == B.columns(),
         "Matrix A and matrix B should be the same size");
  ASSERT(number_of_rows == eigenvectors.rows() &&
                    number_of_rows == eigenvectors.columns(),
                "Matrix A and matrix eigenvectors should have the same size");
  ASSERT(number_of_rows == eigenvalues_real_part.size() &&
                    number_of_rows == eigenvalues_imaginary_part.size(),
                "eigenvalues DataVector sizes should equal number of columns "
                "in Matrix A");

  // Set up parameters for the lapack call
  // Lapack uses chars to decide whether to compute the left eigenvectors,
  // the right eigenvectors, both, or neither. 'N' means do not compute,
  // 'V' means do compute.
  char compute_left_eigenvectors = 'N';
  char compute_right_eigenvectors = 'V';

  // Lapack overwrites the input matrices, so copy the inputs into
  // non-const matrices
  Matrix A_for_lapack = A;
  Matrix B_for_lapack = B;

  // Lapack expects the sizes to be ints, not size_t.
  // NOTE: not const because lapack function dggev_() arguments
  // are not const.
  int matrix_and_vector_size = static_cast<int>(number_of_rows);

  // Lapack splits the eigenvalues into unnormalized real and imaginary
  // parts, which it calls alphar and alphai, and a normalization,
  // which it calls beta. The real and imaginary parts of the eigenvalues are
  // found by dividing the unnormalized results by the normalization.
  DataVector unnormalized_eigenvalue_real_part(number_of_rows, 0.0);
  DataVector unnormalized_eigenvalue_imaginary_part(number_of_rows, 0.0);
  DataVector eigenvalue_normalization(number_of_rows, 0.0);

  // Lapack uses a work vector, that should have a size 8N
  // for doing eigenvalue problems with NxN matrices
  // Note: non-const int, not size_t, because lapack wants an non-cosnt int
  int work_size = number_of_rows * 8;
  DataVector lapack_work(number_of_rows, 0.0);

  //  Lapack uses an integer called info to return its status
  //  info = 0 : success
  //  info = -i: ith argument had bad value
  //  info > 0: some other failure
  int info = 0;

  dggev_(&compute_left_eigenvectors, &compute_right_eigenvectors,
         &matrix_and_vector_size, A_for_lapack.data(),
         &matrix_and_vector_size, B_for_lapack.data(),
         &matrix_and_vector_size, unnormalized_eigenvalue_real_part.data(),
         unnormalized_eigenvalue_imaginary_part.data(),
         eigenvalue_normalization.data(), eigenvectors.data(),
         &matrix_and_vector_size, eigenvectors.data(), &matrix_and_vector_size,
         lapack_work.data(), &work_size, &info);

  ASSERT(info == 0, "Lapack failed to compute generalized eigenvectors");

  // compute the real and imaginary parts of the eigenvalues
  for (size_t i = 0; i < eigenvalues_real_part.size(); ++i) {
    eigenvalues_real_part[i] =
        unnormalized_eigenvalue_real_part[i] / eigenvalue_normalization[i];
    eigenvalues_imaginary_part[i] =
        unnormalized_eigenvalue_imaginary_part[i] / eigenvalue_normalization[i];
  }
}
