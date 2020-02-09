#include <Eigen/Sparse>
#include <mpi.h>
#include <string>

Eigen::SparseMatrix<double, Eigen::RowMajor>
read_petsc_binary(MPI_Comm comm, std::string filename);
