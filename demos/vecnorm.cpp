#include <CL/sycl.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numeric>

#include <spmv/L2GMap.h>
#include <spmv/Vector.h>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm comm{MPI_COMM_WORLD};
  MPI_Comm_rank(comm, &rank);

  constexpr std::size_t local_size = 1000;
  constexpr std::size_t nghost = 10;

  std::vector<double> v(local_size + nghost, 1);
  std::vector<std::int64_t> ghosts(nghost);
  std::iota(ghosts.begin(), ghosts.end(), (rank + 1) * local_size);

  auto map = std::make_shared<spmv::L2GMap>(MPI_COMM_WORLD, local_size, ghosts);
  auto data = std::make_shared<cl::sycl::buffer<double, 1>>(v);

  spmv::Vector<double> vec(data, map);
  double norm = vec.norm();
  double ex_norm = std::sqrt((double)map->global_size());

  if (rank == 0)
  {
    std::cout << std::setprecision(10) << std::endl;
    std::cout << "Computed Norm: " << norm << std::endl;
    std::cout << "Exact Norm: " << ex_norm << std::endl;
  }
  assert(fabs(norm - ex_norm) < 1e-5);

  vec += vec;
  norm = vec.norm();
  if (rank == 0)
  {
    std::cout << std::endl;
    std::cout << "Computed Norm: " << norm << std::endl;
    std::cout << "Exact Norm: " << ex_norm * 2 << std::endl;
  }

  spmv::Vector<double> vec2 = vec + (vec * 2.);
  vec2 *= 0.5;
  norm = vec2.norm();
  if (rank == 0)
  {
    std::cout << std::endl;
    std::cout << "Computed Norm: " << norm << std::endl;
    std::cout << "Exact Norm: " << 3 * ex_norm << std::endl;
  }

  return 0;
}