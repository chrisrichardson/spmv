// // Copyright (C) 2020 Chris Richardson and Igor Baratta
// // SPDX-License-Identifier:    MIT

// #include <CL/sycl.hpp>
// #include <Eigen/SparseCore>

// #include "Matrix.h"

// namespace spmv
// {
// template <typename T>
// spmv::Matrix<T>
// create_matrix(MPI_Comm comm, const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
//               std::int64_t nrows_local, std::int64_t ncols_local,
//               std::vector<std::int64_t> row_ghosts,
//               std::vector<std::int64_t> col_ghosts)
// {

//   int mpi_size, mpi_rank;
//   MPI_Comm_size(comm, &mpi_size);
//   MPI_Comm_rank(comm, &mpi_rank);

//   std::vector<std::int64_t> row_ranges(mpi_size + 1, 0);
//   MPI_Allgather(&nrows_local, 1, MPI_INT64_T, row_ranges.data() + 1, 1,
//                 MPI_INT64_T, comm);
//   for (int i = 0; i < mpi_size; ++i)
//     row_ranges[i + 1] += row_ranges[i];

//   // FIX: often same as rows?
//   std::vector<std::int64_t> col_ranges(mpi_size + 1, 0);
//   MPI_Allgather(&ncols_local, 1, MPI_INT64_T, col_ranges.data() + 1, 1,
//                 MPI_INT64_T, comm);
//   for (int i = 0; i < mpi_size; ++i)
//     col_ranges[i + 1] += col_ranges[i];

//   // Locate owner process for each row
//   std::vector<int> row_owner(row_ghosts.size());
//   for (std::size_t i = 0; i < row_ghosts.size(); ++i)
//   {
//     auto it
//         = std::upper_bound(row_ranges.begin(), row_ranges.end(), row_ghosts[i]);
//     assert(it != row_ranges.end());
//     row_owner[i] = it - row_ranges.begin() - 1;
//     assert(row_owner[i] != mpi_rank);
//   }

//   // Create a neighbour comm, remap row_owner to neighbour number
//   std::set<int> neighbour_set(row_owner.begin(), row_owner.end());
//   std::vector<int> dests(neighbour_set.begin(), neighbour_set.end());
//   std::map<int, int> proc_to_dest;
//   for (std::size_t i = 0; i < dests.size(); ++i)
//     proc_to_dest.insert({dests[i], i});
//   for (auto& q : row_owner)
//     q = proc_to_dest[q];

//   // Get list of sources (may be different from dests, requires AlltoAll to
//   // find)
//   std::vector<char> is_dest(mpi_size, 0);
//   for (int d : dests)
//     is_dest[d] = 1;
//   std::vector<char> is_source(mpi_size, 0);
//   MPI_Alltoall(is_dest.data(), 1, MPI_CHAR, is_source.data(), 1, MPI_CHAR,
//                comm);
//   std::vector<int> sources;
//   for (int i = 0; i < mpi_size; ++i)
//     if (is_source[i] == 1)
//       sources.push_back(i);

//   MPI_Comm neighbour_comm;
//   MPI_Dist_graph_create_adjacent(
//       comm, sources.size(), sources.data(), MPI_UNWEIGHTED, dests.size(),
//       dests.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbour_comm);

//   // send all ghost rows to their owners, using global col idx.
//   const std::int32_t* Aouter = mat.outerIndexPtr();
//   const std::int32_t* Ainner = mat.innerIndexPtr();
//   const T* Aval = mat.valuePtr();

//   std::vector<std::vector<std::int64_t>> p_to_index(dests.size());
//   std::vector<std::vector<T>> p_to_val(dests.size());
//   for (std::size_t i = 0; i < row_ghosts.size(); ++i)
//   {
//     const int p = row_owner[i];
//     assert(p != -1);
//     p_to_index[p].push_back(row_ghosts[i]);
//     p_to_val[p].push_back(0.0);
//     p_to_index[p].push_back(Aouter[nrows_local + i + 1]
//                             - Aouter[nrows_local + i]);
//     p_to_val[p].push_back(0.0);

//     const std::int64_t local_offset = col_ranges[mpi_rank];
//     for (int j = Aouter[nrows_local + i]; j < Aouter[nrows_local + i + 1]; ++j)
//     {
//       std::int64_t global_index;
//       if (Ainner[j] < ncols_local)
//         global_index = Ainner[j] + local_offset;
//       else
//       {
//         assert(Ainner[j] - ncols_local < (int)col_ghosts.size());
//         global_index = col_ghosts[Ainner[j] - ncols_local];
//       }
//       p_to_index[p].push_back(global_index);
//       p_to_val[p].push_back(Aval[j]);
//     }
//   }

//   std::vector<int> send_size(dests.size());
//   std::vector<std::int64_t> send_index;
//   std::vector<T> send_val;
//   std::vector<int> send_offset = {0};
//   for (std::size_t p = 0; p < dests.size(); ++p)
//   {
//     send_index.insert(send_index.end(), p_to_index[p].begin(),
//                       p_to_index[p].end());
//     send_val.insert(send_val.end(), p_to_val[p].begin(), p_to_val[p].end());
//     assert(p_to_val[p].size() == p_to_index[p].size());
//     send_size[p] = p_to_index[p].size();
//     send_offset.push_back(send_index.size());
//   }

//   std::vector<int> recv_size(sources.size());
//   MPI_Neighbor_alltoall(send_size.data(), 1, MPI_INT, recv_size.data(), 1,
//                         MPI_INT, neighbour_comm);

//   std::vector<int> recv_offset = {0};
//   for (int r : recv_size)
//     recv_offset.push_back(recv_offset.back() + r);

//   std::vector<std::int64_t> recv_index(recv_offset.back());
//   std::vector<T> recv_val(recv_offset.back());

//   MPI_Neighbor_alltoallv(send_index.data(), send_size.data(),
//                          send_offset.data(), MPI_INT64_T, recv_index.data(),
//                          recv_size.data(), recv_offset.data(), MPI_INT64_T,
//                          neighbour_comm);

//   MPI_Neighbor_alltoallv(send_val.data(), send_size.data(), send_offset.data(),
//                          mpi_type<T>(), recv_val.data(), recv_size.data(),
//                          recv_offset.data(), mpi_type<T>(), neighbour_comm);

//   // Create new map from global column index to local
//   std::map<std::int64_t, int> col_ghost_map;
//   for (std::int64_t q : col_ghosts)
//     col_ghost_map.insert({q, -1});

//   // Add any new ghost columns
//   int pos = 0;
//   while (pos < (int)recv_index.size())
//   {
//     //    std::int64_t global_row = recv_index[pos];
//     ++pos;
//     int nnz = recv_index[pos];
//     ++pos;
//     for (int k = 0; k < nnz; ++k)
//     {
//       const std::int64_t recv_col = recv_index[pos];
//       ++pos;
//       if (recv_col >= col_ranges[mpi_rank + 1]
//           or recv_col < col_ranges[mpi_rank])
//         col_ghost_map.insert({recv_col, -1});
//     }
//   }

//   // Unique numbering of ghost cols
//   int c = ncols_local;
//   for (auto& q : col_ghost_map)
//     q.second = c++;

//   std::vector<Eigen::Triplet<T>> mat_data;
//   for (int row = 0; row < nrows_local; ++row)
//     for (int j = Aouter[row]; j < Aouter[row + 1]; ++j)
//     {
//       int col = Ainner[j];
//       if (col >= ncols_local)
//       {
//         // Get remapped ghost column
//         std::int64_t global_col = col_ghosts[col - ncols_local];
//         auto it = col_ghost_map.find(global_col);
//         assert(it != col_ghost_map.end());
//         col = it->second;
//         assert(col >= ncols_local);
//       }

//       assert(row >= 0 and row < nrows_local);
//       assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));
//       mat_data.push_back(Eigen::Triplet<T>(row, col, Aval[j]));
//     }

//   // Add received data
//   pos = 0;
//   while (pos < (int)recv_index.size())
//   {
//     std::int64_t global_row = recv_index[pos];
//     assert(global_row >= row_ranges[mpi_rank]
//            and global_row < row_ranges[mpi_rank + 1]);
//     std::int32_t row = global_row - row_ranges[mpi_rank];
//     ++pos;
//     int nnz = recv_index[pos];
//     ++pos;
//     for (int k = 0; k < nnz; ++k)
//     {
//       const std::int64_t global_col = recv_index[pos];
//       const T val = recv_val[pos];
//       ++pos;
//       int col;
//       if (global_col >= col_ranges[mpi_rank + 1]
//           or global_col < col_ranges[mpi_rank])
//       {
//         auto it = col_ghost_map.find(global_col);
//         assert(it != col_ghost_map.end());
//         col = it->second;
//       }
//       else
//         col = global_col - col_ranges[mpi_rank];
//       assert(row >= 0 and row < nrows_local);
//       assert(col >= 0 and col < (int)(ncols_local + col_ghost_map.size()));
//       mat_data.push_back(Eigen::Triplet<T>(row, col, val));
//     }
//   }

//   // Rebuild sparse matrix
//   std::vector<std::int64_t> new_col_ghosts;
//   for (auto& q : col_ghost_map)
//     new_col_ghosts.push_back(q.first);

//   Eigen::SparseMatrix<T, Eigen::RowMajor> B(
//       nrows_local, ncols_local + new_col_ghosts.size());
//   B.setFromTriplets(mat_data.begin(), mat_data.end());

//   std::shared_ptr<spmv::L2GMap> col_map
//       = std::make_shared<spmv::L2GMap>(comm, ncols_local, new_col_ghosts);
//   std::shared_ptr<spmv::L2GMap> row_map = std::make_shared<spmv::L2GMap>(
//       comm, nrows_local, std::vector<std::int64_t>());

//   spmv::Matrix<T> b(B, col_map, row_map);
//   return b;
// }
// } // namespace spmv
