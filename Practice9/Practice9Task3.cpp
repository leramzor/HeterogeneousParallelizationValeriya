#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 200; // Количество узлов в графе
    int rows_per_proc = N / size;
    std::vector<std::vector<int>> local_mat(rows_per_proc, std::vector<int>(N, 500));

    if (rank == 0) {
        std::cout << ">>> Practice9Task3" << std::endl;
        std::cout << ">>> [ALGORITHM] Floyd-Warshall Parallel Computation" << std::endl;
        std::cout << ">>> [NODES] Graph size: " << N << " vertices" << std::endl;
    }

    double start_time = MPI_Wtime();

    for (int k = 0; k < N; k++) {
        int owner = k / rows_per_proc;
        std::vector<int> row_k(N);

        if (rank == owner) row_k = local_mat[k % rows_per_proc];

        // [Лекция 9] Синхронизация: каждый процесс получает строку 'k' для обновления своих путей
        MPI_Bcast(row_k.data(), N, MPI_INT, owner, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                local_mat[i][j] = std::min(local_mat[i][j], local_mat[i][k] + row_k[j]);
            }
        }

        if (rank == 0 && k % (N / 4) == 0) {
            std::cout << ">>> [PROGRESS] Iteration " << k << "/" << N << " completed." << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << ">>> [DONE] Shortest paths computed successfully." << std::endl;
        std::cout << ">>> [TIME] Elapsed: " << (MPI_Wtime() - start_time) << " seconds on " << size << " procs." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
