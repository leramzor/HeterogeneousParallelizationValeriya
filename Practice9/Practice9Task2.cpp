#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 8; // Размер матрицы
    int rows_per_proc = N / size; // [Лекция 10] Декомпозиция данных

    std::vector<std::vector<double>> my_rows(rows_per_proc, std::vector<double>(N + 1, 1.0));

    if (rank == 0) {
        std::cout << "Practice9Task2" << std::endl;
        std::cout << ">>>Starting Distributed Gaussian Elimination..." << std::endl;
        std::cout << ">>> [INFO] System size: " << N << "x" << N << std::endl;
    }

    double start_time = MPI_Wtime();

    for (int k = 0; k < N; k++) {
        int root_process = k / rows_per_proc;
        std::vector<double> pivot_row(N + 1);

        if (rank == root_process) {
            pivot_row = my_rows[k % rows_per_proc];
            // std::cout << "[Step " << k << "] Rank " << rank << " broadcasting pivot row." << std::endl;
        }

        // [Лекция 9, стр. 11] Коллективная рассылка ведущей строки
        MPI_Bcast(pivot_row.data(), N + 1, MPI_DOUBLE, root_process, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;
            if (global_i > k) {
                double factor = my_rows[i][k] / pivot_row[k];
                for (int j = k; j <= N; j++) {
                    my_rows[i][j] -= factor * pivot_row[j];
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << ">>> [SUCCESS] Forward elimination finished." << std::endl;
        std::cout << ">>> [PERF] Total MPI Time: " << (MPI_Wtime() - start_time) << " s." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
