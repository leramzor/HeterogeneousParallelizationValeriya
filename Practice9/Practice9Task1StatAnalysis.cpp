#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // [Лекция 9, стр. 4]
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    std::vector<double> data;
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    // [Лекция 9, часть 2] Балансировка нагрузки: расчет порций для каждого процесса
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = N / size + (i < (N % size) ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    if (rank == 0) {
        std::cout << ">>>Initializing array with " << N << " elements..." << std::endl;
        data.resize(N);
        for (int i = 0; i < N; i++) data[i] = rand() % 100;
    }

    std::vector<double> local_data(sendcounts[rank]);
    double start_time = MPI_Wtime(); // [Лекция 10, стр. 12]

    // [Лекция 9, стр. 12] Распределение данных: Scatterv
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_data.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0, local_sq_sum = 0;
    for (double x : local_data) {
        local_sum += x;
        local_sq_sum += x * x;
    }

    // [Лекция 9, стр. 13] Редукция: сбор сумм и квадратов
    double global_sum, global_sq_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        double mean = global_sum / N;
        double std_dev = sqrt((global_sq_sum / N) - (mean * mean));
        std::cout << "\nPractice9Task1" << std::endl;
        std::cout << "\n================ CONFIGURATION ================" << std::endl;
        std::cout << "Processes Count (NP): " << size << std::endl;
        std::cout << "Array Size (N): " << N << std::endl;
        std::cout << "=================== RESULTS ===================" << std::endl;
        std::cout << "Calculated Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
        std::cout << "Standard Deviation: " << std_dev << std::endl;
        std::cout << "Execution Time: " << (end_time - start_time) << " seconds" << std::endl;
        std::cout << "===============================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}