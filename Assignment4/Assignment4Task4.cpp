#include <mpi.h>    // Лекция №9: Стандарт передачи сообщений
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // Лекция №9: Инициализация MPI среды
    MPI_Init(&argc, &argv);

    int rank, size;
    // Определение ID процесса (rank) и их общего количества (size)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    // Лекция №10: Декомпозиция данных (Data Partitioning)
    int local_n = N / size;
    std::vector<int> local_data(local_n, 1); // Локальная часть массива

    double start_time = MPI_Wtime(); // Лекция №10: Замер времени в MPI

    // Локальное вычисление на каждом узле/процессе
    long long local_sum = 0;
    for (int val : local_data) local_sum += val;

    long long global_sum = 0;
    // Лекция №9: Коллективная операция MPI_Reduce. 
    // Собирает local_sum со всех процессов, суммирует их и отправляет результат процессу 0.
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    // Вывод результата только мастером (Rank 0)
    if (rank == 0) {
        std::cout << "--- MPI RESULTS ---" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Global Sum: " << global_sum << std::endl;
        std::cout << "Execution Time: " << end_time - start_time << " s" << std::endl;
    }

    // Лекция №9: Обязательное завершение работы MPI
    MPI_Finalize();
    return 0;
}
