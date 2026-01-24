#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Инициализация распределенной среды
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Кто я?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Сколько нас всего?

    long N = 10000000; // Общий размер задачи
    double local_sum = 0, global_sum = 0;

    double start = MPI_Wtime(); // [Лекция 10, стр. 12] Замер времени в MPI

    // Распределяем работу (каждый считает свою часть)
    for (long i = 0; i < N / size; i++) {
        local_sum += 1.0;
    }

    // [Лекция 9, стр. 13] Коллективная операция — узкое место при плохой сети
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << ">>> Practice10Task4MPI " << size;
        std::cout << ">>> [MPI] Procs: " << size
            << " | Time: " << (end - start) << "s" << std::endl;
    }

    MPI_Finalize(); // Завершение работы
    return 0;
}