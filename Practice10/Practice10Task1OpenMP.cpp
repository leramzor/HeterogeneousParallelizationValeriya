#include <iostream>
#include <vector>
#include <omp.h> // Библиотека для параллельных вычислений на CPU [Лекция 10, стр. 2]

int main() {
    const int N = 100000000; // Определяем размер массива (100 млн элементов)
    std::vector<double> data(N, 1.5); // Выделяем память и заполняем массив значениями
    double sum = 0.0; // Переменная для хранения итоговой суммы

    // Проверяем производительность для разного числа потоков
    for (int num_threads : {1, 2, 4, 8}) {
        omp_set_num_threads(num_threads); // Устанавливаем количество ядер для расчета

        double start_time = omp_get_wtime(); // [Лекция 10, стр. 12] Точный замер времени начала

#pragma omp parallel for reduction(+:sum) // Параллельный цикл с защитой переменной sum
        for (int i = 0; i < N; i++) {
            sum += data[i]; // Каждый поток считает свою часть суммы
        }

        double end_time = omp_get_wtime(); // Фиксация времени окончания

        // Вывод результатов на английском (как требовалось ранее)
        std::cout << "Practice 10 Task1 ";
        std::cout << ">>> [OPENMP] Threads: " << num_threads
            << " | Time: " << (end_time - start_time) << "s" << std::endl;
    }
    // Вывод: чем больше потоков, тем меньше время, но ускорение не линейно из-за затрат на создание потоков.
    return 0;
}
