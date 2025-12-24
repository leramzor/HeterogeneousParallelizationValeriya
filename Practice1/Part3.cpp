#include <iostream>      // Для вывода
#include <random>        // Для генерации случайных чисел
#include <chrono>        // Для измерения времени
#include <omp.h>         // Для OpenMP

using namespace std;

int main() {
    // Размер массива
    const int N = 10000000;

    // Динамическое выделение массива через указатель
    int* arr = new int[N];

    // Генератор случайных чисел
    mt19937 gen(static_cast<unsigned>(time(nullptr)));
    uniform_int_distribution<int> dist(1, 100);

    // Заполнение массива случайными числами
    for (int i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // Последовательное вычисление среднего
    double seq_sum = 0.0;
    auto start_seq = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        seq_sum += arr[i];
    }
    double seq_avg = seq_sum / N;

    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time = end_seq - start_seq;

    // Параллельное вычисление среднего
    double par_sum = 0.0;
    auto start_par = chrono::high_resolution_clock::now();

    // Параллельное суммирование с reduction
#pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < N; ++i) {
        par_sum += arr[i];
    }
    double par_avg = par_sum / N;

    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = end_par - start_par;

    // Вывод результатов на английском
    cout << "Part 3: Dynamic memory and average calculation\n";
    cout << "Array size: " << N << endl;
    cout << "Sequential average: " << seq_avg << " (time: " << seq_time.count() << " seconds)\n";
    cout << "Parallel average:   " << par_avg << " (time: " << par_time.count() << " seconds)\n";
    cout << "Speedup: " << seq_time.count() / par_time.count() << "x\n";

    // Освобождение динамически выделенной памяти
    delete[] arr;

    return 0;
}