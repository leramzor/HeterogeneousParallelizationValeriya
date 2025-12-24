#include <iostream>
#include <chrono>    // Для точного замера времени
#include <cstdlib>   // Для rand(), srand()
#include <ctime>     // Для time()
#include <omp.h>     // OpenMP — инструмент многопоточности на CPU (Лекция №2)

using namespace std;

int main() {
    // Лекция №1 (стр. 9–11): Последовательное программирование слишком медленно 
    // для обработки больших массивов данных. Параллельное программирование 
    // позволяет ускорить вычисления за счёт одновременной работы нескольких потоков.
    // Лекция №2 (Часть 1): Современные CPU многоядерные — 4, 8, 16 ядер и более.
    // Это позволяет эффективно использовать многопоточность.
    const long long SIZE = 5000000LL;  // 5 миллионов элементов — большой объём данных

    // Динамическое выделение массива типа double (для точности среднего значения)
    double* arr = new double[SIZE];

    // Инициализация генератора случайных чисел один раз
    srand(static_cast<unsigned>(time(nullptr)));

    // Заполнение массива случайными числами от 0.0 до 100.0
    // Имитация реальных данных (например, измерений или результатов моделирования)
    for (long long i = 0; i < SIZE; ++i) {
        arr[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }

    // === Последовательная версия ===
    // Лекция №1: последовательный подход — базовый, но медленный на больших данных
    auto start_seq = chrono::high_resolution_clock::now();

    double sum_seq = 0.0;
    for (long long i = 0; i < SIZE; ++i) {
        sum_seq += arr[i];  // Простое последовательное суммирование
    }
    double avg_seq = sum_seq / SIZE;  // Вычисление среднего

    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_seq = end_seq - start_seq;

    // === Параллельная версия с OpenMP ===
    // Лекция №2 (Часть 3): OpenMP — простой способ добавить параллелизм на CPU
    // #pragma omp parallel for — создаёт потоки и распределяет итерации цикла
    // reduction(+:sum_par) — механизм из лекции №2 (стр. 20)
    // Каждый поток имеет свою локальную копию sum_par
    // В конце OpenMP автоматически суммирует все локальные значения
    // Без reduction возникла бы race condition — несколько потоков одновременно
    // изменяли бы одну переменную sum_par → неверный результат
    double sum_par = 0.0;
    auto start_par = chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+:sum_par)
    for (long long i = 0; i < SIZE; ++i) {
        sum_par += arr[i];
    }

    double avg_par = sum_par / SIZE;

    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_par = end_par - start_par;

    // === Вывод результатов ===
    // Используем английский язык, чтобы избежать проблем с кодировкой в консоли Windows
    cout << "Sequential version:" << endl;
    cout << " Average = " << avg_seq << ", Time: " << dur_seq.count() << " ms" << endl;

    cout << "Parallel version (OpenMP + reduction):" << endl;
    cout << " Average = " << avg_par << ", Time: " << dur_par.count() << " ms" << endl;

    // Ускорение (speedup) — важный показатель эффективности параллелизации
    // Лекция №1: цель параллельного программирования — повышение производительности
    double speedup = dur_seq.count() / dur_par.count();
    cout << "Speedup: approximately " << speedup
        << "x (depends on CPU cores)" << endl;

    // Освобождение памяти
    // Важно! Без delete[] будет утечка памяти
    delete[] arr;

    return 0;
}