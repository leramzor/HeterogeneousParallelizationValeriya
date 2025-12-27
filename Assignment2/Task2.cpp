// task2_min_max_openmp.cpp
#include <iostream>      // Для вывода результатов и отладочной информации (Лекция №1: отладка параллельных программ)
#include <vector>        // Динамический контейнер vector — удобен для хранения массива на CPU (Лекция №2: работа с большими данными)
#include <random>        // Для генерации случайных чисел: mt19937 и uniform_int_distribution — качественный ГСЧ
#include <chrono>        // Для высокоточного измерения времени выполнения — сравнение последовательной и параллельной версий (Лекция №2: оценка производительности OpenMP)
#include <omp.h>         // Заголовок OpenMP — необходим для директив параллелизма (Лекция №2: основы OpenMP)

using namespace std;     // Упрощает код: позволяет использовать cout, vector без префикса std::

// Функция заполнения массива случайными числами от 1 до 1 000 000
void fill_array(vector<int>& arr) {
    // Генератор Mersenne Twister с seed от текущего времени — обеспечивает разные данные при каждом запуске
    mt19937 gen(time(nullptr));
    // Равномерное распределение целых чисел от 1 до 1 000 000
    uniform_int_distribution<int> dist(1, 1000000);
    // Заполняем весь массив случайными значениями
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = dist(gen);
    }
    // Случайные данные создают худший случай для поиска min/max — хорошая нагрузка для теста (Лекция №2: тестирование параллельных алгоритмов)
}

int main() {
    const int N = 10000;  // Размер массива — 10 000 элементов, как требуется в задаче 2
    vector<int> arr(N);   // Массив на CPU для хранения данных

    cout << "Task 2: Min/Max search with OpenMP\n";
    cout << "Array size: " << N << " elements\n";
    cout << "Generating random data...\n";

    fill_array(arr);  // Заполняем массив случайными числами

    // --- Последовательная версия ---
    cout << "Running sequential version...\n";
    auto start_seq = chrono::high_resolution_clock::now();  // Замер времени начала последовательной версии

    int seq_min = arr[0];  // Инициализируем минимум первым элементом
    int seq_max = arr[0];  // Инициализируем максимум первым элементом

    // Последовательный поиск min и max по всему массиву
    for (int i = 1; i < N; ++i) {
        if (arr[i] < seq_min) seq_min = arr[i];  // Обновляем минимум, если найден меньший элемент
        if (arr[i] > seq_max) seq_max = arr[i];  // Обновляем максимум, если найден больший элемент
    }

    auto end_seq = chrono::high_resolution_clock::now();  // Замер времени окончания
    chrono::duration<double> seq_time = end_seq - start_seq;  // Вычисляем время выполнения последовательной версии

    // --- Параллельная версия с OpenMP ---
    cout << "Running parallel version with OpenMP...\n";
    auto start_par = chrono::high_resolution_clock::now();  // Замер времени начала параллельной версии

    int par_min = arr[0];  // Начальное значение для параллельного минимума (будет обновляться через reduction)
    int par_max = arr[0];  // Начальное значение для параллельного максимума

    // Параллельный цикл с reduction для безопасного поиска min и max (Лекция №2: reduction — решение проблемы race condition)
#pragma omp parallel for reduction(min:par_min) reduction(max:par_max)
    for (int i = 0; i < N; ++i) {
        if (arr[i] < par_min) par_min = arr[i];  // Каждый поток обновляет свою локальную копию минимума
        if (arr[i] > par_max) par_max = arr[i];  // Каждый поток обновляет свою локальную копию максимума
    }

    auto end_par = chrono::high_resolution_clock::now();  // Замер времени окончания
    chrono::duration<double> par_time = end_par - start_par;  // Вычисляем время выполнения параллельной версии

    // Вывод результатов на английском языке
    cout << "\nResults:\n";
    cout << "Sequential version:\n";
    cout << "  Min: " << seq_min << ", Max: " << seq_max << "\n";
    cout << "  Time: " << seq_time.count() << " seconds\n\n";

    cout << "Parallel version (OpenMP):\n";
    cout << "  Min: " << par_min << ", Max: " << par_max << "\n";
    cout << "  Time: " << par_time.count() << " seconds\n\n";

    cout << "Speedup: " << seq_time.count() / par_time.count() << "x\n";
    cout << "Conclusion: OpenMP with reduction provides significant speedup for independent operations like min/max search on multi-core CPU (Lecture #2)\n";

    return 0;  // Успешное завершение программы
}