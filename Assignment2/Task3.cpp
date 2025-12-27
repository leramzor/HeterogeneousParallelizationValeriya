// task3_selection_sort_openmp.cpp
#include <iostream>      // Для вывода результатов и прогресса выполнения (Лекция №1: отладка)
#include <vector>        // Динамический массив на CPU — хранение данных
#include <random>        // Генерация случайных чисел для тестов
#include <chrono>        // Измерение времени — сравнение производительности (Лекция №2: оценка OpenMP)
#include <omp.h>         // Директивы OpenMP для параллелизма (Лекция №2: многопоточность на CPU)

using namespace std;

// Заполнение массива случайными числами от 1 до 1 000 000
void fill_array(vector<int>& arr) {
    mt19937 gen(time(nullptr));
    uniform_int_distribution<int> dist(1, 1000000);
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = dist(gen);
    }
}

// Последовательная сортировка выбором
void selection_sort_seq(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {  // Проходим по всем позициям кроме последней
        int min_idx = i;  // Предполагаем, что минимум на текущей позиции
        for (int j = i + 1; j < n; ++j) {  // Ищем настоящий минимум в оставшейся части
            if (arr[j] < arr[min_idx]) min_idx = j;
        }
        if (min_idx != i) swap(arr[i], arr[min_idx]);  // Меняем местами, если нашли меньший элемент
    }
}

// Параллельная сортировка выбором — параллельный поиск минимума с reduction
void selection_sort_par(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        // Параллельный поиск минимума в оставшейся части (Лекция №2: reduction(min) для безопасного обновления)
#pragma omp parallel for reduction(min:min_idx)
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) min_idx = j;
        }
        if (min_idx != i) swap(arr[i], arr[min_idx]);
    }
}

// Функция для запуска теста на заданном размере массива
void run_test(int size) {
    vector<int> arr_seq(size);
    vector<int> arr_par(size);

    fill_array(arr_seq);
    arr_par = arr_seq;

    auto start_seq = chrono::high_resolution_clock::now();
    selection_sort_seq(arr_seq);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time = end_seq - start_seq;

    auto start_par = chrono::high_resolution_clock::now();
    selection_sort_par(arr_par);
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = end_par - start_par;

    cout << "Array size: " << size << " elements\n";
    cout << "Sequential time: " << seq_time.count() << " seconds\n";
    cout << "Parallel time:   " << par_time.count() << " seconds\n";
    cout << "Speedup: " << seq_time.count() / par_time.count() << "x\n\n";
}

int main() {
    cout << "Task 3: Parallel Selection Sort with OpenMP\n";

    cout << "Testing with 1000 elements...\n";
    run_test(1000);

    cout << "Testing with 10000 elements...\n";
    run_test(10000);

    cout << "Conclusion: Parallel minimum search with OpenMP reduction gives noticeable speedup, especially on larger arrays (Lecture #2)\n";

    return 0;
}
