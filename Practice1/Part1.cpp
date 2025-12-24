#include <iostream>      // Для вывода в консоль
#include <vector>        // Для динамического контейнера vector
#include <random>        // Для генерации случайных чисел
#include <chrono>        // Для измерения времени выполнения
#include <omp.h>         // Для директив OpenMP

using namespace std;     // Чтобы не писать std:: перед cout, vector и т.д.

int main() {
    // Размер массива — 10 миллионов элементов (для заметного эффекта параллелизации)
    const int N = 10000000;
    // Создаём динамический массив с помощью vector
    vector<int> arr(N);

    // Генератор случайных чисел Mersenne Twister
    mt19937 gen(static_cast<unsigned>(time(nullptr)));
    // Равномерное распределение от 1 до 100
    uniform_int_distribution<int> dist(1, 100);

    // Заполняем массив случайными числами от 1 до 100
    for (int i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // Переменные для последовательной версии
    int seq_min = arr[0];   // Начальный минимум
    int seq_max = arr[0];   // Начальный максимум

    // Замер времени начала последовательной версии
    auto start_seq = chrono::high_resolution_clock::now();

    // Последовательный поиск минимума и максимума
    for (int i = 1; i < N; ++i) {
        if (arr[i] < seq_min) seq_min = arr[i];
        if (arr[i] > seq_max) seq_max = arr[i];
    }

    // Замер времени окончания последовательной версии
    auto end_seq = chrono::high_resolution_clock::now();
    // Вычисление времени выполнения
    chrono::duration<double> seq_time = end_seq - start_seq;

    // Переменные для параллельной версии
    int par_min = arr[0];
    int par_max = arr[0];

    // Замер времени начала параллельной версии
    auto start_par = chrono::high_resolution_clock::now();

    // Параллельный цикл с reduction для min и max
#pragma omp parallel for reduction(min:par_min) reduction(max:par_max)
    for (int i = 0; i < N; ++i) {
        if (arr[i] < par_min) par_min = arr[i];
        if (arr[i] > par_max) par_max = arr[i];
    }

    // Замер времени окончания параллельной версии
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = end_par - start_par;

    // Вывод результатов
    cout << "Part 1: Working with arrays\n";
    cout << "Array size: " << N << endl;
    cout << "Sequential version:\n";
    cout << "  Minimum: " << seq_min << ", Maximum: " << seq_max << endl;
    cout << "  Time: " << seq_time.count() << " seconds\n\n";

    cout << "Parallel version (OpenMP):\n";
    cout << "  Minimum: " << par_min << ", Maximum: " << par_max << endl;
    cout << "  Time: " << par_time.count() << " seconds\n\n";

    cout << "Speedup: " << seq_time.count() / par_time.count() << "x\n";

    return 0;
}