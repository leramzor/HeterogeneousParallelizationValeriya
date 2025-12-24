#include <iostream>
#include <chrono>    // для замера времени
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    //За основу взяты материалы из Лекция №1 (стр. 15): последовательные алгоритмы на больших массивах работают медленно,
    // что мотивирует переход к параллельному программированию.
    const int SIZE = 1000000;  // Массив из 1 000 000 элементов

    // Динамическое выделение большого массива
    int* arr = new int[SIZE];

    // Заполнение случайными значениями
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < SIZE; ++i) {
        arr[i] = rand() % 1000000;
    }

    // Замер времени выполнения последовательного алгоритма
    auto start = chrono::high_resolution_clock::now();

    // Последовательный поиск минимума и максимума
    int min_val = arr[0];
    int max_val = arr[0];
    for (int i = 1; i < SIZE; ++i) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    // Вывод результатов на английском
    cout << "Minimum value: " << min_val << endl;
    cout << "Maximum value: " << max_val << endl;
    cout << "Sequential execution time: " << duration.count() << " ms" << endl;

    // Освобождение памяти
    delete[] arr;

    return 0;
}