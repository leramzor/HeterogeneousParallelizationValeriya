#include <iostream>      // Для вывода результатов в консоль
#include <vector>        // Динамический массив vector
#include <random>        // Генерация случайных чисел
#include <chrono>        // Измерение времени
#include <omp.h>         // Поддержка OpenMP
#include <cstddef>       // Для size_t (явно, хотя обычно включается через другие заголовки)

using namespace std;     // Используем std без префикса

// Функция заполнения массива случайными числами
void fill_array(vector<int>& arr) {                     // По ссылке, чтобы изменить оригинал
    mt19937 gen(static_cast<unsigned>(time(nullptr)));  // Генератор с seed от времени
    uniform_int_distribution<int> dist(1, 100000);     // Диапазон значений
    for (size_t i = 0; i < arr.size(); ++i) {          // По всем элементам
        arr[i] = dist(gen);                            // Присваиваем случайное значение
    }                                                  // Конец цикла
}                                                      // Конец функции

// Последовательная сортировка вставками
void insertion_sort_seq(vector<int>& arr) {             // По ссылке
    size_t n = arr.size();                             // Размер массива
    for (size_t i = 1; i < n; ++i) {                    // Начинаем со второго элемента (первый считается отсортированным)
        int key = arr[i];                              // Сохраняем текущий элемент — он будет вставлен на правильное место
        int j = static_cast<int>(i) - 1;               // Индекс предыдущего элемента (используем int для простоты)
        while (j >= 0 && arr[j] > key) {               // Пока не дошли до начала и предыдущий элемент больше key
            arr[j + 1] = arr[j];                       // Сдвигаем больший элемент вправо
            --j;                                       // Переходим к следующему левому элементу
        }                                              // Конец while — нашли позицию для вставки
        arr[j + 1] = key;                              // Вставляем сохранённый key на правильное место
    }                                                  // Конец внешнего цикла
}                                                      // Конец функции

// Параллельная версия — внешний цикл распараллелен с защитой от гонок данных
void insertion_sort_par(vector<int>& arr) {             // По ссылке
    size_t n = arr.size();                             // Размер массива

    // Параллелим внешний цикл с динамическим расписанием для балансировки нагрузки
    // Добавляем critical секцию, чтобы избежать race condition при одновременных вставках
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < static_cast<int>(n); ++i) {     // Параллельный цикл по i
        int key = arr[i];                              // Сохраняем текущий элемент

        // Критическая секция: только один поток может модифицировать массив одновременно
#pragma omp critical
        {
            int j = i - 1;                             // Индекс предыдущего элемента
            while (j >= 0 && arr[j] > key) {           // Сдвигаем элементы вправо
                arr[j + 1] = arr[j];                   // Перемещение
                --j;                                   // Двигаемся влево
            }                                          // Конец while
            arr[j + 1] = key;                          // Вставка на место
        }                                              // Конец critical секции
    }                                                  // Конец параллельного цикла

    // Примечание: даже с critical ускорение ограничено из-за последовательного доступа (практика №2)
}                                                      // Конец функции

int main() {                                           // Главная функция
    const size_t N = 10000;                            // Размер массива — 10 тысяч элементов
    vector<int> arr_seq(N);                            // Вектор для последовательной версии
    vector<int> arr_par(N);                            // Вектор для параллельной версии

    fill_array(arr_seq);                               // Заполняем случайными числами
    arr_par = arr_seq;                                 // Копируем для идентичных условий

    auto start_seq = chrono::high_resolution_clock::now();  // Начало замера последовательной версии
    insertion_sort_seq(arr_seq);                       // Выполнение последовательной сортировки
    auto end_seq = chrono::high_resolution_clock::now();    // Конец замера
    chrono::duration<double> seq_time = end_seq - start_seq; // Длительность последовательной

    auto start_par = chrono::high_resolution_clock::now();  // Начало замера параллельной версии
    insertion_sort_par(arr_par);                       // Выполнение параллельной сортировки
    auto end_par = chrono::high_resolution_clock::now();    // Конец замера
    chrono::duration<double> par_time = end_par - start_par; // Длительность параллельной

    cout << "Part 3: Insertion Sort\n";                // Заголовок части
    cout << "Array size: " << N << " elements\n";      // Вывод размера массива
    cout << "Sequential version time: " << seq_time.count() << " seconds\n";  // Время последовательной
    cout << "Parallel version time:   " << par_time.count() << " seconds\n";  // Время параллельной
    cout << "Speedup achieved: " << seq_time.count() / par_time.count() << "x\n";  // Ускорение
    cout << "Note: Limited or no speedup due to strong data dependencies (as stated in Practical Work 2)\n";  // Пояснение

    return 0;                                          // Успешное завершение программы
}                                                      // Конец main