// part2_quick_sort_hybrid.cu — оптимизированная гибридная Quick Sort для GPU (быстрая работа в Colab)
#include <iostream>      // Библиотека для ввода-вывода: используется для вывода результатов и отладочной информации (Лекция №1: отладка гетерогенных программ)
#include <vector>        // Динамический контейнер vector на CPU (хосте) — хранение исходного и отсортированного массива (Лекция №3: хост-память для ввода-вывода)
#include <algorithm>     // Для std::swap (обмен элементов) и std::is_sorted (проверка корректности) — стандартные алгоритмы C++
#include <random>        // Для генерации случайных чисел: mt19937 и uniform_int_distribution — качественный ГСЧ для тестовых данных
#include <chrono>        // Для высокоточного измерения времени выполнения всей сортировки — сравнение CPU и GPU (Лекция №1: оценка производительности)
#include <cuda_runtime.h> // Основной заголовок CUDA: функции управления памятью, запуска ядер и синхронизации (Лекция №3: базовый API CUDA)

using namespace std;     // Упрощает код: позволяет использовать cout, vector, swap без префикса std:: — стандартная практика в учебных программах

// Макрос для проверки ошибок CUDA: при ошибке выводит сообщение с номером строки и завершает программу (Лекция №3: обязательная обработка ошибок в CUDA)
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// Ядро CUDA: каждый поток проверяет, меньше ли его элемент опорного, и атомарно увеличивает счётчик (Лекция №3: атомарные операции для безопасного доступа из тысяч потоков)
__global__ void count_less_than(int *data, int pivot, int *count, int n) {
    // Вычисляем глобальный индекс текущего потока в сетке (grid) — стандартная индексация в CUDA (Лекция №3: threadIdx, blockIdx, blockDim)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Проверяем, что индекс не выходит за границы массива — избежание ошибок чтения памяти
    if (idx < n) {
        // Если элемент меньше опорного — атомарно увеличиваем глобальный счётчик
        if (data[idx] < pivot) {
            atomicAdd(count, 1);  // Атомарная операция — безопасна для параллельного выполнения тысячами потоков (Лекция №3: разрешение гонок данных)
        }
    }
    // Каждый поток независимо проверяет свой элемент — демонстрирует массовую параллельность GPU (Лекция №3: простые операции на тысячах ядер)
}

// Рекурсивная функция Quick Sort: использует GPU только для подсчёта элементов меньше pivot (оптимизированный гибридный подход)
void quick_sort_hybrid(vector<int>& arr, int low, int high) {
    // Базовый случай рекурсии: если подмассив имеет 1 или 0 элементов — сортировка не нужна
    if (low >= high) return;
    
    // Выбираем опорный элемент (pivot) — последний элемент подмассива (стандартный выбор в Quick Sort)
    int pivot = arr[high];
    
    // Размер текущего подмассива
    int n = high - low + 1;
    
    // Указатели на массивы в глобальной памяти GPU
    int *d_data = nullptr;   // Для копии подмассива
    int *d_count = nullptr;  // Для счётчика элементов меньше pivot
    
    // Выделяем память на GPU для данных и счётчика (Лекция №3: cudaMalloc — явное управление device-памятью)
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    
    // Копируем подмассив с CPU на GPU — bottleneck в гетерогенных программах (Лекция №1: передача данных между устройствами)
    CUDA_CHECK(cudaMemcpy(d_data, &arr[low], n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Инициализируем счётчик нулём на GPU
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // Настраиваем конфигурацию запуска ядра: 256 потоков на блок — хороший баланс для большинства GPU (Лекция №3: выбор размера блока влияет на occupancy)
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);  // Округляем количество блоков вверх
    
    // Запускаем ядро: каждый поток проверяет свой элемент и обновляет счётчик
    count_less_than<<<blocks, threads>>>(d_data, pivot, d_count, n);
    
    // Ждём завершения ядра — синхронизация необходима перед чтением результата (Лекция №3: cudaDeviceSynchronize)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Копируем счётчик с GPU на CPU
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Вычисляем позицию pivot: все элементы меньше pivot должны быть слева от неё
    int pi = low + h_count;
    
    // Ручная перестановка элементов на CPU по вычисленной позиции (быстро для одного уровня)
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);  // Ставим pivot на правильное место
    
    // Освобождаем память на GPU — обязательная практика (Лекция №3: избежание утечек памяти)
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_count));
    
    // Рекурсивно сортируем левую часть (меньше pivot) — на CPU
    quick_sort_hybrid(arr, low, pi - 1);
    
    // Рекурсивно сортируем правую часть (больше или равно pivot) — на CPU
    quick_sort_hybrid(arr, pi + 1, high);
    
    // Гибридный подход: GPU ускоряет подсчёт (параллельные сравнения), CPU — рекурсию и перестановку (Лекция №1: эффективное распределение задач между CPU и GPU)
}

int main() {  // Главная функция на CPU — управляет всей программой (Лекция №3: CPU отвечает за последовательные части и запуск GPU)
    const int N = 1000000;  // Размер массива — 1 миллион элементов: достаточно большой для демонстрации ускорения на GPU
    vector<int> arr(N);     // Массив на CPU для хранения исходных и отсортированных данных
    
    cout << "Part 2: Optimized Hybrid Quick Sort (GPU count + CPU partition)\n";  // Заголовок части
    cout << "Array size: " << N << " elements\n";  // Вывод размера массива
    cout << "Generating random data...\n";  // Сообщение о генерации данных
    
    // Генератор случайных чисел
    mt19937 gen(time(nullptr));  // Seed от текущего времени — разные данные при каждом запуске
    uniform_int_distribution<int> dist(1, 1000000);  // Диапазон значений
    generate(arr.begin(), arr.end(), [&]() { return dist(gen); });  // Заполнение массива
    
    cout << "Starting hybrid sorting...\n";  // Начало сортировки
    
    // Замер времени всей сортировки
    auto start = chrono::high_resolution_clock::now();
    
    // Запуск оптимизированной гибридной Quick Sort
    quick_sort_hybrid(arr, 0, N - 1);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time = end - start;  // Общее время выполнения
    
    // Проверка корректности результата
    bool sorted = std::is_sorted(arr.begin(), arr.end());
    
    // Вывод результатов на английском
    cout << "\nSorting completed!\n";
    cout << "Execution time: " << time.count() << " seconds\n";
    cout << "Sorting correct: " << (sorted ? "Yes" : "No") << "\n";
    cout << "Note: Optimized hybrid approach uses GPU for parallel counting and CPU for recursion (Lecture #1, #3)\n";
    
    return 0;  // Успешное завершение программы
}