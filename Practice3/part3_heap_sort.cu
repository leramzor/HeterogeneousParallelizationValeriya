// part3_heap_sort.cu
#include <iostream>      // Библиотека для ввода-вывода: используется для вывода результатов и отладочной информации о выполнении (Лекция №1: отладка в гетерогенных программах)
#include <vector>        // Динамический контейнер vector на CPU (хосте) — хранение исходного и отсортированного массива (Лекция №3: хост-память для ввода-вывода)
#include <algorithm>     // Для std::is_sorted — проверка корректности сортировки после выполнения на GPU
#include <random>        // Для генерации случайных чисел: mt19937 и uniform_int_distribution — качественный ГСЧ для тестовых данных
#include <chrono>        // Для высокоточного измерения времени выполнения на GPU — важно для сравнения производительности (Лекция №1: оценка ускорения)
#include <cuda_runtime.h> // Основной заголовок CUDA: функции управления памятью, запуска ядер и синхронизации (Лекция №3: базовый API CUDA)

using namespace std;     // Упрощает код: позволяет использовать cout, vector, swap без префикса std:: — стандартная практика в учебных примерах

// Макрос для проверки ошибок CUDA: при ошибке выводит сообщение с номером строки и завершает программу (Лекция №3: обязательная обработка ошибок)
#define CUDA_CHECK(err) do { \
    cudaError_t local_err = (err); \
    if (local_err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(local_err) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// Ядро CUDA для параллельного heapify: каждый поток обрабатывает свой узел кучи (Лекция №3: массовая параллельность для независимых операций)
__global__ void heapify_kernel(int *arr, int n, int i) {
    // Вычисляем глобальный индекс текущего потока — стандартная индексация в CUDA (Лекция №3: threadIdx, blockIdx, blockDim)
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Обрабатываем только родительские узлы (индексы < n/2) — листья не нуждаются в heapify
    if (idx < n / 2) {
        // Текущий узел — кандидат на максимум
        int largest = idx;
        
        // Вычисляем индексы левого и правого потомков
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        
        // Сравниваем с левым потомком: если он существует и больше текущего максимума — обновляем
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        // Сравниваем с правым потомком: если он существует и больше текущего максимума — обновляем
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        // Если максимум не в корне поддерева — меняем элементы местами
        if (largest != idx) {
            // Обмен значениями — атомарная операция внутри потока
            int temp = arr[idx];
            arr[idx] = arr[largest];
            arr[largest] = temp;
        }
    }
    // Каждый поток независимо выполняет heapify для своего поддерева — демонстрирует параллелизм на GPU (Лекция №3: тысячи потоков для независимых задач)
}

// Функция Heap Sort на GPU: параллельное построение кучи и последовательное извлечение максимума
void heap_sort_gpu(vector<int>& arr) {
    // Размер массива
    int n = arr.size();
    
    // Указатель на массив в глобальной памяти GPU
    int *d_arr = nullptr;
    
    cout << "Part 3: Heap Sort on GPU (parallel heapify)\n";  // Заголовок части
    cout << "Array size: " << n << " elements\n";  // Вывод размера массива
    cout << "Allocating GPU memory...\n";  // Сообщение о выделении памяти
    
    // Выделяем память на GPU для всего массива (Лекция №3: cudaMalloc — явное управление device-памятью)
    CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int)));
    
    cout << "Copying data from CPU to GPU...\n";  // Копирование данных — bottleneck в гетерогенных программах (Лекция №1)
    CUDA_CHECK(cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Конфигурация запуска ядер: 256 потоков на блок — хороший баланс для большинства GPU (Лекция №3: выбор размера блока)
    dim3 threads(256);
    dim3 blocks((n / 2 + threads.x - 1) / threads.x);  // Округляем количество блоков для покрытия всех родительских узлов
    
    cout << "Grid configuration: " << blocks.x << " blocks x " << threads.x << " threads\n\n";  // Вывод конфигурации запуска
    
    cout << "Building max-heap in parallel...\n";  // Начало построения кучи
    
    // Параллельное построение кучи: начинаем с последнего родительского узла и идём вверх
    for (int i = n / 2 - 1; i >= 0; --i) {
        // Запускаем ядро для heapify от узла i
        heapify_kernel<<<blocks, threads>>>(d_arr, n, i);
        CUDA_CHECK(cudaGetLastError());  // Проверка ошибок запуска ядра
        CUDA_CHECK(cudaDeviceSynchronize());  // Синхронизация — ждём завершения всех потоков (Лекция №3: необходима для корректности)
    }
    
    cout << "Max-heap built. Extracting elements...\n";  // Куча построена — начинаем извлечение
    
    // Последовательное извлечение максимума (корня) с параллельным восстановлением кучи
    for (int i = n - 1; i > 0; --i) {
        // Копируем корень (максимум) на CPU для обмена
        int root;
        CUDA_CHECK(cudaMemcpy(&root, d_arr, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Копируем последний элемент на CPU
        int last;
        CUDA_CHECK(cudaMemcpy(&last, d_arr + i, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Меняем корень и последний элемент на GPU
        CUDA_CHECK(cudaMemcpy(d_arr, &last, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_arr + i, &root, sizeof(int), cudaMemcpyHostToDevice));
        
        // Параллельно восстанавливаем свойство кучи для уменьшенного массива (размер i)
        heapify_kernel<<<blocks, threads>>>(d_arr, i, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    cout << "Copying sorted array back to CPU...\n";  // Копирование результата
    CUDA_CHECK(cudaMemcpy(arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    cout << "Freeing GPU memory...\n";  // Освобождение памяти — обязательная практика (Лекция №3: избежание утечек)
    CUDA_CHECK(cudaFree(d_arr));
}

// Главная функция на CPU — управляет всей программой (Лекция №3: CPU отвечает за последовательные части и запуск GPU)
int main() {
    const int N = 524288;  // Размер массива — 524288 элементов (степень 2, достаточно большой для демонстрации)
    vector<int> arr(N);     // Массив на CPU для хранения данных
    
    cout << "Generating random data...\n";  // Генерация тестовых данных
    
    // Генератор случайных чисел
    mt19937 gen(time(nullptr));  // Seed от текущего времени
    uniform_int_distribution<int> dist(1, 1000000);  // Диапазон значений
    generate(arr.begin(), arr.end(), [&]() { return dist(gen); });  // Заполнение массива
    
    cout << "Starting Heap Sort on GPU...\n";  // Начало сортировки
    
    // Замер времени всей сортировки
    auto start = chrono::high_resolution_clock::now();
    
    // Запуск Heap Sort на GPU
    heap_sort_gpu(arr);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time = end - start;  // Общее время выполнения
    
    // Проверка корректности результата
    bool sorted = std::is_sorted(arr.begin(), arr.end());
    
    // Вывод результатов на английском
    cout << "\nHeap Sort completed!\n";
    cout << "Execution time: " << time.count() << " seconds\n";
    cout << "Sorting correct: " << (sorted ? "Yes" : "No") << "\n";
    cout << "Note: Parallel heapify on GPU demonstrates massive parallelism for independent operations (Lecture #3)\n";
    
    return 0;  // Успешное завершение программы
}