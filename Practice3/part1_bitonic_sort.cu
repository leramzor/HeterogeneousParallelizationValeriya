// part1_bitonic_sort.cu
#include <iostream>      // Библиотека для ввода-вывода: используется для вывода информации о выполнении программы в консоль (Лекция №1: отладка параллельных программ)
#include <vector>        // Динамический контейнер vector на CPU (хосте) — удобен для хранения больших массивов данных перед передачей на GPU (Лекция №3: различие хост- и device-памяти)
#include <algorithm>     // Для функции std::generate (заполнение массива) и std::is_sorted (проверка корректности сортировки) — стандартные алгоритмы C++
#include <random>        // Для генерации псевдослучайных чисел: mt19937 и uniform_int_distribution — качественный ГСЧ для тестовых данных
#include <chrono>        // Для высокоточного измерения времени выполнения на GPU — важно для оценки производительности параллельных алгоритмов (Лекция №1: сравнение CPU и GPU)
#include <cuda_runtime.h> // Основной заголовок CUDA: содержит функции управления памятью, запуска ядер и проверки ошибок (Лекция №3: базовые API CUDA)

using namespace std;     // Упрощает код: позволяет использовать cout, vector, chrono без префикса std:: — стандартная практика в учебных программах

// Макрос для проверки ошибок CUDA: если операция не удалась — выводит сообщение и завершает программу (Лекция №3: важность обработки ошибок в CUDA)
#define CUDA_CHECK(err) do { \
    cudaError_t local_err = (err); \
    if (local_err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(local_err) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// Ядро CUDA: выполняется параллельно тысячами потоков на GPU (Лекция №3: __global__ функция — точка входа с хоста на устройство)
__global__ void bitonic_sort_step(int *dev_array, int j, int k) {
    // Вычисляем глобальный индекс текущего потока: threadIdx.x — внутри блока, blockDim.x — размер блока, blockIdx.x — номер блока (Лекция №3: индексация потоков в CUDA)
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Вычисляем индекс парного элемента для сравнения с помощью побитовой операции XOR — ключевой приём в битонной сортировке
    unsigned int ixj = i ^ j;
    
    // Проверяем границы: парный индекс должен быть больше текущего и не выходить за пределы массива (избежание ошибок доступа к памяти)
    if (ixj > i) {
        // Определяем направление сравнения по значению бита k (Лекция №3: битонная последовательность чередует восходящие и нисходящие фазы)
        if ((i & k) == 0) {
            // Восходящая фаза: меньший элемент должен оказаться по меньшему индексу
            if (dev_array[i] > dev_array[ixj]) {
                // Обмен элементов местами — атомарная операция внутри потока
                int temp = dev_array[i];
                dev_array[i] = dev_array[ixj];
                dev_array[ixj] = temp;
            }
        } else {
            // Нисходящая фаза: больший элемент должен оказаться по меньшему индексу
            if (dev_array[i] < dev_array[ixj]) {
                int temp = dev_array[i];
                dev_array[i] = dev_array[ixj];
                dev_array[ixj] = temp;
            }
        }
    }
    // Каждый поток независимо обрабатывает свою пару — демонстрирует массовую параллельность GPU (Лекция №3: тысячи потоков выполняют простые операции одновременно)
}

int main() {  // Главная функция на CPU (хосте) — управляет всей программой (Лекция №3: гетерогенная модель: CPU управляет, GPU вычисляет)
    const int N = 1 << 16;  // Размер массива — 65536 элементов. Обязательно степень двойки для корректной работы битонной сортировки (алгоритмическое требование)
    vector<int> host_array(N);  // Массив на CPU: здесь хранятся исходные и конечные данные (Лекция №3: хост-память медленнее, но доступна для ввода-вывода)
    
    cout << "Part 1: Bitonic Merge Sort on GPU (CUDA)\n";  // Заголовок части — соответствует структуре практических работ
    cout << "Initializing array with " << N << " random elements...\n";  // Сообщение о начале инициализации
    
    // Генератор случайных чисел для создания тестовых данных
    mt19937 gen(time(nullptr));  // Инициализация seed текущим временем — обеспечивает разные данные при каждом запуске
    uniform_int_distribution<int> dist(1, 100000);  // Диапазон значений от 1 до 100000 — создаёт разнообразные данные
    generate(host_array.begin(), host_array.end(), [&]() { return dist(gen); });  // Заполнение массива случайными числами
    
    cout << "First 10 elements (before sorting): ";  // Вывод первых элементов для визуальной проверки исходного состояния
    for (int i = 0; i < 10 && i < N; ++i) cout << host_array[i] << " ";
    cout << "...\n";
    
    cout << "Last 10 elements (before sorting): ";  // Вывод последних элементов — помогает увидеть хаотичность данных
    for (int i = max(0, N - 10); i < N; ++i) cout << host_array[i] << " ";
    cout << "\n\n";
    
    int *dev_array = nullptr;  // Указатель на массив в глобальной памяти GPU (device)
    
    cout << "Allocating memory on GPU (" << N * sizeof(int) / 1024 / 1024 << " MB)...\n";  // Сообщение о выделении памяти
    CUDA_CHECK(cudaMalloc(&dev_array, N * sizeof(int)));  // Выделение памяти на GPU — критически важный шаг (Лекция №3: явное управление памятью)
    
    cout << "Copying data from CPU to GPU...\n";  // Копирование данных на устройство
    CUDA_CHECK(cudaMemcpy(dev_array, host_array.data(), N * sizeof(int), cudaMemcpyHostToDevice));  // Передача данных — bottleneck в гетерогенных программах (Лекция №1)
    
    // Конфигурация запуска ядер CUDA
    dim3 threads(256);  // 256 потоков на блок — хороший баланс для большинства GPU (Лекция №3: выбор размера блока влияет на occupancy)
    dim3 blocks((N + threads.x - 1) / threads.x);  // Вычисляем необходимое количество блоков с округлением вверх
    
    cout << "Launch configuration: " << blocks.x << " blocks × " << threads.x << " threads = "
         << blocks.x * threads.x << " total threads\n\n";  // Вывод конфигурации — демонстрирует масштабируемость
    
    cout << "Starting GPU sorting (bitonic phases)...\n";  // Начало основного вычисления на GPU
    auto start_gpu = chrono::high_resolution_clock::now();  // Замер времени начала выполнения на GPU
    
    int phase_count = 0;  // Счётчик выполненных фаз для отладки
    // Основной цикл битонной сортировки
    for (int k = 2; k <= N; k <<= 1) {  // k — размер битонной последовательности: 2, 4, 8, ..., N (логарифмическое количество уровней)
        for (int j = k >> 1; j > 0; j >>= 1) {  // j — расстояние между сравниваемыми элементами в текущей последовательности
            // Запуск ядра на GPU с заданной конфигурацией
            bitonic_sort_step<<<blocks, threads>>>(dev_array, j, k);
            CUDA_CHECK(cudaGetLastError());  // Проверка ошибок запуска ядра
            CUDA_CHECK(cudaDeviceSynchronize());  // Синхронизация: ждём завершения всех потоков перед следующей фазой (Лекция №3: необходима для корректности)
            phase_count++;  // Увеличиваем счётчик фаз
            
            // Вывод прогресса каждые 8 фаз — чтобы консоль не засорялась, но было видно движение
            if (phase_count % 8 == 0 || j == 1) {
                cout << "  Completed phase " << phase_count << " (k=" << k << ", j=" << j << ")\n";
            }
        }
    }
    
    auto end_gpu = chrono::high_resolution_clock::now();  // Замер времени окончания
    chrono::duration<double, milli> gpu_time_ms = end_gpu - start_gpu;  // Время в миллисекундах для точности
    
    cout << "\nGPU sorting completed in " << gpu_time_ms.count() << " ms\n";  // Итоговое время выполнения на GPU
    
    cout << "Copying sorted data back to CPU...\n";  // Копирование результата обратно
    CUDA_CHECK(cudaMemcpy(host_array.data(), dev_array, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    cout << "Freeing GPU memory...\n";  // Освобождение памяти — обязательная практика (Лекция №3: избежание утечек)
    CUDA_CHECK(cudaFree(dev_array));
    
    // Проверка корректности результата
    bool is_sorted = std::is_sorted(host_array.begin(), host_array.end());  // Используем стандартный алгоритм для верификации
    
    // Вывод отсортированных элементов для визуального подтверждения
    cout << "\nFirst 10 elements (after sorting): ";
    for (int i = 0; i < 10 && i < N; ++i) cout << host_array[i] << " ";
    cout << "...\n";
    
    cout << "Last 10 elements (after sorting): ";
    for (int i = max(0, N - 10); i < N; ++i) cout << host_array[i] << " ";
    cout << "\n\n";
    
    // Финальный отчёт
    cout << "Sorting correct: " << (is_sorted ? "Yes" : "No") << "\n";
    cout << "Total GPU execution time: " << gpu_time_ms.count() / 1000.0 << " seconds\n";
    cout << "Total phases executed: " << phase_count << "\n";
    cout << "Note: Bitonic sort is a parallel version of merge sort suitable for GPU (Lecture #3)\n";
    
    return 0;  // Успешное завершение программы
}