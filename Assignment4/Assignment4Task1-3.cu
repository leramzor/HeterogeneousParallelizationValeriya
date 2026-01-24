#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h> // Лекция 2: Библиотека OpenMP для CPU
#include <stdio.h>

using namespace std;

// --- МАКРОС ПРОВЕРКИ ОШИБОК (Лекция 3) ---
// Оборачиваем каждый вызов CUDA, чтобы сразу видеть, если что-то упало
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// =========================================================
// ЗАДАНИЕ 1: ЯДРО СУММЫ (Лекция 5 и 7)
// Использование атомиков для глобальной редукции
// =========================================================
__global__ void sum_global_kernel(float* d_arr, float* d_sum, int n) {
    // Лекция 3: Вычисление глобального индекса нити
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        // Лекция 5: Race Condition (Состояние гонки).
        // Если все нити просто напишут *d_sum += ..., будет мусор.
        // atomicAdd блокирует адрес памяти на момент записи.
        atomicAdd(d_sum, d_arr[idx]);
    }
}

// =========================================================
// ЗАДАНИЕ 2: ЯДРО ПРЕФИКСНОЙ СУММЫ (Лекция 4 и 7)
// Алгоритм Хилиса-Стила с использованием Shared Memory
// =========================================================
__global__ void scan_shared_kernel(float* d_out, float* d_in, int n) {
    // Лекция 4: Shared Memory — сверхбыстрая память внутри мультипроцессора (SM).
    // extern означает, что размер мы укажем при запуске <<<...>>>
    extern __shared__ float temp[];

    int thid = threadIdx.x; // Локальный индекс внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс

    // 1. Загрузка данных из медленной Global Memory в быструю Shared Memory
    temp[thid] = (idx < n) ? d_in[idx] : 0.0f;

    // Лекция 4: Барьерная синхронизация.
    // Ждем, пока ВСЕ нити блока загрузят данные, прежде чем считать.
    __syncthreads();

    // 2. Сам алгоритм (Scan)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0;
        if (thid >= offset) {
            val = temp[thid - offset]; // Читаем "соседа" слева
        }
        __syncthreads(); // Важно! Ждем, пока все прочитают

        temp[thid] += val; // Складываем
        __syncthreads(); // Ждем, пока все запишут перед следующим шагом
    }

    // 3. Выгрузка результата обратно в глобальную память
    if (idx < n) d_out[idx] = temp[thid];
}

// =========================================================
// ЗАДАНИЕ 3: ЯДРО УМНОЖЕНИЯ (Лекция 3)
// Простой параллелизм данных
// =========================================================
__global__ void multiply_kernel(float* data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug: Вывод прямо с видеокарты (работает, но медленно, используем только для отладки)
    if (idx == 0) printf("   [GPU Msg] Kernel started on device!\n");

    if (idx < n) data[idx] *= scalar;
}

// =========================================================
// ХОСТ (CPU) КОД
// =========================================================
int main() {
    cout << "=== STARTING ASSIGNMENT 4 ===" << endl << endl;

    // -----------------------------------------------------
    // TASK 1: СУММА
    // -----------------------------------------------------
    cout << "--- Task 1: Global Sum ---" << endl;
    const int N1 = 100000;
    vector<float> h_arr1(N1, 1.0f); // Хост-массив (CPU)

    float *d_arr1, *d_sum1, h_sum1 = 0;

    // Лекция 3: Выделение памяти на устройстве (GPU)
    CUDA_CHECK(cudaMalloc(&d_arr1, N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum1, sizeof(float)));

    // Лекция 3: Копирование Host -> Device
    CUDA_CHECK(cudaMemcpy(d_arr1, h_arr1.data(), N1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum1, 0, sizeof(float))); // Обнуление результата

    // Конфигурация запуска
    int threads = 256;
    int blocks = (N1 + threads - 1) / threads;

    cout << "Launching Kernel: " << blocks << " blocks, " << threads << " threads." << endl;
    sum_global_kernel<<<blocks, threads>>>(d_arr1, d_sum1, N1);

    // Лекция 10: Синхронизация. CPU не должен читать результат, пока GPU не закончит.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копирование результата Device -> Host
    CUDA_CHECK(cudaMemcpy(&h_sum1, d_sum1, sizeof(float), cudaMemcpyDeviceToHost));
    cout << "Result: " << h_sum1 << " (Expected: 100000)" << endl << endl;

    // Освобождение памяти (Лекция 3)
    cudaFree(d_arr1); cudaFree(d_sum1);


    // -----------------------------------------------------
    // TASK 2: ПРЕФИКСНАЯ СУММА (SCAN)
    // -----------------------------------------------------
    cout << "--- Task 2: Prefix Sum (Shared Memory) ---" << endl;
    const int N2 = 256; // Возьмем 1 блок для наглядности
    vector<float> h_in2(N2, 1.0f), h_out2(N2);

    float *d_in2, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_in2, N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, N2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in2, h_in2.data(), N2 * sizeof(float), cudaMemcpyHostToDevice));

    // Лекция 4: Динамическая Shared Memory
    // Третий параметр в <<<...>>> — это размер памяти в байтах!
    // Без этого (256 * sizeof(float)) ядро упадет или выдаст 0.
    size_t sharedMemSize = 256 * sizeof(float);
    scan_shared_kernel<<<1, 256, sharedMemSize>>>(d_out2, d_in2, N2);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out2.data(), d_out2, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Last element: " << h_out2[255] << " (Expected: 256)" << endl << endl;

    cudaFree(d_in2); cudaFree(d_out2);


    // -----------------------------------------------------
    // TASK 3: ГИБРИДНЫЕ ВЫЧИСЛЕНИЯ (OpenMP + CUDA) - Лекция 8
    // -----------------------------------------------------
    cout << "--- Task 3: Hybrid (CPU + GPU) ---" << endl;
    const int N3 = 20;
    vector<float> h_arr3(N3, 10.0f);
    int half = N3 / 2;

    cout << "Original Array: ";
    for(auto x : h_arr3) cout << x << " ";
    cout << endl;

    // --- ЧАСТЬ 1: CPU (OpenMP) ---
    // Лекция 2: #pragma omp parallel for распараллеливает цикл по ядрам процессора
    cout << "[CPU] Processing first half..." << endl;
    #pragma omp parallel for
    for (int i = 0; i < half; i++) {
        h_arr3[i] *= 2.0f;
    }

    // --- ЧАСТЬ 2: GPU (CUDA) ---
    cout << "[GPU] Processing second half..." << endl;
    float* d_arr3;
    // Копируем только вторую половину массива (сдвиг указателя + half)
    CUDA_CHECK(cudaMalloc(&d_arr3, (N3 - half) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_arr3, h_arr3.data() + half, (N3 - half) * sizeof(float), cudaMemcpyHostToDevice));

    multiply_kernel<<<1, 256>>>(d_arr3, 2.0f, N3 - half);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Возвращаем результат второй половины обратно
    CUDA_CHECK(cudaMemcpy(h_arr3.data() + half, d_arr3, (N3 - half) * sizeof(float), cudaMemcpyDeviceToHost));

    cout << "Final Hybrid Array: ";
    for(int i=0; i<N3; i++) {
        if(i == half) cout << "| "; // Разделитель CPU/GPU
        cout << h_arr3[i] << " ";
    }
    cout << endl;

    cudaFree(d_arr3);
    return 0;
}