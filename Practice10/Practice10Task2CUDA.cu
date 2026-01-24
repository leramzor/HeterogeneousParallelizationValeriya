#include "cuda_runtime.h"
#include "device_launch_parameters.h" // [Важно] Для распознавания blockIdx и threadIdx
#include <iostream>

// Ядро 1: Неэффективный (Strided) доступ к памяти
// [Лекция 10, стр. 11] Узкое место: задержки при доступе к памяти
__global__ void badAccessKernel(float* data, int n, int stride) {
    // Вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    // Доступ к памяти с "прыжком" (stride). 
    // Это заставляет контроллер памяти делать несколько транзакций вместо одной.
    if (tid * stride < n) {
        data[tid * stride] += 1.0f; 
    }
}

// Ядро 2: Эффективный (Coalesced) доступ к памяти
// [Лекция 10, стр. 12] Оптимизация: коалесцированный доступ
__global__ void goodAccessKernel(float* data, int n) {
    // Вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    // Потоки с соседними ID обращаются к соседним ячейкам памяти.
    // GPU объединяет эти запросы в одну быструю транзакцию.
    if (tid < n) {
        data[tid] += 1.0f; 
    }
}

int main() {
    int n = 1 << 22; // Размер массива: 4.2 млн элементов
    float *d_data;
    
    // Выделение памяти на устройстве (GPU)
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemset(d_data, 0, n * sizeof(float)); // Инициализация нулями

    // [Лекция 10, стр. 12] Использование cudaEvent для точного профилирования
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256; // Количество потоков в блоке
    int blocks = (n + threads - 1) / threads; // Расчет количества блоков

    // --- ТЕСТ 1: Эффективный доступ ---
    cudaEventRecord(start);
    goodAccessKernel<<<blocks, threads>>>(d_data, n); // Запуск ядра
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Ожидание завершения (Синхронизация)
    
    float ms_good = 0;
    cudaEventElapsedTime(&ms_good, start, stop);
    std::cout << ">>> [CUDA] Coalesced Access Time: " << ms_good << " ms" << std::endl;

    // --- ТЕСТ 2: Неэффективный доступ (шаг 2) ---
    cudaEventRecord(start);
    badAccessKernel<<<blocks, threads>>>(d_data, n, 2); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_bad = 0;
    cudaEventElapsedTime(&ms_bad, start, stop);
    std::cout << ">>> [CUDA] Strided Access Time: " << ms_bad << " ms" << std::endl;

    // Освобождение ресурсов
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}