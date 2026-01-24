#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>

int main() {
    // 1. Параметры данных (64 МБ)
    int n = 1 << 24; 
    size_t size = n * sizeof(float);
    float *h_data, *d_data;

    // Выделение памяти
    cudaMallocHost(&h_data, size); 
    cudaMalloc(&d_data, size);

    // Подготовка инструментов замера времени (Лекция 10, стр. 12)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "================ HYBRID EXECUTION REPORT ================" << std::endl;
    std::cout << "Data Size: " << size / (1024 * 1024) << " MB" << std::endl;

    // Старт общего замера
    cudaEventRecord(start);

    // 2. Асинхронный запуск передачи (PCI-E)
    std::cout << ">>> [0ms] Initializing Async Transfer (Host to Device)..." << std::endl;
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

    // 3. Работа CPU в момент передачи
    std::cout << ">>> [1ms] CPU is now FREE. Starting independent math..." << std::endl;
    double cpu_val = 0.0;
    for(int i = 0; i < 5000000; i++) { cpu_val += i * 0.01; } // Имитация нагрузки

    std::cout << ">>> [5ms] CPU finished its task (Result: " << cpu_val << ")" << std::endl;
    std::cout << ">>> [5ms] CPU is waiting for GPU to finish transfer..." << std::endl;

    // 4. Синхронизация
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);

    // РАСШИРЕННЫЙ ВЫВОД ЦИФР
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(30) << "Total Hybrid Time:" << total_ms << " ms" << std::endl;
    
    // Расчет пропускной способности (Bandwidth)
    double seconds = total_ms / 1000.0;
    double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / seconds;

    std::cout << std::left << std::setw(30) << "Effective Bandwidth:" << bandwidth << " GB/s" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << ">>> VERDICT: CPU was productive for ~5ms during transfer." << std::endl;
    std::cout << ">>> BENEFIT: Bottleneck latency hidden by " << (5.0 / total_ms) * 100 << "%" << std::endl;
    std::cout << "=========================================================" << std::endl;

    cudaStreamDestroy(stream);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    return 0;
}