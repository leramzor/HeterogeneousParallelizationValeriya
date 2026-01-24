#include <iostream>
#include <vector>
#include <omp.h>           // Библиотека для многопоточности на CPU
#include <cuda_runtime.h>  // Основные функции CUDA
#include <device_launch_parameters.h>

using namespace std;

// --- МАКРОС ПРОВЕРКИ ОШИБОК ---
// Обоснование: CUDA вызовы асинхронны. Без проверки мы не узнаем, 
// хватило ли памяти или запустилось ли ядро.
#define CUDA_CHECK(err) { if (err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

// --- CUDA ЯДРО (DEVICE CODE) ---
// Обоснование: GPU эффективен для задач SIMT (одна инструкция - много потоков).
// Каждый поток получает свой уникальный индекс и обрабатывает один элемент массива.
__global__ void processArrayKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f; 
    }
}

int main() {
    const int N = 1000000; // 1 миллион элементов
    size_t size = N * sizeof(float);
    
    vector<float> h_array(N, 1.0f); // Хост-массив (в оперативной памяти)
    double start, end;

    cout << "=== PRACTICE 8: HYBRID CPU & GPU PROCESSING ===" << endl;

    // -----------------------------------------------------------------------
    // ЗАДАНИЕ 1: ТОЛЬКО CPU (OpenMP)
    // Обоснование: Используем все ядра процессора. Эффективно для кэш-ориентированных задач,
    // так как нет задержек на передачу данных по внешней шине.
    // -----------------------------------------------------------------------
    start = omp_get_wtime();
    #pragma omp parallel for // Распараллеливание цикла между ядрами CPU
    for (int i = 0; i < N; i++) {
        h_array[i] *= 2.0f;
    }
    end = omp_get_wtime();
    cout << "1. CPU (OpenMP) Time: " << (end - start) << " s." << endl;

    fill(h_array.begin(), h_array.end(), 1.0f); // Сброс данных

    // -----------------------------------------------------------------------
    // ЗАДАНИЕ 2: ТОЛЬКО GPU (CUDA)
    // Обоснование: Здесь проявляется "узкое место" — шина PCI Express. 
    // Время замера включает копирование Host->Device и Device->Host.
    // -----------------------------------------------------------------------
    float *d_array;
    CUDA_CHECK(cudaMalloc(&d_array, size)); // Выделение видеопамяти
    
    start = omp_get_wtime();
    // Копирование данных на GPU (Медленная операция)
    CUDA_CHECK(cudaMemcpy(d_array, h_array.data(), size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    processArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);
    
    CUDA_CHECK(cudaDeviceSynchronize()); // Ожидание завершения вычислений
    
    // Возврат данных (Медленная операция)
    CUDA_CHECK(cudaMemcpy(h_array.data(), d_array, size, cudaMemcpyDeviceToHost));
    end = omp_get_wtime();
    cout << "2. GPU (CUDA) Time: " << (end - start) << " s (with data transfer)." << endl;

    fill(h_array.begin(), h_array.end(), 1.0f);

    // -----------------------------------------------------------------------
    // ЗАДАНИЕ 3: ГИБРИДНЫЙ РЕЖИМ (ОДНОВРЕМЕННО CPU + GPU)
    // Обоснование: Суть гибридных вычислений — скрыть задержки.
    // Пока GPU занят копированием и расчетом своей части, CPU выполняет свою долю работы.
    // -----------------------------------------------------------------------
    int half = N / 2;
    float *d_hybrid;
    CUDA_CHECK(cudaMalloc(&d_hybrid, (N - half) * sizeof(float)));

    start = omp_get_wtime();

    // Шаг А: Подготовка GPU (вторая половина массива)
    // В реальных задачах здесь лучше использовать Streams для асинхронности
    CUDA_CHECK(cudaMemcpy(d_hybrid, h_array.data() + half, (N - half) * sizeof(float), cudaMemcpyHostToDevice));
    processArrayKernel<<<((N - half) + 255) / 256, 256>>>(d_hybrid, N - half);

    // Шаг Б: Работа CPU (первая половина массива)
    // CPU работает параллельно с GPU.
    #pragma omp parallel for
    for (int i = 0; i < half; i++) {
        h_array[i] *= 2.0f;
    }

    // Шаг В: Синхронизация
    // Ждем, пока GPU закончит свою половину, чтобы собрать массив воедино
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_array.data() + half, d_hybrid, (N - half) * sizeof(float), cudaMemcpyDeviceToHost));
    
    end = omp_get_wtime();
    cout << "3. Hybrid (CPU + GPU) Time: " << (end - start) << " s." << endl;

    // Очистка ресурсов
    cudaFree(d_array);
    cudaFree(d_hybrid);
    return 0;
}