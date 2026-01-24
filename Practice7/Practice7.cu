#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(err) { if (err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

// --- ЗАДАНИЕ 1: ЯДРО РЕДУКЦИИ (СУММА) ---
// Реализация на основе Лекции №7, стр. 4-5
__global__ void reduceSumKernel(int *input, int *output, int n) {
    // Объявление разделяемой памяти [cite: 91]
    extern __shared__ int sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Загрузка данных в разделяемую память [cite: 74, 76]
    sharedData[tid] = (i < n) ? input[i] : 0;
    __syncthreads(); // Синхронизация [cite: 92]

    // Итеративное суммирование с уменьшением шага в 2 раза [cite: 80, 93]
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads(); 
    }

    // Запись результата блока в глобальную память [cite: 84, 85]
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

// --- ЗАДАНИЕ 2: ЯДРО ПРЕФИКСНОЙ СУММЫ (SCAN) ---
// Реализация на основе Лекции №7, стр. 8-9
__global__ void prefixSumKernel(int *input, int *output, int n) {
    extern __shared__ int sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Итеративное вычисление суммы с увеличением шага в 2 раза [cite: 128, 142]
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int temp = 0;
        if (tid >= s) {
            temp = sharedData[tid - s]; // [cite: 131]
        }
        __syncthreads();
        sharedData[tid] += temp;
        __syncthreads();
    }

    if (i < n) output[i] = sharedData[tid]; // [cite: 137]
}

int main() {
    const int N = 1024; // Пример тестового массива
    size_t size = N * sizeof(int);
    vector<int> h_in(N, 1); // Массив из 1024 единиц
    int h_sum_res, *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice));

    // Выполнение редукции
    reduceSumKernel<<<1, N, size>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(&h_sum_res, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "Task 1 (Reduction Sum): " << h_sum_res << " (Expected: 1024)" << endl;

    // Выполнение сканирования
    prefixSumKernel<<<1, N, size>>>(d_in, d_out, N);
    vector<int> h_scan_res(N);
    CUDA_CHECK(cudaMemcpy(h_scan_res.data(), d_out, size, cudaMemcpyDeviceToHost));
    cout << "Task 2 (Prefix Sum) Last element: " << h_scan_res[N-1] << " (Expected: 1024)" << endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}