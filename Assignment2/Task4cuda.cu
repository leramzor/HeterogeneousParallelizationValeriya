// task4_merge_sort_cuda.cu
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

__global__ void bitonic_sort_step(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                int temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            if (arr[i] < arr[ixj]) {
                int temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void run_test(int power_of_two) {
    int size = 1 << power_of_two;
    vector host_arr(size);

    mt19937 gen(time(nullptr));
    uniform_int_distribution dist(1, 1000000);
    generate(host_arr.begin(), host_arr.end(), [&]() { return dist(gen); });

    int *dev_arr;
    CUDA_CHECK(cudaMalloc(&dev_arr, size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_arr, host_arr.data(), size * sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(256);
    dim3 blocks((size + 255) / 256);

    auto start = chrono::high_resolution_clock::now();

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<>>(dev_arr, j, k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration time = end - start;

    CUDA_CHECK(cudaMemcpy(host_arr.data(), dev_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_arr));

    bool sorted = is_sorted(host_arr.begin(), host_arr.end());

    cout << "Array size: " << size << " elements (power of 2)\n";
    cout << "GPU time: " << time.count() << " seconds\n";
    cout << "Sorting correct: " << (sorted ? "Yes" : "No") << "\n\n";
}

int main() {
    cout << "Task 4: Parallel Merge Sort on GPU (Bitonic Sort)\n";

    cout << "Testing with 16384 elements (2^14 ≈ 10 000)...\n";
    run_test(14);  // ~16 384 > 10 000

    cout << "Testing with 131072 elements (2^17 ≈ 100 000)...\n";
    run_test(17);  // ~131 072 > 100 000

    cout << "Conclusion: Bitonic sort works correctly only for power-of-2 sizes and demonstrates massive parallelism on GPU (Lecture #3)\n";

    return 0;
}
     
