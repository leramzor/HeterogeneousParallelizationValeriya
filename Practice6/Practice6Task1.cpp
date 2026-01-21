// main.cpp — Практическая работа №6: Поэлементное сложение массивов на OpenCL (CPU и GPU)
#include <iostream>          // Для вывода результатов и отладки
#include <vector>            // Массивы на CPU
#include <random>            // Генерация случайных чисел
#include <chrono>            // Замер времени
#include <cl.hpp>         // OpenCL C++ bindings (современная версия)

using namespace std;

// Макрос проверки ошибок OpenCL
#define CL_CHECK(err) do { \
    if (err != CL_SUCCESS) { \
        cerr << "OpenCL error code: " << err << " at line " << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

int main() {
    cout << "Practical Work 6: Vector Addition on OpenCL (CPU & GPU)\n\n";

    const int N = 10000000;  // 10 миллионов элементов — большой объём для сравнения
    vector<float> h_a(N), h_b(N), h_c(N);

    // Заполнение случайными числами
    mt19937 gen(time(nullptr));
    uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (int i = 0; i < N; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    // Получаем платформы OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        cerr << "No OpenCL platforms found!" << endl;
        return 1;
    }

    cout << "Available platforms:\n";
    for (size_t i = 0; i < platforms.size(); ++i) {
        cout << "  " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
    }

    // Выбираем первую платформу
    cl::Platform platform = platforms[0];

    // Получаем устройства (CPU и GPU)
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        cerr << "No OpenCL devices found!" << endl;
        return 1;
    }

    cout << "\nAvailable devices:\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        string name = devices[i].getInfo<CL_DEVICE_NAME>();
        string type = (devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU";
        cout << "  " << i << ": " << name << " (" << type << ")\n";
    }

    // Тестируем каждое устройство
    for (const auto& device : devices) {
        string name = device.getInfo<CL_DEVICE_NAME>();
        string type = (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU";
        cout << "\n=== Testing on " << name << " (" << type << ") ===\n";

        // Создаём контекст и очередь
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Код ядра (можно вынести в отдельный .cl файл)
        string kernel_code = R"(
            __kernel void vector_add(__global const float* A,
                                     __global const float* B,
                                     __global float* C,
                                     const int n) {
                int id = get_global_id(0);
                if (id < n) {
                    C[id] = A[id] + B[id];
                }
            }
        )";

        cl::Program::Sources sources = {
            {kernel_code.c_str(), kernel_code.length()}
        };
        cl::Program program(context, sources);
        if (program.build({ device }) != CL_SUCCESS) {
            cout << "Build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
            return 1;
        }

        cl::Kernel kernel(program, "vector_add");

        // Буферы памяти
        cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_a.data());
        cl::Buffer buf_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_b.data());
        cl::Buffer buf_c(context, CL_MEM_WRITE_ONLY, N * sizeof(float));

        kernel.setArg(0, buf_a);
        kernel.setArg(1, buf_b);
        kernel.setArg(2, buf_c);
        kernel.setArg(3, N);

        // Запуск ядра
        cl::NDRange global(N);
        cl::NDRange local = cl::NullRange;

        auto start = chrono::high_resolution_clock::now();
        CL_CHECK(queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local));
        CL_CHECK(queue.finish());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> time = end - start;

        // Чтение результата
        CL_CHECK(queue.enqueueReadBuffer(buf_c, CL_TRUE, 0, N * sizeof(float), h_c.data()));

        cout << "Time: " << time.count() << " seconds\n";
    }


    return 0;
}