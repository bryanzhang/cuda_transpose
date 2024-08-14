#include <cstdlib>
#include <chrono>

#include <iostream>
#include <utility>

using namespace std;

constexpr int nx = 32;
constexpr int ny = 32;

__global__ void smTranspose(float *out, float *in) {
        __shared__ float tile[ny][nx + 1];

        unsigned int ix = blockIdx.x * nx + threadIdx.x;
        unsigned int iy = blockIdx.y * ny + threadIdx.y;

        // 读取一行再往共享内存写一行
        tile[threadIdx.y][threadIdx.x] = in[iy * nx + ix];
        __syncthreads();

        // 共享内存读一列再往全局内存写一行
        int bidx = threadIdx.y * blockDim.x + threadIdx.x;
        out[iy * ny + ix] = tile[(bidx%blockDim.y)][(bidx/blockDim.y)];
}

constexpr int nx_matrix = 4096;
constexpr int ny_matrix = 4096;

double test(float* d_in, float* d_out) {
        cudaMemset(d_out, 0, sizeof(float) * nx_matrix * ny_matrix);
        dim3 grid(nx_matrix / nx, ny_matrix / ny);
        dim3 block(nx, ny);
        auto start = std::chrono::high_resolution_clock::now();
        smTranspose<<<grid, block>>>(d_out, d_in);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
}

int main() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // 0是设备ID
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

        srand(2);
        float* h_in = new float[nx_matrix * ny_matrix];
        for (int i = 0; i < nx_matrix * ny_matrix; ++i) {
                h_in[i] = rand();
        }
        float* h_out = new float[nx_matrix * ny_matrix];

        float* d_in = nullptr;
        cudaMalloc(&d_in, nx_matrix * ny_matrix * sizeof(float));
        cudaMemcpy(d_in, h_in, nx_matrix * ny_matrix * sizeof(float), cudaMemcpyHostToDevice);
        float* d_out = nullptr;
        cudaMalloc(&d_out, nx_matrix * ny_matrix * sizeof(float));
        cout << "d_in=" << d_in << ",\t" << "dout=" << d_out << endl;

        double elapsed = 0.0;
        constexpr int tries = 50;
        for (int i = 0; i < tries; ++i) {
                elapsed += test(d_in, d_out);
        }
        cout << "Tries=" << tries << ", average elpased time: " << elapsed * 1000 / tries << " ms." << endl;

        delete[] h_in;
        delete[] h_out;
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
}
root@iZuf6hns4egiw6oqxq2a96Z:~/transpose#
root@iZuf6hns4egiw6oqxq2a96Z:~/transpose#
root@iZuf6hns4egiw6oqxq2a96Z:~/transpose#
root@iZuf6hns4egiw6oqxq2a96Z:~/transpose# cat 3_unroll.cu
#include <cstdlib>
#include <chrono>

#include <iostream>
#include <utility>

using namespace std;

constexpr int nx = 32;
constexpr int ny = 32;

__global__ void smTranspose(float *out, float *in) {
        __shared__ float tile[ny][2*nx + 1];

        unsigned int ix = 2 * blockIdx.x * nx + threadIdx.x;
        unsigned int iy = blockIdx.y * ny + threadIdx.y;

        // 读取一行再往共享内存写一行
        tile[threadIdx.y][threadIdx.x] = in[iy * nx + ix];
        tile[threadIdx.y][threadIdx.x + nx] = in[iy * nx + ix + nx];
        __syncthreads();

        // 共享内存读一列再往全局内存写一行
        int bidx = threadIdx.y * 2 * nx + threadIdx.x;
        out[iy * ny + ix] = tile[(bidx%blockDim.y)][(bidx/blockDim.y/2)];
        out[(iy + nx) * ny + ix] = tile[(bidx%blockDim.y)][(bidx/blockDim.y/2) + nx];
}

constexpr int nx_matrix = 4096;
constexpr int ny_matrix = 4096;

double test(float* d_in, float* d_out) {
        cudaMemset(d_out, 0, sizeof(float) * nx_matrix * ny_matrix);
        dim3 grid(nx_matrix / nx / 2, ny_matrix / ny);
        dim3 block(nx, ny);
        auto start = std::chrono::high_resolution_clock::now();
        smTranspose<<<grid, block>>>(d_out, d_in);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
}

int main() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // 0是设备ID
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

        srand(2);
        float* h_in = new float[nx_matrix * ny_matrix];
        for (int i = 0; i < nx_matrix * ny_matrix; ++i) {
                h_in[i] = rand();
        }
        float* h_out = new float[nx_matrix * ny_matrix];

        float* d_in = nullptr;
        cudaMalloc(&d_in, nx_matrix * ny_matrix * sizeof(float));
        cudaMemcpy(d_in, h_in, nx_matrix * ny_matrix * sizeof(float), cudaMemcpyHostToDevice);
        float* d_out = nullptr;
        cudaMalloc(&d_out, nx_matrix * ny_matrix * sizeof(float));
        cout << "d_in=" << d_in << ",\t" << "dout=" << d_out << endl;

        double elapsed = 0.0;
        constexpr int tries = 50;
        for (int i = 0; i < tries; ++i) {
                elapsed += test(d_in, d_out);
        }
        cout << "Tries=" << tries << ", average elpased time: " << elapsed * 1000 / tries << " ms." << endl;

        delete[] h_in;
        delete[] h_out;
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
}
