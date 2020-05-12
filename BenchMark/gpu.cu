#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include "sha256.cuh"

#define UPPER 4000000000

__global__ void sha256_cuda(BYTE* buff, unsigned int nonce, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	BYTE digest[64];
	BYTE data[1024];
	for (int i = 0; i < 64; i++)
	{
		digest[i] = 0xff;
	}
	
	// perform sha256 calculation here
	nonce += i;

	unsigend int tmp = nonce;
	int digits = 0;
	while(tmp > 0)
	{
		++digits;
		tmp /= 10;
	}
	tmp = nonce;
	int j = digits + size-1;
	data[j--] = '\0';
	while(tmp > 0)
	{
		data[j--] = '0' + tmp%10;
		tmp /= 10;
	}
	while(j >= 0)
	{
		data[j] = buff[j];
		j--;
	}
	size += digits;

	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, data, size);
	sha256_final(&ctx, digest);
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runJobs(BYTE* buff, int size){
	int gridSize = 2, blockSize = 1024;
	unsigned int nonce = 1;
	while(nonce < UPPER)
	{
		sha256_cuda <<< gridSize, blockSize >>> (buff, nonce, size);
		cudaDeviceSynchronize();
		nonce += blockSize * gridSize;
	}
}


int main(int argc, char **argv) {
	cudaSetDevice(0);
	BYTE * buff;
	
	checkCudaErrors(cudaMallocManaged(&buff, strlen(argv[1])+1));
	memcpy(buff, argv[1], strlen(argv[1])+1);

	pre_sha256();
	std::chrono::high_resolution_clock::time_point start = high_resolution_clock::now();
	runJobs(buff, strlen(argv[1])+1);
	std::chrono::high_resolution_clock::time_point end = high_resolution_clock::now();

	cudaDeviceReset();
	
	std::chrono:duration<double, std::chrono::milli> time = end-start;
	printf("%lf\n", time.count());
	return 0;
}
