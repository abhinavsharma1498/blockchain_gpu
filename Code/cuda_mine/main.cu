#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>

#define DIFFICULTY 2


__device__ bool checkZeroPadding(unsigned char* sha, uint8_t difficulty) {

	bool isOdd = difficulty % 2 != 0;
	uint8_t max = (difficulty / 2) + 1;

	/*
		Odd : 00 00 01 need to check 0 -> 2
		Even : 00 00 00 1 need to check 0 -> 3
		odd : 5 / 2 = 2 => 2 + 1 = 3
		even : 6 / 2 = 3 => 3 + 1 = 4
	*/
	for (uint8_t cur_byte = 0; cur_byte < max; ++cur_byte) {
		uint8_t b = sha[cur_byte];
		if (cur_byte < max - 1) { // Before the last byte should be all zero
			if(b != 0) return false;
		}else if (isOdd) {
			if (b > 0x0F || b == 0) return false;
		}
		else if (b <= 0x0f) return false;
		
	}

	return true;

}

__global__ void sha256_cuda(BYTE* buff, int n, int *g_found, int *g_nonce, BYTE* g_out, int nonce, size_t size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	BYTE digest[64];
	BYTE data[1024];
	for (int i = 0; i < 64; i++)
	{
		digest[i] = 0xff;
	}
	
	// perform sha256 calculation here
	if (i < n){
		nonce += i;

		int tmp = nonce, digits = 0;
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

		if(checkZeroPadding(digest, DIFFICULTY) && atomicExch(g_found, 1) == 0) {
			*g_nonce = nonce;
			memcpy(g_out, digest, sizeof(digest));
			printf("%d\n", *g_nonce);
		    for (j = 0; j < 32; j++)
			{
				printf("%.2x", data[j]);
			}
			printf("\n");
		}
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runJobs(BYTE* buff, int n, int* g_found, int *g_nonce, BYTE* g_out, size_t size){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	int nonce = 1;
	while(!(*g_found))
	{
		sha256_cuda <<< numBlocks, blockSize >>> (buff, n, g_found, g_nonce, g_out, nonce, size);
		nonce += numBlocks;
	}
}


int main(int argc, char **argv) {
	cudaSetDevice(0);
	int n = 10;
	BYTE * buff;
	
	int *g_found;
	int *g_nonce;
	BYTE *g_out;
	checkCudaErrors(cudaMallocManaged(&g_found, sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&g_nonce, sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&g_out, 64*sizeof(BYTE)));
	*g_found = 0;

	char inp[] = "abc";
	checkCudaErrors(cudaMallocManaged(&buff, sizeof(inp)+32));
	memcpy(buff, inp, sizeof(inp));

	pre_sha256();
	runJobs(buff, n, g_found, g_nonce, g_out, sizeof(inp));

	cudaDeviceSynchronize();
    printf("%d\n", *g_nonce);
    printf("%s\n", hash_to_string(g_out));
	cudaDeviceReset();
	return 0;
}
