#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>

// __const__ BYTE * buff;


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

__global__ void sha256_cuda(JOB ** jobs, int n, int *g_found, int *g_nonce) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
//         printf("in kernel: g_nonce = %d, i = %d\n", *g_nonce, i);

		if(checkZeroPadding(jobs[i]->digest, 2) && atomicExch(g_found, 1) == 0) {
			*g_nonce = i;
//             printf("in if statement: g_nonce = %d, i = %d\n", *g_nonce, i);
		}
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runJobs(JOB ** jobs, int n, int* g_found, int *g_nonce){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
//     printf("%d\t%d\n", *g_found, *g_nonce);
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n, g_found, g_nonce);
//     printf("%d\t%d\n", *g_found, *g_nonce);
}


JOB * JOB_init(BYTE * data, long size) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	return j;
}


int main(int argc, char **argv) {
	int i = 0, n = 0;
	BYTE * buff;
	JOB ** jobs;
// 	std::string inp = "abc";
	int *g_found;
	int *g_nonce;
	checkCudaErrors(cudaMallocManaged(&g_found, sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&g_nonce, sizeof(int)));
	*g_found = 0;

	n = 1000;
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

	while (i < n) {
		char inp[512] = "abc";
		int tmp = i, digits = 0;
		
		while(tmp > 0)
		{
			tmp /= 10;
			++digits;
		}
		tmp = i;
		digits += strlen(inp);
		inp[digits--] = '\0';
		while(tmp > 0)
		{
			inp[digits--] = '0' + tmp%10;
			tmp /= 10;
		}
		int size = strlen(inp);
        
        checkCudaErrors(cudaMallocManaged(&buff, (size+1)*sizeof(char)));
        memcpy(buff, inp, sizeof(inp));

        jobs[i++] = JOB_init(buff, size+1);
	}

	pre_sha256();
	runJobs(jobs, n, g_found, g_nonce);

	cudaDeviceSynchronize();
// 	free(buff);
	print_jobs(jobs, n);
// 	printf("%d\n", *g_found);
//     printf("%d\n", *g_nonce);
    cudaDeviceReset();
    
	return 0;
}
