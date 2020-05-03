#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>


__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
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
	unsigned long temp = 3;
	BYTE * buff;
	JOB ** jobs;
	char inp[] = "abc";
// 	checkCudaErrors(cudaMallocManaged(&buff, (temp+1)*sizeof(char)));
// 	memcpy(buff, inp, temp+1);
//     cudaMemcpyToSymbol(buff, inp, sizeof(inp), 0, cudaMemcpyHostToDevice);

	n = 10;
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

	while (i < n) {
        checkCudaErrors(cudaMallocManaged(&buff, (temp)*sizeof(char)));
        cudaMemcpyToSymbol(buff, inp, sizeof(inp), 0, cudaMemcpyHostToDevice);
		jobs[i++] = JOB_init(buff, temp);
	}

	pre_sha256();
	runJobs(jobs, n);

	cudaDeviceSynchronize();
// 	free(buff);
	print_jobs(jobs, n);
	cudaDeviceReset();
	return 0;
}
