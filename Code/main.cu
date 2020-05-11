#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "sha256.cuh"

//	Upper limit of nonce
#define UPPER 4000000000

/*	Check the difficulty of hash.
	i.e. Number of leading zeros in the hash.

	Arguments:
		sha -> The digest (in BYTE format).
		difficulty -> The number of leading zeros required.
	Return value:
		Data type: boolean
		True -> If the number of leading zeros in sha are diffculty.
		False -> Otherwise.
*/
__device__ bool checkZeroPadding(BYTE* sha, uint8_t difficulty)
{
	bool isOdd = difficulty % 2 != 0;
	uint8_t max = (difficulty / 2) + 1;

	/*
	Odd : 00 00 01 need to check 0 -> 2
	Even : 00 00 00 1 need to check 0 -> 3
	odd : 5 / 2 = 2 => 2 + 1 = 3
	even : 6 / 2 = 3 => 3 + 1 = 4
	*/

	for (uint8_t cur_byte = 0; cur_byte < max; ++cur_byte)
	{
		uint8_t b = sha[cur_byte];
		if (cur_byte < max - 1)	// Before the last byte should be all zero
		{
			if(b != 0)	return false;
		}
		else if (isOdd)
		{
			if (b > 0x0F || b == 0)	return false;
		}
		else if (b <= 0x0f)	return false;
	}

	return true;
}

/*	Kernel for SHA256.
	Responsible for formatting the data to be fed to sha256 algo.

	Arguments:
		buff -> String of type BYTE for which the nonce is to be searched.
		g_found -> To identify if the nonce satisfying the difficulty criteria is found or not.
		g_nonce -> Stores the nonce which satisfy the criteria (if one is found).
		nonce -> Base value (lower limit) of nonce for the current batch of kernel.
		size -> Length of the string stored in buff.
		diff -> Difficulty for the current batch.
	Return Value:
		void
*/
__global__ void sha256_cuda(BYTE* buff, int *g_found, unsigned int *g_nonce, unsigned int nonce, int size, int diff)
{
	//	Calculating the thread Index of current thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	BYTE data[1024];
	BYTE digest[64];
	for (int i = 0; i < 64; i++)
	{
		digest[i] = 0xff;
	}

	//	Appending nonce at the end of data

	nonce += i;
	unsigned int tmp = nonce;
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

	//	Starting SHA256 calculations here
	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, data, size-1);
	sha256_final(&ctx, digest);

	//	Checking for the difficulty criteria
	if(checkZeroPadding(digest, diff) && atomicExch(g_found, 1) == 0)
		*g_nonce = nonce;
}

/*	Prepare constant data required for SHA256 calculations in the device memory
*/
void pre_sha256()
{
	//	Copying Symbols from host to device
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

/*	Driver Function which control and oversee kernel calls
	Arguments:
		buff -> String of type BYTE for which the nonce is to be searched.
		g_found -> To identify if the nonce satisfying the difficulty criteria is found or not.
		g_nonce -> Stores the nonce which satisfy the criteria (if one is found).
		size -> Length of the string stored in buff.
		diff -> Difficulty for the current batch.
	Return value:
		void
*/
void runJobs(BYTE* buff, int* g_found, unsigned int *g_nonce, int size, int diff)
{
	//	Initializing the size of grid and block with optimal values (pre-calculated)
	int gridSize = 2, blockSize = 1024;
	
	unsigned int nonce = 1;
	
	while(!(*g_found))
	{
		//	Invkoing the kernel
		sha256_cuda <<< gridSize, blockSize >>> (buff, g_found, g_nonce, nonce, size, diff);
		//	Waiting for current batch to complete execution
		cudaDeviceSynchronize();
		//	Adjusting nonce for next batch
		nonce += blockSize * gridSize;
		//	Checking for allowed upper limit of nonce
		if(nonce  > UPPER)	break;
	}
}

/*	Driver Function the program.

	Expected command line invocation:
	~$./prog string_whose_nonce_is_to_be_found difficulty
*/
int main(int argc, char **argv)
{
	//	Selecting primary GPU
	cudaSetDevice(0);
	
	//	Declaring symbols for Unified Memory
	BYTE *buff;
	int *g_found;
	unsigned int *g_nonce;
	//	Allocating memory in Unified Memory
	checkCudaErrors(cudaMallocManaged(&g_found, sizeof(int)));
	checkCudaErrors(cudaMallocManaged(&g_nonce, sizeof(unsigned int)));
	checkCudaErrors(cudaMallocManaged(&buff, strlen(argv[1])+1));
	//	Initializing the values of those symbols
	*g_found = 0;
	memcpy(buff, argv[1], strlen(argv[1])+1);
	int diff = atoi(argv[2]);

	//	Starting the SHA256 calculations in main()
	pre_sha256();
	runJobs(buff, g_found, g_nonce, strlen(argv[1])+1, diff);

	//	Displaying the nonce on stdout
	printf("%d", *g_nonce);

	//	Free the selected GPU
	cudaDeviceReset();
	return 0;
}