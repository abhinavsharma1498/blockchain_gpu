#include <iostream>
#include <cryptopp/filters.h>
#include <cryptopp/hex.h>
#include <cryptopp/sha.h>
#include <chrono>

#define UPPER 4000000

using namespace std;
using namespace CryptoPP;
using namespace std::chrono;

void hash_benchmark(char* inp, unsigned int nonce, int size)
{
	SHA256 h;
	char data[1024];
	string digest = "";
	for(int i = 0; i < 64; ++i)
		digest += 'f';

	unsigned int temp = nonce;
	int digits = 0;
	while(temp > 0)
	{
		++digits;
		temp /= 10;
	}
	temp = nonce;
	int j = digits + size - 1;
	data[j--] = '\0';
	while(temp > 0)
	{
		data[j--] = '0' + temp%10;
		temp /= 10;
	}
	while(j >= 0)
	{
		data[j] = inp[j];
		j--;
	}
	size += digits;

	StringSource s(data, true, new HashFilter(h, new HexEncoder(new StringSink(digest))));
}

void runJobs(char* inp)
{
	unsigned int nonce = 1;
	int size = strlen(inp) + 1;
	while(nonce < UPPER)
	{
		hash_benchmark(inp, nonce, size);
		nonce += 1;
	}
}

int main(int argc, char** argv)
{
	SHA256 hash;
	string digest;
	string message = "abc\0";

	high_resolution_clock::time_point start = high_resolution_clock::now();
	runJobs(argv[1]);
	high_resolution_clock::time_point end = high_resolution_clock::now();

	duration<double, milli> time = end-start;
	cout << time.count() << endl;
	return 0;
}
