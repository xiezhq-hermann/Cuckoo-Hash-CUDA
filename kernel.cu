#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <ctime>

using namespace std;

__global__ void cuckooKernel(unsigned* keys, unsigned* table, int amount_keys, int bound) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= amount_keys){
        return;
    }
    unsigned location[3];
    unsigned key = keys[index];
    unsigned random_entry = PSEUDO(key);
    location[0] = hash_func0(key);
#pragma unroll
    for (int j = 0; j < bound; j++){
        key = atomicExch(&table[location[j%3]*2 + random_entry], key);
        if (key == EMPTY){
            return;
        }
        location[0] = hash_func0(key);
        location[1] = hash_func1(key);
        location[2] = hash_func2(key);
    }
}
void cuckooHash(unsigned *keys, unsigned *table, int amount_keys){
    int keys_size = amount_keys * sizeof(unsigned);
    int table_size = tableSize * sizeof(unsigned) * 2;
    unsigned *cu_keys, *cu_tables;

    cudaMalloc(&cu_keys, keys_size);
    cudaMemcpy(cu_keys, keys, keys_size, cudaMemcpyHostToDevice);
    cudaMalloc(&cu_tables, table_size);
    cudaMemcpy(cu_tables, table, table_size, cudaMemcpyHostToDevice);

    int N = amount_keys;
    int bound = 0;
    while (N >>= 1) ++bound;

    clock_t start;
    double duration;
    start = std::clock();
    cuckooKernel<<<32768,64>>>(cu_keys, cu_tables, amount_keys, bound);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';
}

void shuffle(unsigned *array, int n)
{
    if (n > 1) {
        int i;
    	for (i = 0; i < n - 1; i++) {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
    	}
    }
}

int main(void){
    int amount_keys = 1<<21;
    unsigned *keys, *table;
    keys = (unsigned*)malloc(amount_keys*sizeof(unsigned));
    table = (unsigned*)malloc(tableSize*2*sizeof(unsigned));

    for (int i=0; i<amount_keys; i++){
        keys[i] = rand();
    }
    for (int i=0; i<tableSize*2; i++){
        table[i] = 0;
    }
    // shuffle(keys, amount_keys);
    cuckooHash(keys, table, amount_keys);
    return 0;
}