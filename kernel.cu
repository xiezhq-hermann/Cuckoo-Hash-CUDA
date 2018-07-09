#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <random>
#include <set>

using namespace std;

__device__ int linearProb(unsigned key, unsigned* auxiliary, unsigned location){
    for (int i=0; i<extraSpace; i++){
        key = atomicExch(&auxiliary[location], key);
        if (key != EMPTY){
            location = (location+1)&(extraSpace-1);
        }else{
            return HIT;
        }
    }
    printf("failed\n");
    return EMPTY;
}

__global__ void cuckooKernel(unsigned* keys, unsigned* table, int amount_keys, int bound, unsigned* auxiliary) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index >= amount_keys){
        return;
    }

    unsigned location[3];
    // int indicator = 0;
    unsigned key = keys[index];
    unsigned random_entry = PSEUDO(key);
    location[0] = hash_func0(key);
    // unsigned location = hash_func0(key);
#pragma unroll
    for (int j = 0; j < bound; j++){
        // key = atomicExch(&table[location*2 + random_entry], key);
        key = atomicExch(&table[location[j%3]*2 + random_entry], key);
        if (key == EMPTY){
            return;
        }
        // switch(indicator){
        //     case 0:
        //         location = hash_func1(key);
        //         indicator = 1;
        //     case 1:
        //         location = hash_func2(key);
        //         indicator = 2;
        //     case 2:
        //         location = hash_func0(key);
        //         indicator = 0;
        // }
        location[0] = hash_func0(key);
        location[1] = hash_func1(key);
        location[2] = hash_func2(key);
        continue;
    }
    linearProb(key, auxiliary, hash_aux(key));
}



__global__ void lookUpKernel(unsigned* keys, unsigned* table, int amount_keys) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= amount_keys){
        return;
    }
    unsigned key = keys[index];
    unsigned entry = key&1;
    unsigned location[3];
    unsigned target;
    location[0] = hash_func0(key)*2+entry;
    location[1] = hash_func1(key)*2+entry;
    location[2] = hash_func2(key)*2+entry;
    for (int i=0; i<3; i++){
        if(atomicCAS(&table[location[i]], key, key) == key){
            return;
        }
    }
    target = hash_aux(key);
    unsigned candidate;
    for (int i=0; i<extraSpace; i++){
        candidate = atomicCAS(&table[target], key, key);
        if (candidate == EMPTY | candidate == key){
            return;
        }
        target = (target+1)&(extraSpace-1);
    }
    return ;
}
void cuckooHash(unsigned *keys, unsigned *table, int amount_keys, unsigned *auxiliary){
    int keys_size = amount_keys * sizeof(unsigned);
    int table_size = tableSize * sizeof(unsigned) * 2;
    unsigned *cu_keys, *cu_tables, *cu_aux;
    float elapsedTime;

    cudaDeviceReset();

    cudaMalloc(&cu_keys, keys_size);
    cudaMemcpy(cu_keys, keys, keys_size, cudaMemcpyHostToDevice);
    cudaMalloc(&cu_tables, table_size);
    cudaMemcpy(cu_tables, table, table_size, cudaMemcpyHostToDevice);
    cudaMalloc(&cu_aux, extraSpace);
    cudaMemcpy(cu_aux, auxiliary, extraSpace, cudaMemcpyHostToDevice);

    int N = amount_keys;
    int bound = 0;
    while (N >>= 1) ++bound;
    bound = bound << 3;
    printf("the bound is: %d\n", bound);

    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start, 0);
    cuckooKernel<<<(amount_keys+63)/64,64>>>(cu_keys, cu_tables, amount_keys, bound, cu_aux);
    cudaEventRecord(cu_stop, 0);
    cudaEventSynchronize(cu_stop);
    cudaEventElapsedTime(&elapsedTime, cu_start, cu_stop);
    cout<<"The time cost is "<< elapsedTime <<" ms\n";
    cout<<"The return status is "<<cudaGetLastError() << "\n";


    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<unsigned> dis(1, 4294967295);

    for (int i=-1; i<=10; i++){
        for (int j=0; j<i*amount_keys/10; j++){
            keys[dis(gen)&(amount_keys-1)] = dis(gen);
        }
        cudaMemcpy(cu_keys, keys, keys_size, cudaMemcpyHostToDevice);

        cudaEventRecord(cu_start, 0);
        lookUpKernel<<<(amount_keys+63)/64,64>>>(cu_keys, cu_tables, amount_keys);
        cudaEventRecord(cu_stop, 0);
        cudaEventSynchronize(cu_stop);
        cudaEventElapsedTime(&elapsedTime, cu_start, cu_stop);

        cout<<"The time cost is "<< elapsedTime <<" ms\n";
        cout<<"The return status is "<<cudaGetLastError() << "\n";
    }
}

int main(void){
    int amount_keys = 1<<24;
    unsigned *keys, *table, *auxiliary;
    keys = (unsigned*)malloc(amount_keys*sizeof(unsigned));
    table = (unsigned*)malloc(tableSize*2*sizeof(unsigned));
    auxiliary = (unsigned*)malloc(extraSpace*sizeof(unsigned));

    set<unsigned> distinctRandom;

    for (int i=0; i<tableSize*2; i++){
        table[i] = 0;
    }

    for (int i=0; i<extraSpace; i++){
        auxiliary[i] = 0;
    }

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<unsigned> dis(1, 4294967295);
    for (int i=0; i<amount_keys; i++){
        // cout<<"at the entry: "<< i <<"\n";
        unsigned cc = dis(gen);
        if (distinctRandom.find(cc) != distinctRandom.end()){
            i--;
        }else{
            distinctRandom.insert(cc);
            keys[i] = cc;
        }
    }
    cout<<"the size of set is "<<distinctRandom.size() << "\n";

    cuckooHash(keys, table, amount_keys, auxiliary);
    return 0;
}