#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#define tableSize 16777216
#define EMPTY 0

#define PSEUDO(i) i*2654435761 % 2
#define hash_func0(i) ((81*i+63)%1144153)&(tableSize-1)
#define hash_func1(i) ((68*i+46)%1144153)&(tableSize-1)
#define hash_func2(i) ((67*i+22)%1144153)&(tableSize-1)

void cuckooHash(int *keys, int *table, int N);

#endif