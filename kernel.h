#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#define tableSize int(16777216 * 1.2)
#define extraSpace 262144
#define EMPTY 0
#define HIT 1

// #define PSEUDO(i) (((i*2654435761)>>32)%3)&1
#define PSEUDO(i) i&1

// #define hash_func0(i) ((81*i+63)%116743349)&(tableSize-1)
// #define hash_func1(i) ((68*i+46)%116743349)&(tableSize-1)
// #define hash_func2(i) ((67*i+22)%116743349)&(tableSize-1)

#define hash_func0(i) ((81*i+63)%116743349)%tableSize
#define hash_func1(i) ((68*i+46)%116743349)%tableSize
#define hash_func2(i) ((67*i+22)%116743349)%tableSize

#define hash_aux(i) ((33*i+87)%116743349)&(extraSpace-1)

void cuckooHash(int *keys, int *table, int N);

#endif