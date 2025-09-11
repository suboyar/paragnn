#include <stddef.h>
#include <stdio.h>

#include "print.h"

// TODO: pointer of array decays
#define ARRAY_LEN(array) (sizeof(array)/sizeof(array[0]))

void print_arr_f(double *arr)
{
    size_t len = ARRAY_LEN(arr);
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%f", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_arr_d(int *arr)
{
    size_t len = ARRAY_LEN(arr);
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%d", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_arr_zu(size_t *arr)
{
    size_t len = ARRAY_LEN(arr);
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%zu", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_arr_s(char **arr)
{
    size_t len = ARRAY_LEN(arr);
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%s", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}
