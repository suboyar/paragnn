#include <stddef.h>
#include <stdio.h>

#include "print.h"

// TODO: pointer of array decays


void print_farr(double *arr, size_t len)
{
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%f", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_darr(int *arr, size_t len)
{
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%d", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_zuarr(size_t *arr, size_t len)
{
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%zu", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}

void print_sarr(char **arr, size_t len)
{
    printf("[");
    for (size_t i = 0; i < len; i++) {
        printf("%s", arr[i]);
        if (i < len-1) printf(", ");
    }
    printf("]\n");
}
