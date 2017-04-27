// Multiply two matrices A * B = C
 
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define WA 3   // Matrix A width
#define HA 3   // Matrix A height
#define WB 3   // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height
 
// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void multiply(float* h_A, float* h_B, float* C, int size_C)
{
    float sum;
    for (int row=0; row < size_C; row++){
        for (int col=0; col < size_C; col++){
            sum = 0.f;
            for (int n=0; n<size_C; n++){
                sum += h_A[row*size_C+n]*h_B[n*size_C+col];
            }
            C[row*size_C+col] = sum;
        }
    }
}
 
/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main(int argc, char** argv)
{
 
    // set seed for rand()
    srand(2006);
 
    // 1. allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
 
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);
 
    // 2. initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
  
    // 3. print out A and B
    printf("\n\nMatrix A\n");
    for(int i = 0; i < size_A; i++)
    {
       printf("%f ", h_A[i]);
       if(((i + 1) % WA) == 0)
          printf("\n");
    }
 
    printf("\n\nMatrix B\n");
    for(int i = 0; i < size_B; i++)
    {
       printf("%f ", h_B[i]);
       if(((i + 1) % WB) == 0)
          printf("\n");
    }
 
    // 4. allocate host memory for the result C
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);
    
    // 5. perform the calculation
    multiply(h_A, h_B, h_C, size_C);
 
    // 6. print out the results
    printf("\n\nMatrix C (Results)\n");
    for(int i = 0; i < size_C; i++)
    {
       printf("%f ", h_C[i]);
       if(((i + 1) % WC) == 0)
          printf("\n");
    }
    printf("\n");
 
    // 7. clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
}
