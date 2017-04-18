#include <stdlib.h> //for rand
#include <iostream> // for cout
#include <vector>  //for vector

int DIMX_LIM = 10;
int DIMY_LIM = 10;
int MAT_COUNT = 100;

//generate random sized matrices, add pointer to each, to vector


int main(){

    int* mat[MAT_COUNT]; //pointer to pointer to int

    for(int z = 0; z < MAT_COUNT; z++){

        int dimx = DIMX_LIM;
        int dimy = DIMY_LIM;

        mat[z] = (int *)malloc(sizeof(int) * dimx * dimy);
        std::cout << mat[z] <<" ";

//        int *mat_tmp = mat[z]; //must specify size here
//        std::cout << mat_tmp << "\n";

        for(int i = 0; i < dimx; i++){
            for(int j = 0; j < dimy; j++){
                (int(*)[dimx][dimy])(mat[z])[i][j] = 1;
            }
        } 
//         
//
//        for(int i = 0; i < dimx; i++){
//            for(int j = 0; j < dimy; j++){
//                std::cout << mat_tmp[i][j] << " ";
//            }
//            std::cout << "\n";
//        }
    }











};
