#include <stdio.h>

int DIM_LIM = 100;
int MAT_COUNT = 2;
int SEED = 10; //seed for rand

class matrix {
public:
    int row; //number of rows, y
    int col; //number of columns, x
    double* data;
    
    matrix(int columns, int rows) :
        col(columns), row(rows),
        data(new double[col * row])
        {}

    __host__ __device__ double& getdata(int x, int y){
       return data[y * col + x]; //vertical position * row length + pos in row
    };
};


__global__ void d_printMat(matrix *mat)
{   
        int dimxn = mat->col;
        int dimyn = mat->row;
        printf("Dim x %d, Dim y %d\n", dimxn, dimyn);
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                printf("%lf ", mat->getdata(x,y));
            }
            printf("\n");
        }
        printf("\n");
}
__host__ void printMat(matrix *mat)
{   
        int dimxn = mat->col;
        int dimyn = mat->row;
        printf("Dim x %d, Dim y %d\n", dimxn, dimyn);
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                printf("%lf ", mat->getdata(x,y));
            }
            printf("\n");
        }
        printf("\n");
}

matrix** initialize(){
    srand(SEED); //init random gen
    int dim[MAT_COUNT + 1]; //stores matrix sizes
    for(int z = 0; z <= MAT_COUNT; z++){
        dim[z] = rand()%DIM_LIM + 1;//random between 1 and limit
    }

    //declare matrix array as pointer
    matrix **mat = (matrix **)malloc(MAT_COUNT * sizeof(matrix*));
    for(int z = 0; z < MAT_COUNT; z++){
        //each matrix shares a dimension with the previous
        int dimx = dim[z];
        int dimy = dim[z+1];

        mat[z] = new matrix(dimx,dimy); //dimx columns, dimy rows
        for(int x = 0; x<dimx; x++){
            for(int y = 0; y<dimy; y++){
                //TODO change to random double
                mat[z]->getdata(x,y) = 5; //initialize each element
            }
        }
    }
    return mat;
}

int main(){

    matrix **mat_arr = initialize();

    matrix *mat = mat_arr[0];

    printMat(mat);

    matrix *d_mat;
    cudaMalloc(&d_mat, sizeof(matrix));
    cudaMemcpy(d_mat, mat, sizeof(matrix),
            cudaMemcpyHostToDevice);
    double *mat_data;
    cudaMalloc(&mat_data, sizeof(double) * mat->col * mat->row);
    cudaMemcpy(mat_data, mat->data, sizeof(double) * mat->col * mat->row,
            cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_mat->data),&mat_data, sizeof(double *),
            cudaMemcpyHostToDevice);

    d_printMat<<<1,1>>>(d_mat);
    cudaDeviceSynchronize();


}

