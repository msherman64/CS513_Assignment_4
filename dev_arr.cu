#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>



int DIM_LIM = 10;
int MAT_COUNT = 20;
int SEED = 15; //seed for rand


class matrix {
public:
    int row; //number of rows, y
    int col; //number of columns, x
    double* d_data;
    bool isCopy;
    
    matrix(int columns, int rows) :
        col(columns), row(rows)        
        {cudaMalloc(&d_data, sizeof(double) * columns * rows );}

    matrix( const matrix& _orig ) { *this = _orig; isCopy = true;}
    ~matrix(){if(!isCopy) cudaFree(d_data);}

    __device__ double& getdata(int x, int y){
       return d_data[y * col + x]; //vertical position * row length + pos in row
    }
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

void init_matrix(matrix * mat){
	int x_dim = mat->row;
	int y_dim = mat->col;
        double arr[x_dim][y_dim];

	for(int x = 0; x < x_dim; x++){
		for(int y = 0; y < y_dim; y++){
			arr[x][y] = 5;
		}
	}

	cudaMemcpy(mat->d_data, arr, sizeof(arr), cudaMemcpyHostToDevice);
}

int main(){

	matrix *foo = new matrix(5,4);
	init_matrix (foo);
	
	matrix *d_foo;
	cudaMalloc(&d_foo, sizeof(matrix)); //1 allocate device struct
	cudaMemcpy(d_foo, foo, sizeof(matrix), cudaMemcpyHostToDevice); //copy matrix object


	d_printMat<<<1,1>>>(d_foo);
	cudaDeviceSynchronize();




	








	printf("finished!");
	return 0;
}
