#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>



int DIM_LIM = 10;
int MAT_COUNT = 10;
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

    __device__ double& getData(int x, int y){
       return d_data[y * col + x]; //vertical position * row length + pos in row
    }
};

void init_matrix(matrix * mat){
	int x_dim = mat->row;
	int y_dim = mat->col;
        double arr[x_dim][y_dim];

	for(int x = 0; x < x_dim; x++){
		for(int y = 0; y < y_dim; y++){
			arr[x][y] = 1;
		}
	}

	cudaMemcpy(mat->d_data, arr, sizeof(arr), cudaMemcpyHostToDevice);
}

__global__ void d_printMat(matrix *mat)
{   
        int dimxn = mat->col;
        int dimyn = mat->row;
        printf("Dim x %d, Dim y %d\n", dimxn, dimyn);
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                printf("%lf ", mat->getData(x,y));
            }
            printf("\n");
        }
        printf("\n");
}

__global__ void d_multMat(matrix *mat_a, matrix *mat_b, matrix *result)
{
	//input: [a x b] * [b x c] = [a x c]
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;

	if(mat_a->row != mat_b->col){
            printf("does not match!");
	}
	else {

		int tmp=0;
		for(int x=0; x < dim_a; x++){
		    for(int y=0; y < dim_c; y++){
			tmp=0;	
			for(int z=0; z < dim_b; z++){
			    tmp += mat_a->getData(x,z) * mat_b->getData(z,y);
			}
			result->getData(x,y)=tmp;
		    }
		}
			    
	}







}


int main(){
	//initialize random number gen, get array sizes
    	srand(SEED); //init random gen
        int dim[MAT_COUNT + 1]; //stores matrix sizes
        for(int z = 0; z <= MAT_COUNT; z++){
            dim[z] = rand()%DIM_LIM + 1;//random between 1 and limit
        }
	//end initialize

	//generate array of matrices from size array
	matrix *mat_arr[MAT_COUNT];
	matrix *d_mat_arr[MAT_COUNT];

	for(int i=0; i<MAT_COUNT; i++){
		mat_arr[i] = new matrix(dim[i],dim[i+1]);
		init_matrix(mat_arr[i]);
		cudaMalloc(&d_mat_arr[i], sizeof(matrix));
		cudaMemcpy(d_mat_arr[i], mat_arr[i], sizeof(matrix), cudaMemcpyHostToDevice);
	//	d_printMat<<<1,1>>>(d_mat_arr[i]);
	//	cudaDeviceSynchronize();
	}

	cudaDeviceSynchronize();
	//end generate array


	for(int i = 0; i < MAT_COUNT-1; i++){

		int dimxn = mat_arr[i]->col;
		int dimyn = mat_arr[i+1]->row;
		//printf("Dim x %d, Dim y %d\n", dimxn, dimyn);

		matrix *result = new matrix(dimxn,dimyn);
		matrix *d_result;
		cudaMalloc(&d_result, sizeof(matrix));
		cudaMemcpy(d_result, result, sizeof(matrix), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		d_multMat<<<1,1>>>(d_mat_arr[i],d_mat_arr[i+1],d_result);
		cudaDeviceSynchronize();
		d_printMat<<<1,1>>>(d_result);
		cudaDeviceSynchronize();

		cudaFree(d_result); //free device copy
		delete result;  //free host copy, includes destructor for device data member

	}



	printf("finished!");
	return 0;
}
