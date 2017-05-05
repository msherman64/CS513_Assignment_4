#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>



int DIM_LIM = 10;
int MAT_COUNT = 20;
int SEED = 15; //seed for rand

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};


class Matrix : public Managed {
public:
    int row; //number of rows, y
    int col; //number of columns, x
    double* d_data;
    bool isCopy;
    
    Matrix(int columns, int rows) :
        col(columns), row(rows)        
        {cudaMalloc(&d_data, sizeof(double) * columns * rows );}

    Matrix( const Matrix& _orig ) { *this = _orig; isCopy = true;}
    ~Matrix(){if(!isCopy) cudaFree(d_data);}

    __device__ double& getData(int x, int y){
       return d_data[y * col + x]; //vertical position * row length + pos in row
    }
};

void init_matrix(Matrix *mat){
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

__global__ void d_printMat(Matrix *mat)
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

__global__ void d_multMat(Matrix *mat_a, Matrix *mat_b, Matrix *result)
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


__global__ void d_multMat_thd(Matrix *mat_a, Matrix *mat_b, Matrix *result)
{
	//input: [a x b] * [b x c] = [a x c]
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;

	int idx = threadIdx.x; 
	int idy = threadIdx.y; 
	int tmp = 0;
	if(mat_a->row != mat_b->col)
	{
            printf("does not match!");
	}
	else 
	{
		if(idx < dim_a){
		    if(idy < dim_c){
			for(int z=0; z < dim_b; z++){
			    tmp += mat_a->getData(idx,z) * mat_b->getData(z,idy);
			}
			result->getData(idx,idy) = tmp;
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
//	matrix *mat_arr[MAT_COUNT];  //pointer to array of matrices on host
	Matrix **d_mat_arr;  //pointer to array of matrices on device
    cudaMallocManaged(&d_mat_arr, sizeof(Matrix*) * MAT_COUNT); //malloc space for pointer array

	for(int i=0; i<MAT_COUNT; i++){
		d_mat_arr[i] = new Matrix(dim[i],dim[i+1]); //array and matrix are shared
		init_matrix(d_mat_arr[i]);                  //init values
//		cudaMalloc(&d_mat_arr[i], sizeof(matrix));  //make space for matrix object on dev
//		cudaMemcpy(d_mat_arr[i], mat_arr[i], sizeof(matrix), cudaMemcpyHostToDevice); //copy matrix object to dev
	}

	//end generate array

    /*
       starting array is on device
       cudaMalloc space for first result
       store result there

       free i and i+1, point i+1 to result
       iterate
        


       */
/*
    matrix *result;
    matrix *d_result;
    cudaMallocManaged(&d_result, sizeof(matrix));

    for(int i=0; i < MAT_COUNT - 1; i++){

        int dimxn = dim[0];
        int dimyn = dim[i+2];

        //allocate memory for correctly sized result matrix
        result = new matrix(dimxn,dimyn); //we will be leaking host memory here
        cudaMemcpy(d_result, result, sizeof(matrix), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();


        dim3 numBlocks(1);
        dim3 threadsPerBlock(10,10);
        //multiply matrix i, i+1, store in d_result
        d_multMat_thd<<<numBlocks,threadsPerBlock>>>(d_mat_arr[i],d_mat_arr[i+1],d_result);
        
        
        cudaDeviceSynchronize();
        d_printMat<<<1,1>>>(d_result);
        cudaDeviceSynchronize();

//        cudaFree(d_mat_arr[i]); //free source values
//        cudaFree(d_mat_arr[i+1]);
        d_mat_arr[i+1] = d_result;  //point position i+1 to result, for next iteration

        cudaDeviceSynchronize();
        d_printMat<<<1,1>>>(d_mat_arr[i+1]);
        cudaDeviceSynchronize();


    }

        cudaFree(d_result); //free device copy
        delete result;  //free host copy, includes destructor for device data member

*/

	printf("finished!\n");
	return 0;
}
