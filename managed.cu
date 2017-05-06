#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>


//constant for architecture
int DIM_LIM = 32;
int SEED = 15; //seed for rand
int CHUNK_SIZE = 2<<14; //memory limit for ilab machine
//set via argument
double INIT_VAL = 0.06;
int MAT_COUNT = 1000;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    gpuErrchk(cudaMallocManaged(&ptr, len));
    gpuErrchk(cudaDeviceSynchronize());
    return ptr;
  }

  void operator delete(void *ptr) {
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(ptr));
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

//    Matrix( const Matrix& _orig ) { *this = _orig; isCopy = true;}
//    ~Matrix(){if(!isCopy) cudaFree(d_data);}
    ~Matrix(){cudaFree(d_data);}

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
			arr[x][y] = INIT_VAL;
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
                printf("%.10e ", mat->getData(x,y));
            }
            printf("\n");
        }
        printf("\n");
}

__device__ void d_multMat(Matrix *mat_a, Matrix *mat_b, Matrix *result)
{
	//input: [a x b] * [b x c] = [a x c]
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;

	if(mat_a->row != mat_b->col){
            printf("does not match!");
	}
	else {
		double tmp = 0;
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
	double tmp = 0;
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

Matrix * chain_sequential(Matrix **mat_arr, int count){

    Matrix *d_result; //pointer to result matrix
    cudaMallocManaged(&d_result, sizeof(Matrix)); //pointer valid on host and device

    for(int i=0; i < count - 1; i++){

        int dimxn = mat_arr[i]->col;
        int dimyn = mat_arr[i+1]->row;

        //allocate memory for correctly sized result matrix
        d_result = new Matrix(dimxn,dimyn); //we will be leaking host memory here

        //multiply matrix i, i+1, store in d_result
//        d_multMat<<<1,1>>>(mat_arr[i],mat_arr[i+1],d_result);
        dim3 threaddim(DIM_LIM,DIM_LIM);
        d_multMat_thd<<<1,threaddim>>>(mat_arr[i],mat_arr[i+1],d_result);
        cudaDeviceSynchronize(); //must sync before host operation on device memory 
        delete mat_arr[i];
        delete mat_arr[i+1];
        mat_arr[i+1] = d_result;
    }

    return d_result;

}

//check if value is power of two, using bitwise AND
bool isPowerOfTwo(ulong x)
{
    return (x & (x - 1)) == 0;
}

__global__ void d_multmat_chain(Matrix **mat_arr, Matrix **mat_result, int count){

   //get matrix pair 
	int idx = blockIdx.x; 
    d_multMat(mat_arr[2 * idx], mat_arr[2 * idx +1], mat_result[idx]);

}

Matrix * chain_tree(Matrix **mat_arr, int count){
    /* do k/2 multiplications, then store results
       repeat with k/2/2, etc, until only one result
       */
        if(isPowerOfTwo(count))
        {
            int result_size = count/2;
            Matrix **d_result; //pointer to result matrix
            gpuErrchk(cudaMallocManaged(&d_result, sizeof(Matrix*) * result_size)); //pointer valid on host and device

            for(int j = 0; j < result_size; j++){
                int dimxn = mat_arr[2 * j]->col;
                int dimyn = mat_arr[(2 * j) + 1]->row;
                d_result[j] = new Matrix(dimxn, dimyn);
            }
            d_multmat_chain<<<result_size,1>>>(mat_arr, d_result, result_size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            //free input data
            for(int mat_index = 0; mat_index < count; mat_index++){
                delete mat_arr[mat_index];
            }
            gpuErrchk(cudaFree(mat_arr));

            //recurse
            if(result_size == 1){
                Matrix *result = d_result[0];
                gpuErrchk(cudaFree(d_result));
                return result;
            }
            else{
                Matrix *tmp = chain_tree(d_result, result_size);
/*                for(int mat_index = 1; mat_index < result_size; mat_index++){
                    delete d_result[mat_index];
                }
                gpuErrchk(cudaFree(d_result)); */

                return tmp;
            }
        }
        else
        {
            return NULL;
        }
}



Matrix **generate(int *dim, int count){
	Matrix **d_mat_arr;  //pointer to array of matrices on device
    cudaMallocManaged(&d_mat_arr, sizeof(Matrix*) * count); //malloc space for pointer array

	for(int i = 0; i < count; i++){
		d_mat_arr[i] = new Matrix(dim[i], dim[i+1]); //array and matrix are shared
		init_matrix(d_mat_arr[i]);                  //init values
        //printf("matrix %d size %dx%d\n", i, d_mat_arr[i]->col, d_mat_arr[i]->row);
	}
    return d_mat_arr;    
}

int main(int argc, char *argv[]){
    if(argc == 3){
        INIT_VAL = atof(argv[1]);
        MAT_COUNT = atoi(argv[2]);
        printf("%d matrices of initial value is %f\n", MAT_COUNT, INIT_VAL);
    }



	//initialize random number gen, get array sizes
    srand(SEED); //init random gen
    int dim[MAT_COUNT + 1]; //stores matrix sizes
    for(int z = 0; z <= MAT_COUNT; z++){
        dim[z] = rand()%DIM_LIM + 1;//random between 1 and limit
    }
	//end initialize

	//generate array of matrices from size array

    int *dim_ptr = dim;
    for(int i = MAT_COUNT; i > 0; i = i - CHUNK_SIZE){
        int count = min(i, CHUNK_SIZE);
        printf("main: chunk loop %d, %d matrices\n", i, count);

        Matrix **d_mat_arr = generate(dim_ptr, count);

        //Matrix *d_result = chain_sequential(d_mat_arr, MAT_COUNT);
        Matrix *d_result = chain_tree(d_mat_arr, count);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk(cudaDeviceSynchronize());

        if(d_result){
            d_printMat<<<1,1>>>(d_result);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            printf("finished!\n");
        }
        else{
            printf("no valid result\n");
        }
        dim_ptr += count;
    }
	return 0;
}
