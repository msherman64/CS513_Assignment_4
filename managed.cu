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
int MODE = -1;


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


__device__ void d_multMat_thd(Matrix *mat_a, Matrix *mat_b, Matrix *result)
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

__global__ void d_multmat_pair(Matrix *mat_a, Matrix *mat_b, Matrix *result){
    d_multMat_thd(mat_a, mat_b, result);
}

Matrix * chain_sequential(Matrix **mat_arr, int count){

    Matrix *d_result; //pointer to result matrix
    gpuErrchk(cudaMallocManaged(&d_result, sizeof(Matrix))); //pointer valid on host and device

    for(int i=0; i < count - 1; i++){

        int dimxn = mat_arr[i]->col;
        int dimyn = mat_arr[i+1]->row;

        //allocate memory for correctly sized result matrix
        d_result = new Matrix(dimxn,dimyn); //we will be leaking host memory here

        //multiply matrix i, i+1, store in d_result
//        d_multMat<<<1,1>>>(mat_arr[i],mat_arr[i+1],d_result);
        dim3 threaddim(DIM_LIM,DIM_LIM);
        d_multmat_pair<<<1,threaddim>>>(mat_arr[i],mat_arr[i+1],d_result);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk(cudaDeviceSynchronize());
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

int largestPowTwo(int num)
{
    int i = 1;
    while(i < num){
        i = i * 2;
    }
    return i/2;

}


Matrix * multmat_accum(Matrix *accum, Matrix *step){
    int dimxn = accum->col;
    int dimyn = step->row;
	
	if(accum->row != step->col){
            printf("does not match!");
            return NULL;
	}
    //allocate memory for correctly sized result matrix
    Matrix *d_result = new Matrix(dimxn,dimyn);

    dim3 threads(DIM_LIM,DIM_LIM);
    d_multmat_pair<<<1,threads>>>(accum, step, d_result);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());

//    delete accum;
//    delete step;
    return d_result;
}




__global__ void d_multmat_chain(Matrix **mat_arr, Matrix **mat_result, int count){

   //get matrix pair 
	int idx = blockIdx.x; 
//    d_multMat(mat_arr[2 * idx], mat_arr[2 * idx +1], mat_result[idx]);
    d_multMat_thd(mat_arr[2 * idx], mat_arr[2 * idx +1], mat_result[idx]);

}

Matrix * chain_tree(Matrix **mat_arr, int count){
    /* do k/2 multiplications, then store results
       repeat with k/2/2, etc, until only one result
       */
        if(count == 1){
            return mat_arr[0];
        } else if(isPowerOfTwo(count))
        {
            int result_size = count/2;
            dim3 threaddim(DIM_LIM,DIM_LIM);
            Matrix **d_result; //pointer to result matrix
            gpuErrchk(cudaMallocManaged(&d_result, sizeof(Matrix*) * result_size)); //pointer valid on host and device

            for(int j = 0; j < result_size; j++){
                int dimxn = mat_arr[2 * j]->col;
                int dimyn = mat_arr[(2 * j) + 1]->row;
                d_result[j] = new Matrix(dimxn, dimyn);
            }
            d_multmat_chain<<<result_size,threaddim>>>(mat_arr, d_result, result_size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            for(int mat_index = 0; mat_index < count; mat_index++){
                delete mat_arr[mat_index];
            }

            //recurse
            if(result_size == 1){
                Matrix *result = d_result[0];
                gpuErrchk(cudaFree(d_result));
                return result;
            }
            else{
                Matrix *tmp = chain_tree(d_result, result_size);
                //free input data
                gpuErrchk(cudaFree(d_result));

                return tmp;
            }
        } else {
            //call chain_tree on pow2 part
            //call chain_tree on remainder
            //mult results
            //ex, 258, calls 256, calls 2, calls pair
            int lpow2 = largestPowTwo(count);
            printf("chain_tree: largest power of 2 is %d for result size %d\n", lpow2, count);

            Matrix *tmp_a = chain_tree(mat_arr, lpow2);
            printf("chain_tree: finished compute for %d\n", lpow2);

            Matrix *tmp_b = chain_tree(mat_arr + lpow2, count - lpow2);
            printf("chain_tree: finished compute for %d\n", count - lpow2);
            return multmat_accum(tmp_a, tmp_b);
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
    if(argc == 4){
        INIT_VAL = atof(argv[1]);
        MAT_COUNT = atoi(argv[2]);
        printf("main: %d matrices of initial value is %f\n", MAT_COUNT, INIT_VAL);
        printf("main: mode is %s\n", argv[3]);
        
        if(strcmp(argv[3], "seq") == 0)
        {
            MODE = 1;
        } else if(strcmp(argv[3], "tree") == 0)
        {
            MODE = 2;
        }
        if(MODE == -1){       
            printf("main: incorrect mode, must be tree or seq\n");
            return -1;
        }
    } else {
        printf("main: incorrect number of arguments, must be initial value, matrix count, mode = seq or tree\n");
        return -1;
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
    Matrix *total_result; //pointer for total result
    Matrix *chunk_result; //pointer to result for each chunk

    dim3 threaddim(DIM_LIM, DIM_LIM); //threads per block, based on matrix size.

    for(int i = MAT_COUNT; i > 0; i = i - CHUNK_SIZE){
        int count = min(i, CHUNK_SIZE);
        printf("main: chunk loop %d, %d matrices\n", i, count);
        
        Matrix **d_mat_arr = generate(dim_ptr, count); //generate or input step

        printf("main: generated %d matrices\n", count);
        if(MODE == 1){
            chunk_result = chain_sequential(d_mat_arr, count);
            printf("main: computed seq result for chunk %d\n", i);
        } else if(MODE == 2){
            chunk_result = chain_tree(d_mat_arr, count);
            printf("main: computed tree result for chunk %d\n", i);
        }

        dim_ptr += count;
        if(chunk_result){
            if(i == MAT_COUNT){
                total_result = chunk_result;
            }else if(i < MAT_COUNT){ // not on first loop
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
                total_result = multmat_accum(total_result, chunk_result); //tally results
            }
            
            //gpuErrchk( cudaPeekAtLastError() );
            //gpuErrchk( cudaDeviceSynchronize() );
            printf("main: finished printing result for chunk %d\n", i);
        }
        else{
            printf("main: no valid result for chunk %d\n", i);
        } 

    }

    d_printMat<<<1,1>>>(total_result);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());

	return 0;
}
