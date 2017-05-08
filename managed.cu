#include <stdio.h> //for fprintf, fscanf, printf, etc.
#include <float.h> //for DBL_MAX

// CUDA runtime
#include <cuda_runtime.h> //various error checking additions

#define MOD_MODE 1 //0 for no modulus, 1 for single modulus around mult + sum, 2 for modulus of mult, then modulus of sum
#define ADD_CONSTANT 7
//constant for architecture
int DIM_LIM = 32;
int SEED = 15; //seed for rand
int CHUNK_SIZE = 2<<14; //memory limit for ilab machine
double fmod_arg = pow(2,52);
//set via argument
long INIT_VAL = 5;
int MAT_COUNT = 1000;
int MODE = -1;
long MODULO = 1<<16;


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
    long* d_data;
    bool isCopy;
    
    Matrix(int columns, int rows) :
        col(columns), row(rows)        
        {cudaMalloc(&d_data, sizeof(long) * columns * rows );}

//    Matrix( const Matrix& _orig ) { *this = _orig; isCopy = true;}
//    ~Matrix(){if(!isCopy) cudaFree(d_data);}
    ~Matrix(){cudaFree(d_data);}

    __device__ long& getData(int x, int y){
       return d_data[y * col + x]; //vertical position * row length + pos in row
    }
};


__global__ void d_printMat(Matrix *mat)
{   
        int dimxn = mat->col;
        int dimyn = mat->row;
        printf("Dim x %d, Dim y %d\n", dimxn, dimyn);
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                //printf("%.10e ", mat->getData(x,y));
                printf("%ld ", mat->getData(x,y));
            }
            printf("\n");
        }
        printf("\n");
}
void printMat(Matrix *mat, FILE *f)
{
        int dimxn = mat->col;
        int dimyn = mat->row;
        size_t data_size = sizeof(long) * dimxn * dimyn;
        fprintf(f, "Dim x %d, Dim y %d\n", dimxn, dimyn);
        long* data = (long *)malloc(data_size);
        cudaMemcpy(data, mat->d_data, data_size, cudaMemcpyDeviceToHost);
        long tmp = 0.;
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                tmp = data[y * dimxn + x]; 
                fprintf(f, "%ld ", tmp);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
}


//Multiply matrices single thread on Device
__device__ void d_multMat(Matrix *mat_a, Matrix *mat_b, Matrix *result, int modulo)
{
	//input: [a x b] * [b x c] = [a x c]
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;

	if(mat_a->row != mat_b->col){
            printf("does not match!");
	}
	else {
		long tmp = 0;
		for(int x=0; x < dim_a; x++){
		    for(int y=0; y < dim_c; y++){
			tmp=0;	
			for(int z=0; z < dim_b; z++){
                switch(MOD_MODE) {
                    case 0 :
                        tmp += mat_a->getData(x,z) * mat_b->getData(z,y) + ADD_CONSTANT;
                        break;
                    case 1 :
                        tmp = (tmp + (mat_a->getData(x,z) * mat_b->getData(z,y))) % modulo + ADD_CONSTANT;
                        break;
                    case 2 :
                        tmp = (tmp + ((mat_a->getData(x,z) * mat_b->getData(z,y)) % modulo)) % modulo + ADD_CONSTANT;
                        break;
                }
			}
			result->getData(x,y)=tmp;
		    }
		}
	}
}

//Multiply matrices multi thread on Device
__device__ void d_multMat_thd(Matrix *mat_a, Matrix *mat_b, Matrix *result, int modulo)
{
	//input: [a x b] * [b x c] = [a x c]
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;

    if(mat_a->row == mat_b->col){
        for(int idx = threadIdx.x; idx < dim_a; idx += blockDim.x){ //block stride x
            for(int idy = threadIdx.y; idy < dim_c; idy += blockDim.y){ //block stride y
                long tmp = 0;
                for(int z=0; z < dim_b; z++)
                {
                    switch(MOD_MODE) {
                        case 0 :
                            tmp += mat_a->getData(idx,z) * mat_b->getData(z,idy) + ADD_CONSTANT;
                            break;
                        case 1 :
                            tmp = tmp + ((mat_a->getData(idx,z) * mat_b->getData(z,idy)) % modulo) + ADD_CONSTANT;
                            break;
                        case 2 :
                            tmp = (tmp + ((mat_a->getData(idx,z) * mat_b->getData(z,idy)) % modulo)) % modulo + ADD_CONSTANT;
                            break;
                    }
                }
                result->getData(idx,idy) = tmp;
            }
        }
    } 
}

//Multiply matrices single thread on Host
void host_multMat(Matrix *mat_a, Matrix *mat_b, Matrix *result, int modulo){
	int dim_a = mat_a->col;
	int dim_b = mat_a->row;
	int dim_c = mat_b->row;
    long tmp = 0;
        
	if(mat_a->row != mat_b->col)
	{
            printf("does not match!");
            return;
	}
        for(int x=0; x<dim_a; x++){
            for(int y=0; y<dim_c; y++){
                for(int z=0; z<dim_b; z++){
//                    tmp = (tmp + mat_a->getDataHost(x,z) * mat_b->getDataHost(z,y)) % modulo + ADD_CONSTANT; 
                }
 //               result->getDataHost(x,y) = tmp;
            }
        }
}

__global__ void d_multmat_pair(Matrix *mat_a, Matrix *mat_b, Matrix *result, int modulo){
    d_multMat_thd(mat_a, mat_b, result, modulo);
}

//Chain in sequential, Individual matrix in parallel
Matrix * chain_sequential(Matrix **mat_arr, int count){

    Matrix *d_result; //pointer to result matrix
    gpuErrchk(cudaMallocManaged(&d_result, sizeof(Matrix))); //pointer valid on host and device
    if(count == 1){
        return mat_arr[0];
    } else {
        for(int i=0; i < count - 1; i++){

            int dimxn = mat_arr[i]->col;
            int dimyn = mat_arr[i+1]->row;

            //allocate memory for correctly sized result matrix
            d_result = new Matrix(dimxn,dimyn); //we will be leaking host memory here

            //multiply matrix i, i+1, store in d_result
    //        d_multMat<<<1,1>>>(mat_arr[i],mat_arr[i+1],d_result);
            dim3 threaddim(DIM_LIM,DIM_LIM);
            d_multmat_pair<<<1,threaddim>>>(mat_arr[i],mat_arr[i+1],d_result, MODULO);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk(cudaDeviceSynchronize());
            delete mat_arr[i];
            delete mat_arr[i+1];
            mat_arr[i+1] = d_result;
        }

        return d_result;
    }
}

Matrix * chain_sequential_onHost(Matrix **mat_arr, int count){
    Matrix *d_result; //pointer to result matrix
    gpuErrchk(cudaMallocManaged(&d_result, sizeof(Matrix))); //pointer valid on host and device
    if(count == 1){
        return mat_arr[0];
    } else {
        for(int i=0; i < count - 1; i++){

            int dimxn = mat_arr[i]->col;
            int dimyn = mat_arr[i+1]->row;
            d_result = new Matrix(dimxn,dimyn);
            //host_multMat(mat_arr[i],mat_arr[i+1],d_result, MODULO);
            delete mat_arr[i];
            delete mat_arr[i+1];
            mat_arr[i+1] = d_result;
        }
        return d_result;
    }
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
    d_multmat_pair<<<1,threads>>>(accum, step, d_result, MODULO);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());

//    delete accum;
//    delete step;
    return d_result;
}




__global__ void d_multmat_chain(Matrix **mat_arr, Matrix **mat_result, int count, int modulo){

   //get matrix pair 
	int idx = blockIdx.x; 
//    d_multMat(mat_arr[2 * idx], mat_arr[2 * idx +1], mat_result[idx]);
    d_multMat_thd(mat_arr[2 * idx], mat_arr[2 * idx +1], mat_result[idx], modulo);

}

//Chain in parallel, Individual matrix in sequential
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
            d_multmat_chain<<<result_size,threaddim>>>(mat_arr, d_result, result_size, MODULO);
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




void init_from_file(Matrix *mat, FILE *fp){
	int x_dim = mat->row;
	int y_dim = mat->col;
    long arr[x_dim][y_dim];
	for(int x = 0; x < x_dim; x++){
		for(int y = 0; y < y_dim; y++){
            fscanf(fp, "%ld", &(arr[x][y])); //fscanf requires ld for double
		}
	}
	gpuErrchk(cudaMemcpy(mat->d_data, arr, sizeof(arr), cudaMemcpyHostToDevice));
}

Matrix **read_file(int *dim, int count, FILE *fp){
	Matrix **d_mat_arr;  //pointer to array of matrices on device
    cudaMallocManaged(&d_mat_arr, sizeof(Matrix*) * count); //malloc space for pointer array

	for(int i = 0; i < count; i++){
		d_mat_arr[i] = new Matrix(dim[i], dim[i+1]); //array and matrix are shared
		init_from_file(d_mat_arr[i], fp);                  //init values
	}
    return d_mat_arr;    
}


int main(int argc, char *argv[]){
    if(argc == 3){
        //INIT_VAL = atof(argv[1]);
        //MAT_COUNT = atoi(argv[2]);
        //printf("main: %d matrices of initial value is %f\n", MAT_COUNT, INIT_VAL);
        printf("main: mode is %s\n", argv[1]);
        
        if(strcmp(argv[1], "seq") == 0)
        {
            MODE = 1;
        } else if(strcmp(argv[1], "tree") == 0)
        {
            MODE = 2;
        }
        if(MODE == -1){       
            printf("main: incorrect mode, must be tree or seq\n");
            return -1;
        }
    } else {
        //printf("main: incorrect number of arguments, must be initial value, matrix count, mode = seq or tree\n");
        printf("main: incorrect number of arguments, must be , mode = seq or tree, followed by output filename\n");
        return -1;
    }



    FILE *fp_in = fopen("input.txt", "r");
    int dim_arr_count = 0;
    fscanf(fp_in, "%d", &dim_arr_count);
    
    int dim[dim_arr_count]; //stores matrix sizes
    for(int z = 0; z < dim_arr_count; z++){
        fscanf(fp_in, "%d", &dim[z]);
    }

    MAT_COUNT = dim_arr_count - 1;



	//generate array of matrices from size array

    int *dim_ptr = dim;
    Matrix *total_result; //pointer for total result
    Matrix *chunk_result; //pointer to result for each chunk

    dim3 threaddim(DIM_LIM, DIM_LIM); //threads per block, based on matrix size.

    for(int i = MAT_COUNT; i > 0; i = i - CHUNK_SIZE){
        int count = min(i, CHUNK_SIZE);
        printf("main: chunk loop %d, %d matrices\n", i, count);
        
        Matrix **d_mat_arr = read_file(dim_ptr, count, fp_in); //generate or input step

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
    FILE *fp = fopen(argv[2], "w");
    if (fp != NULL)
    {
        printMat(total_result, fp);
        fclose(fp);
    }
    d_printMat<<<1,1>>>(total_result);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());

	return 0;
}
