#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>



int DIM_LIM = 32;
int MAT_COUNT = 1000;
int SEED = 15; //seed for rand
double INIT_VAL = 0.06;

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
	Matrix **d_mat_arr;  //pointer to array of matrices on device
    cudaMallocManaged(&d_mat_arr, sizeof(Matrix*) * MAT_COUNT); //malloc space for pointer array

	for(int i=0; i<MAT_COUNT; i++){
		d_mat_arr[i] = new Matrix(dim[i],dim[i+1]); //array and matrix are shared
		init_matrix(d_mat_arr[i]);                  //init values
        //printf("matrix %d size %dx%d\n", i, d_mat_arr[i]->col, d_mat_arr[i]->row);
	}

	//end generate array

    /*
       starting array is on device
       cudaMalloc space for first result
       store result there

       free i and i+1, point i+1 to result
       iterate
       */
    

    Matrix *d_result = chain_sequential(d_mat_arr, MAT_COUNT);
    cudaDeviceSynchronize();
    d_printMat<<<1,1>>>(d_result);
    cudaDeviceSynchronize();

	printf("finished!\n");
	return 0;
}
