#include <stdio.h>

int DIM_LIM = 10;
int MAT_COUNT = 20;
int SEED = 15; //seed for rand
    

class matrix {
public:
    int row; //number of rows, y
    int col; //number of columns, x
    double* data;
    
    __host__ __device__ matrix(int columns, int rows) :
        col(columns), row(rows),
        data(new double[columns * rows])
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

__global__ void mat_mult(matrix a, matrix b, matrix *ans){
    if(a.row == b.col){
    
        int iter = a.row; //number of mults needed
        printf("a(%d, %d) b(%d,%d)\n",a.col,a.row,b.col,b.row);
        //ans = new matrix(a.col, b.row);
        printf("result %d rows %d cols\n", ans->row, ans->col);

        for(int x = 0; x < ans->col; x++){
            for(int y = 0; y < ans->row; y++){
                ans->getdata(x,y) = 0; //initialize
                for(int z = 0; z < iter; z++){
                    printf("x %d, y %d\n", x, y);
                    double tmp_a = a.getdata(x,y);
                    double tmp_b = b.getdata(y,x);
                    double tmp_mult = tmp_a * tmp_b;
                    ans->getdata(x,y) += tmp_mult;
                }
                //printf("value at %d %d is %f\n", x,y, ans->getdata(x,y));
            }
        }
    }
    else{
        printf("matrix size mismatch");
    }
};

//note, kernel functions cannot use pass by reference. e.g. duh.
__global__ void test_matrix(matrix **tmp){
    matrix *ans = new matrix(1,1);
    ans->getdata(1,1) = 42;
    printf("row %d, col %d, value %lf\n",
           ans->row, ans->col, ans->getdata(1,1));
    *tmp = ans;
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

matrix* copyMatrixDev(matrix *host){
        matrix *d_mat;
        double *tmp_data;
        cudaMalloc(&d_mat, sizeof(matrix));
        cudaMemcpy(d_mat, host, sizeof(matrix),
                cudaMemcpyHostToDevice);
        cudaMalloc(&tmp_data, sizeof(double) * host->col * host->row);
        cudaMemcpy(tmp_data, host->data, sizeof(double) * host->col * host->row,
                cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_mat->data),&tmp_data, sizeof(double *),
                cudaMemcpyHostToDevice);
        return d_mat;
}

matrix* hostMultMat(matrix *a, matrix *b){
    matrix *result = new matrix(a->col,b->row);
    matrix *d_result = copyMatrixDev(result);
    delete result;

    cudaDeviceSynchronize();
    mat_mult<<<1,1>>>(*a,*b,d_result);

    cudaDeviceSynchronize();
    return d_result;

}


int main(){

    matrix **mat_arr = initialize();
    matrix *d_mat[MAT_COUNT];
    double *mat_data[MAT_COUNT];


    for(int i = 0; i < MAT_COUNT; i++){

        cudaMalloc(&d_mat[i], sizeof(matrix));
        cudaMemcpy(d_mat[i], mat_arr[i], sizeof(matrix),
                cudaMemcpyHostToDevice);
        cudaMalloc(&mat_data[i], sizeof(double) * mat_arr[i]->col * mat_arr[i]->row);
        cudaMemcpy(mat_data[i], mat_arr[i]->data, sizeof(double) * mat_arr[i]->col * mat_arr[i]->row,
                cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_mat[i]->data),&mat_data[i], sizeof(double *),
                cudaMemcpyHostToDevice);
    //    printMat(mat_arr[i]);
    //    d_printMat<<<1,1>>>(d_mat[i]);
    }

        //matrix *d_result = hostMultMat(mat_arr[0], d_mat[0], mat_arr[1], d_mat[1]);
        //d_printMat<<<1,1>>>(d_result);
        //cudaDeviceSynchronize();


    matrix *d_result;
//    for(int i = 0; i < MAT_COUNT-1; i++){
    for(int i = 0; i < 1; i++){
        d_result = hostMultMat(mat_arr[i], mat_arr[i+1]);
        d_printMat<<<1,1>>>(d_result);
        cudaFree(d_result);

    }




}

