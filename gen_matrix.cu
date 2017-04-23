#include <stdlib.h> //for rand
#include <iostream> // for cout
using std::cout;

int DIM_LIM = 100;
int MAT_COUNT = 10;

int SEED = 10; //seed for rand

//generate random sized matrices, add pointer to each, to vector
//template <typename T> //handle multiple types
class matrix {
public:
    int row; //number of rows, y
    int col; //number of columns, x
    double* data;

    matrix(int columns, int rows) :
        col(columns), row(rows),
        data(new double[col * row])
        {}

    double& getdata(int x, int y){
       return data[y * col + x]; //vertical position * row length + pos in row
    };
};

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

    /* steps for getting matrices onto device
       create data on host
       malloc array of pointers to matrices on device
       malloc each matrix on device
       malloc each matrix's data on device
       copy matrix 
       copy data pointer
       copy data */

    //get pointer to array of initialized matrices on host
    matrix **mat = initialize(); 
    // pointer to array of matrices on device
    matrix **d_mat; 

    //allocate space on device for array of pointers
    cudaMalloc(&d_mat, MAT_COUNT * sizeof(matrix*)); 
    // temporary array of pointers to each matrix on device, for malloc
    matrix *d_tmp_mat[MAT_COUNT];
    double *d_mat_data[MAT_COUNT]; //pointer to each matrice's data, for malloc

    for(int i = 0; i < MAT_COUNT; i++){
        cudaMalloc(&d_tmp_mat[i], sizeof(matrix)); //allocate each matrix object
        cudaMemcpy(d_tmp_mat[i], &mat[i], sizeof(matrix),
                cudaMemcpyHostToDevice); //copy matrix object from host to device
        cudaMemcpy(&d_mat[i], &d_tmp_mat[i], sizeof(matrix *),
                cudaMemcpyHostToDevice); //copy pointer to device matrix into place

        cudaMalloc(&d_mat_data[i], sizeof(double *) * mat[0]->row * mat[0]->col);
        cudaMemcpy(&d_mat_data[i], mat[i]->data, 
                sizeof(double *) * mat[i]->row * mat[i]->col,
                cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_tmp_mat[i]->data), &d_mat_data[i], sizeof(double *),
                cudaMemcpyHostToDevice);

    }







    //debug by printing size and elements of each matrix in mat
    for(int z = 0; z < MAT_COUNT; z++){
        int dimxn = mat[z]->col;
        int dimyn = mat[z]->row;
        std::cout << dimxn <<" ";
        std::cout << dimyn <<" ";
        std::cout << "\n";
        for(int y = 0; y<dimyn; y++){
            for(int x = 0; x<dimxn; x++){
                cout << mat[z]->getdata(x,y) << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }



};
