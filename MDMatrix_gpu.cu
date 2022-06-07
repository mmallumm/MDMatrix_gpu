#include "MDMatrix_gpu.h"
#include "..\..\Core\MDMatrix.h"
#include <math.h>
#include <iostream>

#define MAXGRIDDIM 65535
#define imin(a,b) (a<b?a:b)
#define imax(a, b) (a>b?a:b)

namespace RDK {

    //--------------------------------kernels' block--------------------

    //FillMatrixValue----------
    template<class T>
    __global__ void _FillMatrixValue(T* data, T defValue, int rows, int cols) {
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        int step = gridDim.x * blockDim.x;
        int size = rows * cols;

        //in case of kernels' number > matrix' size
        if(step >= size) {
            if(id < size) {
                data[id] = defValue;
            }
        } else {
            for(int c = id; c < size; c += step) {
                if(c < size) {
                    data[c] = defValue;
                }
            }
        }
    }

    //FillMatrixFrom----------
    template<class T>
    __global__ void _FillMatrixFrom(T* dest_data, T* src_data, int rows, int cols) {
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        int step = gridDim.x * blockDim.x;
        int size = rows * cols;

        //in case of kernels' number > matrix' size
        if(step >= size) {
            if(id < size) {
                dest_data[id] = src_data[id];
            }
        } else {
            for(int c = id; c < size; c += step) {
                if(c < size) {
                    dest_data[c] = src_data[c];
                }
            }
        }
    }

    //FillMatrixValue_2dim-----
    template<class T>
    __global__ void _FillMatrixValue_2dim(T* dest_data, T defValue, int rows, int cols, int start_row, int start_col) {
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        int bid = blockIdx.x + blockIdx.y * gridDim.x;
        int nthreads = blockDim.x * blockDim.y; //number of threads per block
        int id = tid + nthreads * bid; //absolute id
        int step_x = blockDim.x * gridDim.x;;
        int step_y = blockDim.y * gridDim.y;
        int size = cols * rows;

        int c_row = blockIdx.y * blockDim.y + threadIdx.y; //current row
        int c_col = blockIdx.x * blockDim.x + threadIdx.x; //current col

        if(step_x * step_y >= size) {
            if(id < size && c_row >= start_row && c_col >= start_col) {
                dest_data[c_row * cols + c_col] = defValue;
            }
        } else {
            for(int i = c_row; i < rows; i += step_y) {
                for(int j = c_col; j < cols; j += step_x) {
                    if(i >= start_row && j >= start_col && i*j < size)
                        dest_data[i * cols + j] = defValue;
                }
            }
        }
    }

    //_AddMatrix
    остановился тут
    template<class T>
    __global__ void _AddMatrix(T* dest_data, T* sec_data, size_t size)
    {
        size_t thId = threadIdx.x;
        size_t id = blockDim.x * blockIdx.x + threadIdx.x;
        size_t step = gridDim.x * blockDim.x;

        if(step >= size) {
            if(id < size) {
                dest_data[id] += sec_data[id];
            }
        } else {
            for(int c = id; c < size; c += step) {
                if(c < size) {
                    dest_data[c] += sec_data[c];
                }
            }
        }
    }

    //_SubMatrix
    template<class T>
    __global__ void _SubMatrix(T* dest_data, T* sec_data, size_t size)
    {
        size_t thId = threadIdx.x;
        size_t id = blockDim.x * blockIdx.x + threadIdx.x;
        size_t step = gridDim.x * blockDim.x;

        if(step >= size) {
            if(id < size) {
                dest_data[id] -= sec_data[id];
            }
        } else {
            for(int c = id; c < size; c += step) {
                if(c < size) {
                    dest_data[c] -= sec_data[c];
                }
            }
        }
    }

    template<class T>
    __global__ void _SetElement(T* data, int stride, int i, int j, T value)
    {
        data[i * stride + j] = value;
    }

    template<class T>
    __device__ T _GetElement(_DeviceMatrix<T>& A, int i, int j)
    {
        return A.Data[i * A.Stride + j];
    }

    template<class T>
    __device__ MDMatrix_gpu<T> _GetSubMatrix(_DeviceMatrix<T> A, int i, int j)
    {
        _DeviceMatrix<T> ASub;
        ASub.Cols = BLOCKSIZE;
        ASub.Rows = BLOCKSIZE;
        ASub.Stride = A.Stride;
        ASub.Data = &A.Data[A.Stride * BLOCKSIZE * i + BLOCKSIZE * j];
        return ASub;
    }

    template<class T>
    __global__ void _BadMatMul(const _DeviceMatrix<T> A, const _DeviceMatrix<T> B, _DeviceMatrix<T> C)
    {
        int ROW = blockIdx.y * blockDim.y + threadIdx.y;
        int COL = blockIdx.x * blockDim.x + threadIdx.x;
        T tmpSum = 0;

        if (ROW < A.Rows && COL < B.Cols) {
            for (int i = 0; i < A.Cols; i++) {  //A.Cols==B.Rows
                tmpSum += A.Data[ROW * A.Cols + i] * B.Data[i * B.Cols + COL];
            }
            C.Data[ROW * C.Cols + COL] = tmpSum;
        }
        __syncthreads();
    }



    //-----------------------------MDMatrix_gpu block--------------------
    //constructors
    //MDMatrix_gpu(int rows, int cols, T defvalue)
    template<class T>
    MDMatrix_gpu<T>::MDMatrix_gpu(int rows, int cols, T defvalue)
        : Data(0), Rows(0), Cols(0), Capacity(0)
    {
        GetDeviceProps();
        Resize(rows, cols);
        unsigned int N = rows * cols;
        unsigned int threadsPerBlock = _DeviceProps[0].maxThreadsDim[0];
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        _FillMatrixValue << <blocksPerGrid, threadsPerBlock >> > (Data, defvalue, rows, cols);
    };

    //MDMatrix_gpu(const MDMatrix_gpu<T>& other)
    template<class T>
    MDMatrix_gpu<T>::MDMatrix_gpu(const MDMatrix_gpu<T>& other)
        : Data(0), Rows(0), Cols(0), Capacity(0)
    {
        GetDeviceProps();
        *this = other;
    };

    //MDMatrix_gpu(const MDMatrix_gpu<T>&& other)
    template<class T>
    MDMatrix_gpu<T>::MDMatrix_gpu(const MDMatrix_gpu<T>&& other)
        : Data(0), Rows(0), Cols(0), Capacity(0)
    {
        GetDeviceProps();
        *this = other;
    };

    template<class T>
    MDMatrix_gpu<T>::MDMatrix_gpu(const _DeviceMatrix<T>& other)
        : Data(other.Data), Rows(other.Rows), Cols(other.Cols), Capacity(other.Capacity)
    {
        GetDeviceProps();
    }

    //methods
    //Resize(int rows, int cols, T defvalue);
    template <class T>
    void MDMatrix_gpu<T>::Resize(int rows, int cols, T defvalue) {
        if (rows == Rows && cols == Cols)
            return;

        if (rows < 0 || cols < 0)
            return;

        if (rows == 0 || cols == 0)
        {
            Rows = rows;
            Cols = cols;
            return;
        }

        T* new_data(0);
        int new_capacity(0);

        if (rows && cols && rows * cols > Capacity)
        {
            if (cols > Cols || rows > Rows + RDK_MMATRIX_RESIZE_ROW_LINEAR_CONSTANT)
                new_capacity = int(rows * cols * RDK_MMATRIX_RESIZE_AMORTIZED_CONSTANT);
            else
                new_capacity = (rows + RDK_MMATRIX_RESIZE_ROW_LINEAR_CONSTANT) * cols;

            unsigned int N = rows * cols;
            unsigned int threadsPerBlock = _DeviceProps[0].maxThreadsDim[0];
            unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            dim3 grid = dim3(blocksPerGrid, 1, 1);
            dim3 block = dim3(threadsPerBlock, 1, 1);
            cudaMalloc((void**)&new_data, sizeof(T) * new_capacity);

            if (!Data) // �������� ������ � ������� �� ����
            {
                if (!defvalue)
                    cudaMemset(new_data, 0, rows * cols * sizeof(T));
                else
                {
                    _FillMatrixValue << <grid, block >> > (new_data, defvalue, rows, cols);
                }
            }
            else {
                if (new_data) // ������� �����������, �������� ����� ������, ��������� ������, ������� ��� ��������)
                {
                    int c_rows = (Rows < rows) ? Rows : rows;
                    int c_cols = (Cols < cols) ? Cols : cols;

                    //�������� �����������: ����� �� ��������� ��� ������� ���������� ��-���������
                    _FillMatrixValue << <grid, block >> > (new_data, defvalue, rows, cols);
                    _FillMatrixFrom << <grid, block >> > (new_data, Data, c_rows, c_cols);
                }
            }

        }
        else
            if (cols < Cols) // ������� ����������� �� ����� �������, ��������� ������ � ������ ������
            {
                T* old_pos = Data + 1 * Cols;
                T* new_pos = Data + cols;
                T* buff;
                cudaMalloc((void**)&buff, sizeof(T) * cols);

                for (int i = 1; i < rows; i++)
                {
                    cudaMemcpy(buff, old_pos, sizeof(T) * cols, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(new_pos, buff, sizeof(T) * cols, cudaMemcpyDeviceToDevice);
                    new_pos += cols;
                    old_pos += Cols;
                }

                // �������� �����������
                dim3 grid = dim3(255, 255, 1);
                dim3 block = dim3(32, 32, 1);
                _FillMatrixValue_2dim << <grid, block >> > (Data, defvalue, rows, cols, Rows, 0);
                _FillMatrixValue_2dim << <grid, block >> > (Data, defvalue, rows, cols, 0, Cols);

                cudaFree(buff);
            }

            else // ������� ����������� ��� ����������, ��������� ������ � ������ ������
            {
                int c_rows = (Rows < rows) ? Rows : rows;
                T* old_pos = Data + (c_rows - 1) * Cols;
                T* new_pos = Data + (c_rows - 1) * cols;
                T* buff;
                cudaMalloc((void**)&buff, sizeof(T) * Cols);

                for (int i = c_rows; i > 0; i--)
                {
                    cudaMemcpy(buff, old_pos, sizeof(T) * Cols, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(new_pos, buff, sizeof(T) * Cols, cudaMemcpyDeviceToDevice);
                    new_pos -= cols;
                    old_pos -= Cols;
                }
                dim3 grid = dim3(255, 255, 1);
                dim3 block = dim3(32, 32, 1);
                _FillMatrixValue_2dim << <grid, block >> > (Data, defvalue, rows, cols, Rows, 0);
                _FillMatrixValue_2dim << <grid, block >> > (Data, defvalue, rows, cols, 0, Cols);

                cudaFree(buff);
            }


        if (new_data)
        {
            cudaFree(Data);
            Data = new_data;
            Capacity = new_capacity;
        }
        Rows = rows;
        Cols = cols;
    }

    template<class T>
    void MDMatrix_gpu<T>::SetElement(int i, int j, T value)
    {
        if (i >= Rows, j >= Cols)
            return;
        _SetElement << <1, 1 >> > (Data, Cols, i, j, value);
    }

    template<class T>
    int MDMatrix_gpu<T>::GetCols() const
    {
        return Cols;
    }

    template<class T>
    int MDMatrix_gpu<T>::GetRows() const
    {
        return Rows;
    }

    template<class T>
    T* MDMatrix_gpu<T>::GetData()
    {
        return Data;
    }


    //operators block
    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator = (const MDMatrix_gpu<T>& copy)
    {
        Resize(copy.Rows, copy.Cols);
        cudaMemcpy(Data, copy.Data, Rows * Cols * sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator = (const MDMatrix_gpu<T>&& other)
    {
        Resize(other.Rows, other.Cols);
        Data = other.Data;
        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator = (T value)
    {
        unsigned int N = Rows * Cols;
        unsigned int threadsPerBlock = _DeviceProps[0].maxThreadsDim[0];
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        T* p = Data;

        _FillMatrixValue << <blocksPerGrid, threadsPerBlock >> > (Data, value, Rows, Cols);

        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator = (const T* data)
    {
        cudaMemcpy(Data, data, Rows * Cols * sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator = (const void* data)
    {
        cudaMemcpy(Data, data, Rows * Cols * sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator += (const MDMatrix_gpu<T>& M)
    {
        if (Rows != M.Rows || Cols != M.Cols) {
            return *this;
        }

        unsigned int N = Rows * Cols;
        unsigned int threadsPerBlock = _DeviceProps[0].maxThreadsDim[0];
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        _AddMatrix << <blocksPerGrid, threadsPerBlock >> > (Data, M.Data, Rows * Cols);

        return *this;
    }

    template<class T>
    MDMatrix_gpu<T>& MDMatrix_gpu<T>::operator -= (const MDMatrix_gpu<T>& M)
    {
        if (Rows != M.Rows || Cols != M.Cols) {
            return *this;
        }

        unsigned int N = Rows * Cols;
        unsigned int threadsPerBlock = _DeviceProps[0].maxThreadsDim[0];
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        _SubMatrix << <blocksPerGrid, threadsPerBlock >> > (Data, M.Data, Rows * Cols);

        return *this;
    }

    template<class T>
    MDMatrix_gpu<T> MDMatrix_gpu<T>::operator + (const MDMatrix_gpu<T>& M2)
    {
        MDMatrix_gpu<T> res(*this);
        res += M2;
        return res;
    }

    template<class T>
    MDMatrix_gpu<T> MDMatrix_gpu<T>::operator - (const MDMatrix_gpu<T>& M2)
    {
        MDMatrix_gpu<T> res(*this);
        res -= M2;
        return res;
    }

    template<class T>
    MDMatrix_gpu<T> MDMatrix_gpu<T>::operator * (const MDMatrix_gpu<T>& M2)
    {
        _DeviceMatrix<T> A(*this);
        _DeviceMatrix<T> B(M2);
        _DeviceMatrix<T> C(A.Rows, B.Cols);
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((A.Cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (B.Rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        _BadMatMul << <blocksPerGrid, threadsPerBlock >> > (A, B, C);

        MDMatrix_gpu<T> out(C);


        return out;
    }

    template<class T>
    T& MDMatrix_gpu<T>::operator () (int i, int j)
    {
        return Data[i * Cols + j];
    }

    // Общие методы доступа к параметрам матрицы
    // -----------------------------------
    /// Возвращает размерность матрицы
    template<class T>
    int MDMatrix_gpu<T>::GetDimensions() const {
        return 2;
    }

    template<class T>
    int MDMatrix_gpu<T>::GetSize(int i) const
    {
        return (i == 0) ? Rows : ((i == 1) ? Cols : 0);
    }

    template<class T>
    int MDMatrix_gpu<T>::GetSize() const
    {
        return Rows * Cols;
    }

    template<class T>
    bool MDMatrix_gpu<T>::IsEmpty() const
    {
        return (!Rows || !Cols);
    }

    template<class T>
    RDK::MMatrixSize MDMatrix_gpu<T>::GetMatrixSize() const
    {
        MMatrixSize size(Rows, Cols);
        return size;
    }

    template<class T>
    bool MDMatrix_gpu<T>::Resize(const RDK::MMatrixSize& size)
    {
        if (size.GetDimensions() != 2)
            return false;

        Resize(size[0], size[1]);
        return true;
    }

    template<class T>
    int MDMatrix_gpu<T>::GetByteSize() const
    {
        return Rows * Cols * sizeof(T);
    }

    template<class T>
    int MDMatrix_gpu<T>::GetElementByteSize() const
    {
        return sizeof(T);
    }

    template<class T>
    const void* MDMatrix_gpu<T>::GetVoid() const
    {
        return Void;
    }

    template<class T>
    void* MDMatrix_gpu<T>::GetVoid()
    {
        return Void;
    }


    //support block
    //GetDeviceProps()
    template<class T>
    void MDMatrix_gpu<T>::GetDeviceProps()
    {
        cudaGetDeviceCount(&_DeviceCount);
        _DeviceProps.resize(_DeviceCount);

        for (int a = 0; a < _DeviceCount; ++a) {
            cudaGetDeviceProperties(&_DeviceProps[a], a);
        }
    }

    //print()
    template<class T>
    void MDMatrix_gpu<T>::print() const
    {
        T* temp = new T[Rows * Cols];
        cudaMemcpy(temp, Data, Rows * Cols * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                std::cout << temp[i * Cols + j] << '\t';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
        delete[]temp;
    }

    //
    /*
    template<class T>
    void MDMatrix_gpu<T>::CopyFromCPU(const MMatrixBase* other_I)
    {
        RDK::MDMatrix* other = dynamic_cast<MDMatrix*>(other_I);
        //Resize(other->)
    }*/


}