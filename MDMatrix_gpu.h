#ifndef MDMATRIX_GPU_H
#define MDMATRIX_GPU_H

#include "..\..\Core\MMatrixBase.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

namespace RDK {



	template <class T>
	struct _DeviceMatrix;

	//MDMatrix_gpu class
	template <class T>
	class MDMatrix_gpu : public MMatrixBase
	{
		union
		{
			T* Data;
			double* Double;
			int* Int;
			unsigned char* UChar;
			char* Char;
			void* Void;
			void** PVoid;
		};

		int _DeviceCount;
		std::vector<cudaDeviceProp> _DeviceProps;

	protected:
		int Rows;
		int Cols;
		int Capacity;

	public:
		MDMatrix_gpu(int rows, int cols, T defvalue);
		MDMatrix_gpu(const MDMatrix_gpu<T>& other);
		MDMatrix_gpu(const MDMatrix_gpu<T>&& other);
		MDMatrix_gpu(const _DeviceMatrix<T>& other);

		void SetElement(int i, int j, T value);
		int GetRows() const;
		int GetCols() const;
		T* GetData();

		MDMatrix_gpu<T>& operator = (const MDMatrix_gpu<T>& copy);
		MDMatrix_gpu<T>& operator = (const MDMatrix_gpu<T>&& other);
		MDMatrix_gpu<T>& operator = (T value);
		MDMatrix_gpu<T>& operator = (const T* data);
		MDMatrix_gpu<T>& operator = (const void* data);

		MDMatrix_gpu<T>& operator += (const MDMatrix_gpu<T>& M);
		MDMatrix_gpu<T> operator + (const MDMatrix_gpu<T>& M2);
		MDMatrix_gpu<T>& operator -= (const MDMatrix_gpu<T>& M);
		MDMatrix_gpu<T> operator - (const MDMatrix_gpu<T>& M2);
		MDMatrix_gpu<T> operator * (const MDMatrix_gpu<T>& M2);

		T& operator () (int i, int j);

		void print() const;

		virtual int GetDimensions() const override;
		virtual int GetSize(int i) const override;
		virtual int GetSize(void) const override;
		virtual bool IsEmpty(void) const override;
		virtual MMatrixSize GetMatrixSize(void) const override;
		virtual bool Resize(const MMatrixSize& size) override;
		virtual int GetByteSize(void) const override;
		virtual int GetElementByteSize(void) const override;
		virtual const void* GetVoid(void) const override;
		virtual void* GetVoid(void) override;

		//void CopyFromCPU(const MMatrixBase* other_I);

	private:
		void GetDeviceProps();
		void Resize(int rows, int cols, T defvalue = 0);


		friend struct _DeviceMatrix<T>;

	};


	template <class T>
	struct _DeviceMatrix
	{
		T* Data;
		int Rows;
		int Cols;
		int Capacity;
		_DeviceMatrix(const MDMatrix_gpu<T>& other) {
			Rows = other.Rows;
			Cols = other.Cols;
			Capacity = other.Capacity;
			Data = other.Data;
		}
		_DeviceMatrix(int Rows, int Cols) {
			this->Cols = Cols;
			this->Rows = Rows;
			this->Capacity = int(Rows * Cols * RDK_MMATRIX_RESIZE_AMORTIZED_CONSTANT);
			cudaMalloc((void**)&Data, sizeof(T) * Capacity);
		}
	};

}

#endif