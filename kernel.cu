#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>

#include "..\..\Core\MDMatrix.h"
#include "..\..\Core\TIM\Timer.h"
#include "MDMatrix_gpu.cu"

using namespace RDK;


int main() {
	setlocale(LC_ALL, "Russian");
	
	std::cout << "CPU operator+\n";
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration(0);

		
		RDK::MDMatrix<int> a(1000, 1000, 5);
		RDK::MDMatrix<int> b(1000, 1000, 6);

		start = std::chrono::high_resolution_clock::now();
		RDK::MDMatrix<int> c = a+b;
		end = std::chrono::high_resolution_clock::now();

		duration = end - start;
		std::cout << duration.count() << "s\n" << std::endl;
	}

	std::cout << "CPU operator+ in loop\n";
	{
		std::chrono::duration<double> duration(0);
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		for (int i = 0; i < 100; ++i) {

			RDK::MDMatrix<int> a(1000, 1000, 5);
			RDK::MDMatrix<int> b(1000, 1000, 6);

			start = std::chrono::high_resolution_clock::now();
			RDK::MDMatrix<int> c = a + b;
			end = std::chrono::high_resolution_clock::now();

			duration += (end - start);
		}

		duration /= 100;
		std::cout << duration.count() << "s\n" << std::endl;
	}
	

	std::cout << "GPU operator+\n";
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration(0);

		MDMatrix_gpu<int> a(1000, 1000, 5);
		MDMatrix_gpu<int> b(1000, 1000, 6);

		start = std::chrono::high_resolution_clock::now();
		MDMatrix_gpu<int> c = a + b;
		cudaDeviceSynchronize();
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		std::cout << duration.count() << "s\n" << std::endl;
	}

	std::cout << "GPU operator + in loop\n";
	{
		std::chrono::duration<double> duration(0);
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		for (int i = 0; i < 100; ++i) {

			MDMatrix_gpu<int> a(1000, 1000, 5);
			MDMatrix_gpu<int> b(1000, 1000, 6);

			start = std::chrono::high_resolution_clock::now();

			MDMatrix_gpu<int> c = a + b;

			cudaDeviceSynchronize();

			end = std::chrono::high_resolution_clock::now();

			duration += end - start;
		}

		duration /= 100;
		std::cout << duration.count() << "s\n" << std::endl;
	}
	

	std::cout << "CPU operator*\n";
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration(0);


		RDK::MDMatrix<int> a(1000, 1000, 5);
		RDK::MDMatrix<int> b(1000, 1000, 6);

		start = std::chrono::high_resolution_clock::now();
		RDK::MDMatrix<int> c = a * b;
		end = std::chrono::high_resolution_clock::now();

		duration = end - start;
		std::cout << duration.count() << "s\n" << std::endl;
	}

	/*std::cout << "CPU operator* in loop\n";
	{
		std::chrono::duration<double> duration(0);
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		for (int i = 0; i < 10; ++i) {

			RDK::MDMatrix<int> a(1000, 1000, 5);
			RDK::MDMatrix<int> b(1000, 1000, 6);

			start = std::chrono::high_resolution_clock::now();
			RDK::MDMatrix<int> c = a * b;
			end = std::chrono::high_resolution_clock::now();

			duration += (end - start);
		}

		duration /= 10;
		std::cout << duration.count() << "s\n" << std::endl;
	}*/
	

	std::cout << "GPU operator*\n";
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration(0);

		MDMatrix_gpu<int> a(1000, 1000, 5);
		MDMatrix_gpu<int> b(1000, 1000, 6);

		start = std::chrono::high_resolution_clock::now();
		MDMatrix_gpu<int> c = a * b;
		cudaDeviceSynchronize();
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		std::cout << duration.count() << "s\n" << std::endl;
	}

	std::cout << "GPU operator * in loop\n";
	{
		std::chrono::duration<double> duration(0);
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		for (int i = 0; i < 100; ++i) {

			MDMatrix_gpu<int> a(1000, 1000, 5);
			MDMatrix_gpu<int> b(1000, 1000, 6);

			start = std::chrono::high_resolution_clock::now();

			MDMatrix_gpu<int> c = a * b;

			cudaDeviceSynchronize();

			end = std::chrono::high_resolution_clock::now();

			duration += end - start;
		}

		duration /= 100;
		std::cout << duration.count() << "s\n" << std::endl;
	}


}




