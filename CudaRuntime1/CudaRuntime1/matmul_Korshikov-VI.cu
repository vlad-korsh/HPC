#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

//задаём размер блока
#define BLOCK_SIZE 16

//пропишем функцию ядра
__global__ void matrixMult(const double* A, const double* B, double* C, int n)
{
	int ai = n * (blockDim.y * blockIdx.y + threadIdx.y);	// индекс начала строки матрицы A
	int bj = blockDim.x * blockIdx.x + threadIdx.x;			// индекс начала строки матрицы B
	double sum = 0;											// промежуточная переменная для вычиселний
	for (int k = 0; k < n; k++)
		sum += A[ai + k] * B[k * n + bj];					// вычисление произведения
	int index = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x; // индекс вычисляемого элемента матрицы C 
	C[index] = sum;											// заполнение массива результатми
}

// генерируем матрицы
double* generateRandMatrix(int n, size_t sizeMatrix) {
	double* matrix = (double*)malloc(sizeMatrix);			// выделение памяти под массив
	for (int i = 0; i < n * n; i++) {
		matrix[i] = (double)rand() / (double)RAND_MAX;		// заполнение массива случайными числами
	}
	return matrix;											// возврат заполненной матрицы
}


// функция для последовательного варианта умножения матриц
void matrixMultCPU(double* A, double* B, double* C, int n) {
	// реализация математического алгоритма умножения матриц
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

// проверка результатов умножения
bool checkMult(double* C1, double* C2, int n) {
	double accuracy = 1.e-6;						//точность проверки
	for (int i = 0; i < n * n; i++) {				// перебираем все ячейки в цикле 
		if (abs(C1[i] - C2[i]) >= accuracy)			// смотрим, есть ли разница в результате вычислений на CPU и GPU
													// если есть, то результат некорректен
			return false;
	}
	return true;									// Если нет, то результат корректен
}

int main(int argc, char* argv[])
{
	int N=1024;										//ПРОПИШЕМ РАЗМЕРМЕРНОСТЬ МАТРИЦЫ
	
	// Начало и окончание события
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	srand(time(NULL));
	size_t sizeMatrix = sizeof(double) * N * N;			// создадим запас по размерности массива

	// сгенерируем массивы для работы на CPU
	double* A_CPU = generateRandMatrix(N, sizeMatrix);
	double* B_CPU = generateRandMatrix(N, sizeMatrix);
	double* C_CPU = (double*)malloc(sizeMatrix);			// матрица, содержащая результаты, из GPU
	double* C_seq_CPU = (double*)malloc(sizeMatrix);
	for (int i = 0; i < N * N; i++) {
		C_seq_CPU[i] = 0;
	}

	// high_resolution_clock - определяем время работы
	high_resolution_clock::time_point t1 = high_resolution_clock::now();		// точка начала отсчёта 
	matrixMultCPU(A_CPU, B_CPU, C_seq_CPU, N);									// расчитываем матричное произведения
	high_resolution_clock::time_point t2 = high_resolution_clock::now();		// точка окончания отсчёта 
	duration<double, std::milli> time_span = t2 - t1;							// расчитываем затраченное время						
	double cpu_time = time_span.count();
	printf("The time: %f milliseconds\n", cpu_time);

	// выделение памяти на GPU
	double* A_GPU;
	cudaMalloc((void**)&A_GPU, sizeMatrix);
	double* B_GPU;
	cudaMalloc((void**)&B_GPU, sizeMatrix);
	double* C_GPU;
	cudaMalloc((void**)&C_GPU, sizeMatrix);

	// копируем данные на GPU
	cudaMemcpy(A_GPU, A_CPU, sizeMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, sizeMatrix, cudaMemcpyHostToDevice);

	// определим размерность сетки для работы функции ядра
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(N / BLOCK_SIZE, N / BLOCK_SIZE);

	cudaEventRecord(start, 0);														// точка начала отсчёта времени
	matrixMult << <blocksPerGrid, threadsPerBlock >> > (A_GPU, B_GPU, C_GPU, N);	// работа функции ядра
	cudaEventRecord(stop, 0);														// точка окончания отсчёта времени
	cudaEventSynchronize(stop);														// проведём синхронизацию

	// вычислим время работы функции на GPU
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f milliseconds\n", KernelTime);

	// вычислим ускорение
	double S = cpu_time / KernelTime;
	printf("Acceleration: %f\n", S);

	// копируем результаты с GPU для проверки
	cudaMemcpy(C_CPU, C_GPU, sizeMatrix, cudaMemcpyDeviceToHost);


	// проверяем корректность вычислений
	if (checkMult(C_CPU, C_seq_CPU, N))
		printf("The multiplication results are correct.\n");
	else
		printf("Multiplication results are NOT correct.\n");

	// очистка памяти
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);
	free(C_seq_CPU);

	return 0;
}