#include <iostream>
#include <cstdlib>
#include <immintrin.h>
#include <windows.h>


#define l 2048
#define m 2048
#define n 2048
#define d 128


using namespace std;


void fillMatrix(float**& Matrix, int line1, int column1) {
	for (int i = 0; i < line1; i++)
		for (int j = 0; j < column1; j++)
			Matrix[i][j] = rand() % 20 + 1;
}


float** initialize(int line1, int column1) {
	float** arr = new float* [line1];
	for (int x = 0; x < line1; x++)
		arr[x] = new float[column1];
	return arr;
}


void setZero(float** C) {
	for (int i = 0; i < n; ++i)
	{
		float* c = C[i];
		for (int j = 0; j < n; j++)
			c[j] = 0;
	}			
}


void blockMul(float** A, float** B, float** C) {
	for (int di = 0; di < n; di += d)
		for (int dk = 0; dk < n; dk += d)
			for (int dj = 0; dj < n; dj += d)
				for (int i = di; i < min(di + d, n); ++i)
				{
					for (int k = dk; k < min(dk + d, n); k++)
					{
						__m256 a = _mm256_set1_ps(A[i][k]);
						for (int j = dj; j < min(dj + d, n); j += 8)
						{
							/*_mm256_storeu_ps(C[i] + j, _mm256_fmadd_ps(a,
								_mm256_loadu_ps(B[k] + j), _mm256_loadu_ps(C[i] + j)));*/
								/**(__m256*)(C[i] + j) = _mm256_fmadd_ps(a,
									*(__m256*)(B[k] + j), *(__m256*)(C[i] + j));*/
							_mm256_storeu_ps(C[i] + j, _mm256_add_ps(_mm256_mul_ps(a,
								_mm256_loadu_ps(B[k] + j)), _mm256_loadu_ps(C[i] + j)));
						}
					}
				}
}


void vecMulIns(int L, int N, int M, float** A, float** B, float** C)
{
	for (int i = 0; i < L; ++i)
	{
		for (int j = 0; j < N; j += 8)
			_mm256_storeu_ps(C[i] + j, _mm256_setzero_ps());
		for (int k = 0; k < M; ++k)
		{
			__m256 a = _mm256_set1_ps(A[i][k]);
			for (int j = 0; j < N; j += 8)
			{
				/*_mm256_storeu_ps(C[i] + j, _mm256_fmadd_ps(a,
					_mm256_loadu_ps(B[k] + j), _mm256_loadu_ps(C[i] + j)));*/
					/**(__m256*)(C[i] + j) = _mm256_fmadd_ps(a,
						*(__m256*)(B[k] + j), *(__m256*)(C[i] + j));*/
				_mm256_storeu_ps(C[i] + j, _mm256_add_ps(_mm256_mul_ps(a,
					_mm256_loadu_ps(B[k] + j)), _mm256_loadu_ps(C[i] + j)));
			}
		}
	}
}


void main() {


	float** A = initialize(l, m);
	float** B = initialize(m, n);
	float** C_Ins = initialize(l, n);
	float** C_cashe = initialize(l, n);


	fillMatrix(A, l, m);
	fillMatrix(B, m, n);
	setZero(C_cashe);


	auto start_clocks = GetTickCount64();
	vecMulIns(l, n, m, A, B, C_Ins);
	auto end_clocks = GetTickCount64();
	cout << end_clocks - start_clocks << " msecs ManualVect" << endl;

	start_clocks = GetTickCount64();
	blockMul(A, B, C_cashe);
	end_clocks = GetTickCount64();
	cout << end_clocks - start_clocks << " msecs blockMul" << endl;


		for (int i = 0; i < l; i++)
		for (int j = 0; j < n; j++)
			if (C_cashe[i][j] != C_Ins[i][j]) { 
				cout << "Matrix(C_cashe, C_Ins) not equal" << endl; 
				i = l;
				break;
			}
			else if ((i == l - 1) && (j == n - 1) && (C_cashe[i][j] == C_Ins[i][j])) cout << "Matrix(C_cashe, C_Ins) are equal" << endl;
}
