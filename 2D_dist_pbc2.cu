#include <iostream>     // std::cout, std::endl
#include <iomanip>      // std::setw
#include <cstdlib> /* malloc, free, rand */
#include <cstring>
//#include <stdio.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <fstream>      // std::ifstream
#include <sstream>
#include <new>
#include "myheader.h"
using namespace std;
#define Dim 3 // dimension of data

int main(void)
{
	int Nmol=100, NOd=18, NOH=70, nbin = 200;
	float Lx=1, Ly=1;
	float rcut2D = 0.3, zcut =3, dz = zcut/nbin;
	float** Od = new float*[NOd];
	float** OH = new float*[NOH];
	float** mol = new float*[Nmol];
	float** Dens_Od = new float*[Nmol];
	float** Dens_OH = new float*[Nmol];
	Od[0] = new float[NOd*Dim];
	for (int i=1; i<NOd; ++i) Od[i] = Od[i-1] + Dim;
	for (int i=0; i<NOd; ++i)
	{
		for(int j =0; j<Dim; ++j)
		{
			Od[i][j] = i*Dim+j;
		}
	}

        OH[0] = new float[NOH*Dim];
        for (int i=1; i<NOH; ++i) OH[i] = OH[i-1] + Dim;
        for (int i=0; i<NOH; ++i)
        {
                for(int j =0; j<Dim; ++j)
                {
                        OH[i][j] = i*Dim+j;
                }
        }

	mol[0] = new float[Nmol*Dim];
        for (int i=1; i<Nmol; ++i) mol[i] = mol[i-1] + Dim;
        for (int i=0; i<Nmol; ++i)
        {
                for(int j =0; j<Dim; ++j)
                {
                        mol[i][j] = i*Dim+j;
                }
        }

        Dens_Od[0] = new float[Nmol*nbin];
        for (int i=1; i<Nmol; ++i) Dens_Od[i] = Dens_Od[i-1] + nbin;
        for (int i=0; i<Nmol; ++i)
        {
                for(int j =0; j<nbin; ++j)
                {
                        Dens_Od[i][j] = i*nbin+j;
                }
        }

        Dens_OH[0] = new float[Nmol*nbin];
        for (int i=1; i<Nmol; ++i) Dens_OH[i] = Dens_OH[i-1] + nbin;
        for (int i=0; i<Nmol; ++i)
        {
                for(int j =0; j<nbin; ++j)
                {
                        Dens_OH[i][j] = i*nbin+j;
                }
        }

	int* Odval;
        Odval = (int *)malloc(Nmol * sizeof(int));
	
	float* r2DOd = (float *)malloc(Nmol*NOd*sizeof(float)); 
        float* r2DOH = (float *)malloc(Nmol*NOH*sizeof(float));
	float* zdiff_Od = (float *)malloc(Nmol*NOd*sizeof(float));
	float* zdiff_OH = (float *)malloc(Nmol*NOH*sizeof(float));

	float *d_Od, *d_OH, *d_mol, *d_rcut2D, *d_dz, *d_r2DOd, *d_r2DOH, *d_zdiff_Od, *d_zdiff_OH, *d_Dens_Od, *d_Dens_OH, *d_Lx, *d_Ly;
	int *d_nbin, *d_NOd, *d_NOH, *d_Odval;
	
	int size_Od = NOd*Dim*sizeof(float), size_OH = NOH*Dim*sizeof(float);
	int size_mol = Nmol*Dim*sizeof(float);
	int size_r2DOd = NOd*Nmol*sizeof(float), size_r2DOH = NOH*Nmol*sizeof(float);
	int size_zdiff_Od = Nmol*NOd*sizeof(float), size_zdiff_OH = Nmol*NOH*sizeof(float);
	int size_DensOd = Nmol*nbin*sizeof(float), size_DensOH = Nmol*nbin*sizeof(float);
	clock_t* time;
	clock_t time_used[Nmol * 2];

	cudaMalloc((void **)&d_Od, size_Od); cudaMalloc((void **)&d_OH, size_OH);
	cudaMalloc((void **)&d_mol, size_mol);
	cudaMalloc((void **)&d_rcut2D, sizeof(float)); cudaMalloc((void **)&d_dz, sizeof(float));
	cudaMalloc((void **)&d_nbin, sizeof(int));
	cudaMalloc((void **)&d_r2DOd, size_r2DOd); cudaMalloc((void **)&d_r2DOH, size_r2DOH);
	cudaMalloc((void **)&d_zdiff_Od, size_zdiff_Od); cudaMalloc((void **)&d_zdiff_OH, size_zdiff_OH);
	cudaMalloc((void **)&d_Dens_Od, size_DensOd);  
	cudaMalloc((void **)&d_Dens_OH, size_DensOH);
	cudaMalloc((void **)&d_Odval, sizeof(int)*Nmol);
	cudaMalloc((void **)&d_Lx, sizeof(float)); cudaMalloc((void **)&d_Ly, sizeof(float));
	cudaMalloc((void **)&d_NOd, sizeof(int)); cudaMalloc((void **)&d_NOH, sizeof(int));
	cudaMalloc((void**) &time, sizeof(clock_t)) * Nmol * 2;

	for (int i=0; i<NOd; i++)
	{
		for (int j=0; j<Dim; j++)
		{
			if(i==0) Od[i][j] = 0;
			else Od[i][j] = (float)i/2;
		}
	}

        for (int i=0; i<NOH; i++)
        {
                for (int j=0; j<Dim; j++)
                {
                        if(i==0) OH[i][j] = 0;
                        else OH[i][j] = (float)i/2;
                }
        }

	for (int i=0; i<Nmol; i++)
	{
		for (int j=0; j<Dim; j++)
		{
			mol[i][j] = i+1;
		}
	}

	cudaMemcpy(d_Od, Od[0], size_Od, cudaMemcpyHostToDevice);
	cudaMemcpy(d_OH, OH[0], size_OH, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mol, mol[0], size_mol, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rcut2D, &rcut2D, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dz, &dz, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbin, &nbin, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Lx, &Lx, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ly, &Ly, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nbin, &nbin, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_NOd, &NOd, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_NOH, &NOH, sizeof(int), cudaMemcpyHostToDevice);
	
	//Invoke CUDA functions
	dim3 grid(Nmol,1);
	dim3 block(Dim,(NOd+NOH)*9/Dim);
	size_t blocksize = (NOd+NOH)*Dim+Dim+(NOd+NOH)*9+9+9+(NOd+NOH)*2;
	Dist<<<grid,block,blocksize*sizeof(float)>>>(d_Od, d_OH, d_mol, d_NOd, d_NOH, d_Lx, d_Ly, d_rcut2D, d_dz, d_nbin, d_r2DOd, d_r2DOH, d_zdiff_Od, d_zdiff_OH, d_Dens_Od, d_Dens_OH, d_Odval);
	// make the host block until the device is finished with foo
	//cudaDeviceSynchronize();
	// check for error
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaMemcpy(r2DOd, d_r2DOd, size_r2DOd, cudaMemcpyDeviceToHost); 
	cudaMemcpy(zdiff_Od, d_zdiff_Od, size_zdiff_Od, cudaMemcpyDeviceToHost);
        cudaMemcpy(r2DOH, d_r2DOH, size_r2DOH, cudaMemcpyDeviceToHost);
        cudaMemcpy(zdiff_OH, d_zdiff_OH, size_zdiff_OH, cudaMemcpyDeviceToHost);
	cudaMemcpy(Dens_Od[0], d_Dens_Od, size_DensOd, cudaMemcpyDeviceToHost);
	cudaMemcpy(Dens_OH[0], d_Dens_OH, size_DensOH, cudaMemcpyDeviceToHost);
	cudaMemcpy(Odval, d_Odval, sizeof(int)*Nmol, cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used, time, sizeof(clock_t) * Nmol * 2, cudaMemcpyDeviceToHost);


        cout << "r2DOd: "<< endl;
        for (int i=0; i<29; i++)
        {
                cout << i+1 << ". ";
                for (int j=0; j<NOd; j++)
                {
                        cout << r2DOd[NOd*i+j] << " ";
                }
                cout << endl;
        }
        cout << endl;

	// GPU Timing 
	clock_t min_start, max_end;
	min_start = time_used[0];
	max_end = time_used[Nmol];
	for(int i = 1; i < Nmol; i++) 
	{
        	if(min_start > time_used[i]) min_start = time_used[i];
		if(max_end < time_used[i + Nmol]) max_end = time_used[i + Nmol];
	}

	cout << "time:	" << (float) (max_end - min_start)/CLOCKS_PER_SEC << " (sec)" << endl; 

	free(Od); free(OH); free(mol); free(r2DOd); free(r2DOH); free(zdiff_Od); free(zdiff_OH); free(Dens_Od); free(Dens_OH); free(Odval);
	cudaFree(d_Od); cudaFree(d_OH); cudaFree(d_mol); cudaFree(d_r2DOd); cudaFree(d_r2DOH);
	cudaFree(d_zdiff_Od); cudaFree(d_zdiff_OH); cudaFree(d_Dens_Od); cudaFree(d_Dens_OH); cudaFree(time); cudaFree(d_Odval);

	return 0;
}
