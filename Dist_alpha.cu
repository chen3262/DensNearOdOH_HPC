#include <stdio.h>
#include <iostream>
#include "myheader.h"
#define Dim 3

__global__ void Dist(float* Od, float* OH, float* mol, int* NOd, int* NOH, float* Lx, float* Ly, float* rcut2D, float* dz, int* nbin, float* r2DOd, float* r2DOH, float* zdiff_Od, float* zdiff_OH, float* Dens_Od, float* Dens_OH, int* Odval) 
{
	const int t_NOd=*NOd, t_NOH=*NOH;
        float t_Lx=*Lx, t_Ly=*Ly;
	extern __shared__ float buffer[];
	float* t_Od = &buffer[0]; //size: t_NOd*Dim
	float* t_OH = &buffer[t_NOd*Dim]; //size: t_NOH*Dim
	float* t_mol = &buffer[(t_NOd+t_NOH)*Dim];	//size: Dim
	float* t_r2DOd = &buffer[ (t_NOd+t_NOH)*Dim+Dim ]; //size: t_NOd*9
	float* t_r2DOH = &buffer[ (t_NOd+t_NOH)*Dim+t_NOd*9 ]; //size: t_NOH*9
	float* pbcIdx1 = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9 ];//size:9
	float* pbcIdx2 = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9+9 ];//size: 9
	float* t_zdiff_Od = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9+9+9 ];//size: t_NOd
	float* t_zdiff_OH = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9+9+9+t_NOd ];//size: t_NOH
	float* count_Od = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9+9+9+t_NOd+t_NOH ];//size: t_NOd
	float* count_OH = &buffer[ (t_NOd+t_NOH)*Dim+Dim+(t_NOd+t_NOH)*9+9+9+t_NOd+t_NOH+t_NOd ];//size: t_NOd
	int lindex = threadIdx.y*Dim + threadIdx.x;
	int lindex2 = lindex - t_NOd*9;
	float zcut = (*dz)*(*nbin);
	//Permutations of periodic cells in xy plane
	pbcIdx1[0]=0; pbcIdx1[1]=1; pbcIdx1[2]=-1; pbcIdx1[3]=0; pbcIdx1[4]=0; pbcIdx1[5]=1; pbcIdx1[6]=1; pbcIdx1[7]=-1; pbcIdx1[8]=-1;
	pbcIdx2[0]=0; pbcIdx2[1]=0; pbcIdx2[2]=0; pbcIdx2[3]=1;	pbcIdx2[4]=-1; pbcIdx2[5]=1; pbcIdx2[6]=-1; pbcIdx2[7]=1; pbcIdx2[8]=-1;

	//Copy the full Od array (t_NOd*Dim) into the shared memory of each block
	if (lindex < t_NOd*Dim) 
	{
		t_Od[lindex] = Od[lindex];
	}
	if (lindex < t_NOH*Dim)
	{
		t_OH[lindex] = OH[lindex];
	}

	// Copy 1 mol coordinates to the t_mol list in each block;
	if (lindex < Dim)
	{
		t_mol[lindex] = mol[blockIdx.x*Dim+lindex];
	}
	__syncthreads();

	if (lindex < *nbin)
	{
		Dens_Od[blockIdx.x* *nbin + lindex] = 0;
		Dens_OH[blockIdx.x* *nbin + lindex] = 0;
	}
	__syncthreads();

	// Calculate mol - Od distance projected on the xy-plane
	if (lindex < t_NOd*9)// asking for t_NOd*9 threads for each mol atom
	{
		t_r2DOd[lindex] = 0;
		t_r2DOd[lindex] += ( t_mol[0] - (t_Od[ (lindex/9)*Dim +0] + pbcIdx1[ (int)fmodf( (float)lindex,9 ) ]*t_Lx) )
					*( t_mol[0] - (t_Od[ (lindex/9)*Dim +0] + pbcIdx1[ (int)fmodf( (float)lindex,9 ) ]*t_Lx) );
		t_r2DOd[lindex] += ( t_mol[1] - (t_Od[ (lindex/9)*Dim +1] + pbcIdx2[ (int)fmodf( (float)lindex,9 ) ]*t_Ly) )
					*( t_mol[1] - (t_Od[ (lindex/9)*Dim +1] + pbcIdx2[ (int)fmodf( (float)lindex,9 ) ]*t_Ly) );

		//Find mol-Od for each Od: min of ever 9 t_r2DOd[lindex];
		for (unsigned int stride =1; stride < 9; stride *=2)
		{
			__syncthreads();
			if ( (lindex - (lindex/9)*9 )%(2*stride) == 0 )
			{
				if( (lindex+stride < (lindex/9)*9 + 9) && (lindex >= (lindex/9)*9 ) )
				{
					if( t_r2DOd[lindex] > t_r2DOd[lindex + stride] )
					{
						t_r2DOd[lindex] = t_r2DOd[lindex + stride]; // store rmin in 0, 9, ...., 9*t_NOd elements
					}
				}
			}
		}


		//Calculated number and zdiff for qualified mol-Od pairs; fill zero for excluded mol-Od pairs
		if( fmodf( (float)lindex, 9 ) == 0)// lindex = 0, 9, 18 ,....
		{
			t_r2DOd[lindex] = sqrtf( t_r2DOd[lindex]);

			//For checking, print out r2DOd for the minimum r2DOd for each Od
			r2DOd[blockIdx.x*t_NOd + lindex/9] = t_r2DOd[lindex];

			if ( t_r2DOd[lindex] <= *rcut2D )	
			{
				t_zdiff_Od[lindex/9] = fabs( t_mol[2] - t_Od[lindex/9*Dim+2] );
				count_Od[lindex/9] = 1;
			}	
			else
			{
				t_zdiff_Od[lindex/9] = 0;
				count_Od[lindex/9] = 0;
			}
			zdiff_Od[ blockIdx.x*t_NOd + lindex/9 ] = t_zdiff_Od[lindex/9];
		}

                if (lindex == 0)
                {
			Odval[blockIdx.x] = 0;
                        for (int i = 0; i < t_NOd; i++)
                        {
				if ( t_zdiff_Od[i] < zcut)
				{
                                	Dens_Od[blockIdx.x * *nbin + (int)(t_zdiff_Od[i] / *dz)] += count_Od[i];
				}
				Odval[blockIdx.x] += count_Od[i];
                        }
                }

	}


        // Calculate mol - OH distance projected on the xy-plane
        if ( (lindex2 >= 0) && (lindex2 < t_NOH*9))// asking for t_NOH*9 threads for each mol atom
        {
                t_r2DOH[lindex2] = 0;
                t_r2DOH[lindex2] += ( t_mol[0] - (t_OH[ (lindex2/9)*Dim +0] + pbcIdx1[ (int)fmodf( (float)lindex2,9 ) ]*t_Lx) )
                                        *( t_mol[0] - (t_OH[ (lindex2/9)*Dim +0] + pbcIdx1[ (int)fmodf( (float)lindex2,9 ) ]*t_Lx) );
                t_r2DOH[lindex2] += ( t_mol[1] - (t_OH[ (lindex2/9)*Dim +1] + pbcIdx2[ (int)fmodf( (float)lindex2,9 ) ]*t_Ly) )
                                        *( t_mol[1] - (t_OH[ (lindex2/9)*Dim +1] + pbcIdx2[ (int)fmodf( (float)lindex2,9 ) ]*t_Ly) );


                //Find mol-OH for each OH: min of ever 9 t_r2DOH[lindex2];
                if (lindex2 < t_NOH*9)
                {
			__syncthreads();
                        for (unsigned int stride =1; stride < 9; stride *=2)
                        {
                                if ( (lindex2 - (lindex2/9)*9 )%(2*stride) == 0 )
                                {
                                        if( (lindex2+stride < (lindex2/9)*9 + 9) && (lindex2 >= (lindex2/9)*9 ) )
                                        {
                                                if( t_r2DOH[lindex2] > t_r2DOH[lindex2 + stride] )
                                                {
                                                        t_r2DOH[lindex2] = t_r2DOH[lindex2 + stride]; // store rmin in 0, 9, ...., 9*t_NOH elements
                                                }
                                        }
                                }
                        }
                }

                //Calculated number and zdiff for qualified mol-OH pairs; fill zero for excluded mol-OH pairs
                if( fmodf( (float)lindex2, 9 ) == 0)
                {
                        t_r2DOH[lindex2] = sqrtf( t_r2DOH[lindex2]);

                        //For checking, print out the minimum r2DOH for each OH
                        r2DOH[blockIdx.x*t_NOH + lindex2/9] = t_r2DOH[lindex2];

                        if (t_r2DOH[lindex2] <= *rcut2D)
                        {
                                t_zdiff_OH[lindex2/9] = fabs( t_mol[2] - t_OH[lindex2/9*Dim+2] );
                                count_OH[lindex2/9] = 1;
				if (t_zdiff_OH[lindex2/9] >= zcut)
				{
					t_zdiff_OH[lindex2/9] = 0;
					count_OH[lindex2/9] = 0;
				}
                        }
                        else
                        {
                                t_zdiff_OH[lindex2/9] = 0;
                                count_OH[lindex2/9] = 0;
                        }
                        zdiff_OH[ blockIdx.x*t_NOH + lindex2/9 ] = t_zdiff_OH[lindex2/9];
                }

		if (lindex2 == 0)
		{
			for (int i = 0; i < t_NOH; i++)
			{
				Dens_OH[blockIdx.x * *nbin + (int)(t_zdiff_OH[i] / *dz)] += count_OH[i];
			}
		}

        }
}

