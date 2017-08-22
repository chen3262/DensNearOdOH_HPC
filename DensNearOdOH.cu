#include <iostream>     // std::cout, std::endl
#include <iomanip>      // std::setw
#include <cstdlib> /* malloc, free, rand */
#include <cstring>
#include <time.h>
#include <cmath>
#include <vector>
#include <fstream>      // std::ifstream
#include <sstream>
#include <new>
#include "/mnt/singer_group3/Sihan/gromacs/gromacs-5.0.7/_install/include/gromacs/fileio/trnio_mod.h"
#include "myheader.h"
using namespace std;
#define Dim 3 // dimension of data

extern t_fileio *open_trn(const char *fn, const char *mode);
/* Open a trj / trr file */

extern void close_trn(t_fileio *fio);
/* Close it */

extern void read_trn(const char *fn, int *step, real *t, real *lambda, matrix *box, int *natoms, rvec *x, rvec *v, rvec *f);
/* Read a single trn frame from file fn, which is closed afterwards */

extern gmx_bool fread_trn(t_fileio *fio, int *step, real *t, real *lambda, matrix *box, int *natoms, rvec *x, rvec *v, rvec *f);
/* Read a trn frame, including the header from fp. box, x, v, f may  be NULL, in which case the data will be skipped over. return FALSE on
 *  error  */

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

inline
cudaError_t checkCuda(cudaError_t result)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	#endif
	return result;	
}

int main(int argc, char* argv[])
{
	//File I/O
        ifstream inputSOL, molinfo;
        ofstream output1, log;
        inputSOL.open(argv[2]);
        molinfo.open(argv[3]);
        output1.open(argv[4]);
        log.open(argv[5]);

	// Declare system parameters, some will need informatoin from the .mol file
	int nmol, Nmol, NOd, NOH, nbin = 300;
	float rcut2D = 0.33, zcut=3, dz = zcut/nbin;
	float Lx, Ly;
	int skip=0; // # of ps to be skipped (1ps might contain 10 frames, check mdp settings)
        float dV;
	time_t nStart = time(NULL);//Start timer
	//float ms; // elapsed time in milliseconds

        /*Read molecules information: atom names, mass, charges*/
        string line,word;
        stringstream linestream;
        int a;
        float d;
        getline(molinfo,line); //1st line "[ molecule ]"
        getline(molinfo,line); // 2nd line "Name"
        getline(molinfo,line);//3rd line
        linestream.clear(); linestream.str(line); linestream >> word;
        string Sysname = word;
        getline(molinfo,line); //"[Number of molecules]"
        getline(molinfo,line); // 5th line
        linestream.clear(); linestream.str(line); linestream >> a;
        nmol = a;
        getline(molinfo,line); // "[ Time steps (ps) ]"
        getline(molinfo,line); linestream.clear(); linestream.str(line); linestream >> d;
        float dt = d; //Sampling time steps in ps
        getline(molinfo,line); //[Red Atom]
        getline(molinfo,line); //Name
        getline(molinfo,line); //Od
        getline(molinfo,line); //[ Number of Ref atoms ]
        getline(molinfo,line); linestream.clear(); linestream.str(line); linestream >> a;
        NOd = a;
        getline(molinfo,line); //[Red Atom]
        getline(molinfo,line); //Name
        getline(molinfo,line); //OH
        getline(molinfo,line); //[ Number of Ref atoms ]
        getline(molinfo,line); linestream.clear(); linestream.str(line); linestream >> a;
        NOH = a;
	
	int frameblock = 2500*50/nmol, frameblock2 = frameblock;
	int nStreams = ceil((float)2500/nmol)*nmol;
	Nmol = nmol*frameblock;
        //Output system information to log
        log << "System information: " << endl;
        log << "Atom:   " << Sysname << endl;
        log << "nmol:   " << nmol << endl;
        log << "Sampling Time step (ps): "<< dt << endl;
        log << "nbin:   " << nbin << endl;
        log << "zcut:   " << zcut << " nm" << endl;
        log << "rcut2D  " << rcut2D << " nm" << endl;
        log << "Skipped steps:  " << skip << endl;
        log << "NOd:    " << NOd << endl;
        log << "NOH:    " << NOH << "\n" << endl;
	log << "CUDA information: " << endl;
	log << "frameblock:	" << frameblock << endl;
	log << "nStreams:	" << nStreams << "\n"<< endl;

        /*Reading the inputSOL to get index of target molecules*/
        Atom atom;
        vector<int> molidx, Odidx, OHidx;
        vector<Atom> molGRO, OdGRO, OHGRO;
        int count=0;
        getline(inputSOL,line);//skip the 1st and 2nd line
        getline(inputSOL,line);
        while(!inputSOL.eof())
        {
                getline(inputSOL,line);
                if (line.empty())
                {
                        break;
                }
                else
                {
                        count++;
                        atom.name = line.substr(10,5);
                        if (line.length() >= 44) atom.z = atof(line.substr(36,8).c_str());
                        linestream.clear(); linestream.str(atom.name);
                        linestream >> word;
                        atom.name = word;
                        if(atom.name.find(Sysname) != string::npos)
                        {
                                molGRO.push_back(atom);
                                molidx.push_back(count-1);//molidx start from 0 in C
                        }
                        else if (atom.name.find("Od") != string::npos)
                        {
                                OdGRO.push_back(atom);
                                Odidx.push_back(count-1);
                        }
                        else if ( (atom.name.find("OSil") != string::npos) || (atom.name.find("OGem") != string::npos)  )
                        {
                                if ( (atom.z > 2) && (atom.z < 13))
                                {
                                        OHGRO.push_back(atom);
                                        OHidx.push_back(count-1);
                                }
                        }
                }
        }

        if ( molGRO.size() != nmol)
        {
                cout << "The number of " << Sysname << " in the gro file doesn't match the number in the .mol file! Job Abortted!" << endl;
                exit (EXIT_FAILURE);
        }
        if ( OdGRO.size() != NOd)
        {
                cout << "The number of Od in the gro file doesn't match the number in the .mol file! Job Abortted!" << endl;
                exit (EXIT_FAILURE);
        }
        if ( OHGRO.size() != NOH)
        {
                cout << "The number of OH in the gro file doesn't match the number in the .mol file! Job Abortted!" << endl;
                exit (EXIT_FAILURE);
        }

	//Declare arrays to store coordinates of mol, Od, and OH
	float* Od = new float[NOd*Dim]();
        float* OH = new float[NOH*Dim]();
	float* mol;
	checkCuda( cudaMallocHost((void**)&mol,Nmol*Dim*sizeof(float)) );
	float* Dens_Od;
	checkCuda( cudaMallocHost((void**)&Dens_Od,Nmol*nbin*sizeof(float)) );
	float* Dens_OH;
	checkCuda( cudaMallocHost((void**)&Dens_OH,Nmol*nbin*sizeof(float)) );

        //Declare arrays to store number of molecules in each bin over time frames
        float *nMol_bin_Od = new float[nbin]();
        float *nMol_bin_OH = new float[nbin]();
        for (int i=0; i<nbin; i++)
        {
                nMol_bin_Od[i]=0; nMol_bin_Od[i]=0;
        }

	// declare pointers on device
	float *d_Od, *d_OH, *d_mol, *d_Lx, *d_Ly, *d_rcut2D, *d_dz, *d_Dens_Od, *d_Dens_OH;
	int *d_nbin, *d_NOd, *d_NOH, *d_Odval;

	int size_Od = NOd*Dim*sizeof(float), size_OH = NOH*Dim*sizeof(float);
	int size_mol = Nmol*Dim*sizeof(float);
	int size_DensOd = Nmol*nbin*sizeof(float), size_DensOH = Nmol*nbin*sizeof(float);

	//Allocate memories on host
	int* Odval;
	checkCuda( cudaMallocHost((void**)&Odval,Nmol*sizeof(int)) );

	// Allocate memories on device
	cudaMalloc((void **)&d_Od, size_Od); cudaMalloc((void **)&d_OH, size_OH); cudaMalloc((void **)&d_mol, size_mol);
	cudaMalloc((void **)&d_rcut2D, sizeof(float)); cudaMalloc((void **)&d_dz, sizeof(float));
	cudaMalloc((void **)&d_Lx, sizeof(float)); cudaMalloc((void **)&d_Ly, sizeof(float));
	cudaMalloc((void **)&d_nbin, sizeof(int));
	cudaMalloc((void **)&d_NOd, sizeof(int)); cudaMalloc((void **)&d_NOH, sizeof(int));
	cudaMalloc((void **)&d_Dens_Od, size_DensOd);  cudaMalloc((void **)&d_Dens_OH, size_DensOH);
	cudaMalloc((void **)&d_Odval, sizeof(int)*Nmol);

        //Open TRR file
        t_fileio *TRR;
        TRR = open_trn(argv[1],"r");
        int natoms=count-1; /*The initial Total atom number is obtained from the inputSOL.gro file. Very Important! */
        int step=9, step2=10, nframes=0;
        real t, lambda;
        matrix box;
        rvec *x = (rvec *)malloc(natoms *sizeof(rvec));
        rvec *v = (rvec *)malloc(natoms *sizeof(rvec));
        rvec *f = (rvec *)malloc(natoms *sizeof(rvec));
        int dtmd=100; //how many fs for each md step
        int stride=float(dt)/(float(dtmd)/1000.); //sample from every stride step           

/*Start reading from the 1st frame*/
	while( frameblock == frameblock2 )
	{
		for ( int fr=0; fr< frameblock; ++fr)
		{
			fread_trn(TRR, &step, &t, &lambda, &box, &natoms, x, v, f);
			if (step == step2)
			{
				cout << "\n" << endl;
				frameblock2 = fr;
				Nmol = nmol*frameblock2;
				break;
			}
			else if ( step/dtmd % int(stride) == 0 )
			{
				if (step == 0)
				{
					Lx = box[0][0]; Ly = box[1][1];
					for (int i=0; i<NOd; i++)
					{
						for(int j=0; j<Dim; j++)
						{
							Od[i*Dim+j] = x[Odidx[i]][j];
						}
					}
					for (int i=0; i<NOH; i++)
					{
						for(int j=0; j<Dim; j++)
						{
							OH[i*Dim+j] = x[OHidx[i]][j];
						}
					}
				}

				step2=step;
				nframes++;
				if (int(t*10) %100 == 0)
				{
					cout << "\rReading step: " << step << " (" << t << " ps)" << flush;
					log << "Reading step: " << step << " (" << t << " ps)" << endl;
				}

				for (int i=0; i<nmol; i++)
				{
					for(int j=0; j<Dim; j++)
					{
						mol[ fr*nmol*Dim +i*Dim + j ] = x[molidx[i]][j];
					}
				}
			}
		}
		

		// Create events and streams
		cudaEvent_t startEvent, stopEvent, dummyEvent;
		cudaStream_t stream[nStreams];
		checkCuda( cudaEventCreate(&startEvent) );
		checkCuda( cudaEventCreate(&stopEvent) );
		checkCuda( cudaEventCreate(&dummyEvent) );
		for (int i = 0; i < nStreams; ++i)
			checkCuda( cudaStreamCreate(&stream[i]) );		

		//asynchronous version 1: loop over {copy, kernel, copy}
		cudaMemcpy(d_Od, Od, size_Od, cudaMemcpyHostToDevice);
		cudaMemcpy(d_OH, OH, size_OH, cudaMemcpyHostToDevice);
		cudaMemcpy(d_rcut2D, &rcut2D, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dz, &dz, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Lx, &Lx, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Ly, &Ly, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_nbin, &nbin, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_NOd, &NOd, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_NOH, &NOH, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		
		int gridx = ceil(Nmol/nStreams);
		if ( gridx < 1 ) gridx=1;
		
		dim3 grid(gridx,1);
		dim3 block(Dim,(NOd+NOH)*9/Dim);
		size_t blocksize = (NOd+NOH)*Dim+Dim+(NOd+NOH)*9+9+9+(NOd+NOH)*2;

		checkCuda( cudaEventRecord(startEvent,0) );
		for (int i = 0; i< nStreams; ++i)
		{
			int offset = i*Nmol/nStreams;
			checkCuda( cudaMemcpyAsync(&d_mol[offset*Dim], &mol[offset*Dim], size_mol/nStreams, cudaMemcpyHostToDevice, stream[i]) );
			Dist<<<grid,block,blocksize*sizeof(float),stream[i]>>>(
				d_Od, d_OH, &d_mol[offset*Dim], d_NOd, d_NOH, d_Lx, d_Ly, d_rcut2D, d_dz, d_nbin, 
				&d_Dens_Od[offset*nbin], &d_Dens_OH[offset*nbin], &d_Odval[offset]);
				cudaError_t error = cudaGetLastError();
				if( error != cudaSuccess)
				{
					printf("CUDA error: %s\n", cudaGetErrorString(error));
					exit(-1);
				}
			checkCuda( cudaMemcpyAsync(&Dens_Od[offset*nbin], &d_Dens_Od[offset*nbin], size_DensOd/nStreams, cudaMemcpyDeviceToHost, stream[i]) );
			checkCuda( cudaMemcpyAsync(&Dens_OH[offset*nbin], &d_Dens_OH[offset*nbin], size_DensOH/nStreams, cudaMemcpyDeviceToHost, stream[i]) );
			checkCuda( cudaMemcpyAsync(&Odval[offset], &d_Odval[offset], sizeof(int)*Nmol/nStreams, cudaMemcpyDeviceToHost, stream[i]) );
		}
		checkCuda( cudaEventRecord(stopEvent, 0) );
		checkCuda( cudaEventSynchronize(stopEvent) );

		cudaDeviceSynchronize();
		#pragma omp parallel num_threads(1)
		{
			#pragma omp for
			for (int i=0; i<Nmol; i++)
			{
				for (int j=0; j<nbin; j++)
				{
					#pragma omp critical
					{	
						//nMol_bin_Od[j] += Dens_Od[i][j];
						nMol_bin_Od[j] += Dens_Od[i*nbin+j];
						if (Odval[i] == 0)
						{
							//nMol_bin_OH[j] += Dens_OH[i][j];
							nMol_bin_OH[j] += Dens_OH[i*nbin+j];
						}
					}
				}
			}
		}

		// cleanup
		checkCuda( cudaEventDestroy(startEvent) );
		checkCuda( cudaEventDestroy(stopEvent) );
		checkCuda( cudaEventDestroy(dummyEvent) );
		for (int i = 0; i < nStreams; ++i)
		checkCuda( cudaStreamDestroy(stream[i]) );
	}//<--- End of loop over time frames


        //Wrting outputs
        output1 << "# This file was created " << currentDateTime() << endl;
        output1 << "# Command line:" << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5]  << endl;
        output1 << "@    title \"axial Density density of " << Sysname << " w/r Od and OH atoms." << "\"" << endl; //output header for OW number density
        output1 << "@    xaxis  label \"zdiff (nm)\"" << endl;
        output1 << "@    yaxis  label \"Density (1/nm^3)\"" << endl;
        output1 << "@TYPE xy" << endl;
        output1 << "@ view 0.15, 0.15, 0.75, 0.85" << endl;
        output1 << "@ legend on" << endl;
        output1 << "@ legend box on" << endl;
        output1 << "@ legend loctype view" << endl;
        output1 << "@ legend 0.78, 0.8" << endl;
        output1 << "@ legend length 2" << endl;
        output1 << "@ s0 legend \"near Od\"" << endl;
        output1 << "@ s1 legend \"near OH\"" << endl;

        dV = box[0][0]*box[1][1]*dz;
        float sum = 0;
        for (int i=0; i<nbin; i++)
        {
                output1 << setw(18) << setprecision(6) << fixed << dz*i;
                output1 << setw(18) << setprecision(6) << fixed << nMol_bin_Od[i]/nframes/dV/2;//for rcut2D=0.3, 23.14% area are Od
                output1 << setw(18) << setprecision(6) << fixed << nMol_bin_OH[i]/nframes/dV/2 << endl;// 48.12 area are OH
                //debug for mass density and polarization
		sum += nMol_bin_Od[i];
        }
        sum = sum/nframes;

        //Estimate cpu time for the calculations
        time_t nEnd = time(NULL);//End timer
        cout<<"Elapsed time is :  "<< nEnd-nStart << " seconds " << endl;
        log << "Elapsed time is :  "<< nEnd-nStart << " seconds " << endl;

        //Output dbg inform
	log << "\nDebug information: " << endl;
	log << "Number of frames: " << nframes << endl;
	log << "Total number of : " << Sysname << " atoms near Od within zcut = " << zcut << " nm: ";
	log << setw(12) << setprecision(6) << fixed << sum << endl;

	// Release memories
	free(Od); free(OH); cudaFreeHost(mol); cudaFreeHost(Dens_Od); cudaFreeHost(Dens_OH); cudaFreeHost(Odval);
	free(nMol_bin_Od); free(nMol_bin_OH);
	cudaFree(d_Od); cudaFree(d_OH); cudaFree(d_mol);
	cudaFree(d_Dens_Od); cudaFree(d_Dens_OH); cudaFree(d_Odval);

	// Close files
        close_trn(TRR);
        inputSOL.close();
        molinfo.close();
        output1.close();
        log.close();

	return 0;

}
