using namespace std;

class Atom{
public:
int sn;
string res;
string name;
int idx;
double x,y,z;
};

const double pi = 3.1415926535897;
float dist_pbc_better (float *a, float  *b, float Lx, float Ly);
float dist_pbc (Atom a, Atom b, float Lx, float Ly);
float dist3D_pbc(Atom a, Atom b, float Lx, float Ly);
Atom cartesian_pbc(Atom a, Atom b, float Lx, float Ly);
double cos (Atom a);
double cosOd_Ion(Atom od, Atom ion);
double cosOdOH_SOL(Atom ow, Atom hw1, Atom hw2);
//CUDA kernel
//__global__ void Dist(float* Od, float* OH, float* OW, int* NOd, int* NOH, float* Lx, float* Ly, float* rcut2D, float* dz, int* nbin, float* r2DOd, float* r2DOH, float* zdiff_Od, float* zdiff_OH, float* Dens_Od, float* Dens_OH, int* Odval);
__global__ void Dist(float* Od, float* OH, float* OW, int* NOd, int* NOH, float* Lx, float* Ly, float* rcut2D, float* dz, int* nbin, float* Dens_Od, float* Dens_OH, int* Odval);
