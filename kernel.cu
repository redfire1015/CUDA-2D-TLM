//**********************************//
//			James Holdsworth		//
//			  January 2024			//
//			  2D CUDA TLM			//
//**********************************//

//Cuda Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//Additional Includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>  // for setprecision
#include <ctime> //For Timing

//Program Defines 
#define c 299792458			// speed of light in a vacuum
#define PI 3.1415926589793  // mmmm PI
#define mu0 PI*4e-7         // magnetic permeability in a vacuum H/m
#define eta0 c*mu0          // wave impedance in free space 

#define maxNT 4096			//Sets the maximum number of timesteps. Going to use a power of 2 as makes FFT easier


using namespace std;

//*************	Utility Function Declarations *************
double** declare_array2D(int, int);							//Generic function to create 2D arrays
double** allocateAndCopyToDevice(double** V, int NX, int NY); //Function to create 2D arrays on Cuda Device
void freeDeviceMemory(double** d_V, int NX);

//************* TLM Function Declarations *************
void TLMsource(double& E0, int& n, double& dt, double& delay, double& width, ofstream& gaussian_time, double** V1, double** V2, double** V3, double** V4, int* Ein);				// excitation function
__global__ void TLMscatter(double*, double*, int, double);	// TLM scatter process
__global__ void TLMconnect(double*, double*, int);			// TLM connect process, inlcuding boundary conditions
__global__ void TLMboundry(double&, double&, double&);



int main()
{
	cudaError_t cudaStatus;

	//Good practice to set device to use for kernal code
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to set Device to Device 0");
	}

	//Setup Timing
	clock_t start, end;
	start = clock();

	//Setup Host Variables
	int NX = 100; //Number of Nodes in X Direction
	int NY = 100; //Number of Nodes in Y Direction
	int NT = 512; //Number of Time steps
	double dl = 1; //Set the node line segment length in metres
	double dt = dl / (sqrt(2.) * c); //Set the time step duration

	//2D mesh variables
	double I = 0, tempV = 0, E0 = 0, V = 0;
	double** V1 = declare_array2D(NX, NY);
	double** V2 = declare_array2D(NX, NY);
	double** V3 = declare_array2D(NX, NY);
	double** V4 = declare_array2D(NX, NY);

	//2D Mesh variables - Device
	double** d_V1 = allocateAndCopyToDevice(d_V1, NX, NY);
	double** d_V2 = allocateAndCopyToDevice(d_V1, NX, NY);
	double** d_V3 = allocateAndCopyToDevice(d_V1, NX, NY);
	double** d_V4 = allocateAndCopyToDevice(d_V1, NX, NY);

	//Value for Impedance
	double Z = eta0 / sqrt(2.);

	//boundary coefficients
	double rXmin = -1;
	double rXmax = -1;
	double rYmin = -1;
	double rYmax = -1;

	//input / output
	double width = 20 * dt * sqrt(2.); //Gaussian Width
	double delay = 100 * dt * sqrt(2.); // set time delay before starting excitation
	int Ein[] = { 10,10 }; //Location of the input
	int Eout[] = { 15,15 }; //Location of the Voltage Probe

	//File streaming - Timing Data
	ofstream computation_time("GPUTiming.out");  // log Timing Information to file - Comment out for timing

	for (; NT <= maxNT; NT *= 2) //Looping for Multiple Values of NT (varying timestep)
	{
		//File straming - Output Data
		ofstream output("2D_GPU_Voltage.out"); //Output of the voltage  - Comment out for timing 
		ofstream gaussian_time("2D_GPU_gaussian_excitation.out");  // log excitation function to file - Comment out for timing

		// Start of TLM algorithm
		// loop over total time NT in steps of dt

		for (int n = 0; n < NT; n++) {

			//SOURCE
			TLMsource(E0, n, dt, delay, width, gaussian_time, V1, V2, V3, V4, Ein);

			//SCATTER

			// Launch the kernel
			dim3 block(16, 16);
			dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);
			//TLMscatter << <grid, block >> > (d_V, NX, NY);

			//Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("cudaGetLastError() failed: %s\n", cudaGetErrorString(cudaStatus));
			}

			//Check for Errors when Synchornising CudaDevice
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
			}

			//Scattering
			for (int x = 0; x < NX; x++) {
				for (int y = 0; y < NY; y++) {
					I = (2 * V1[x][y] + 2 * V4[x][y] - 2 * V2[x][y] - 2 * V3[x][y]) / (4 * Z);

					V = 2 * V1[x][y] - I * Z;         //port1
					V1[x][y] = V - V1[x][y];
					V = 2 * V2[x][y] + I * Z;         //port2
					V2[x][y] = V - V2[x][y];
					V = 2 * V3[x][y] + I * Z;         //port3
					V3[x][y] = V - V3[x][y];
					V = 2 * V4[x][y] - I * Z;         //port4
					V4[x][y] = V - V4[x][y];
				}
			}







			//connect
			for (int x = 1; x < NX; x++) {
				for (int y = 0; y < NY; y++) {
					tempV = V2[x][y];
					V2[x][y] = V4[x - 1][y];
					V4[x - 1][y] = tempV;
				}
			}
			for (int x = 0; x < NX; x++) {
				for (int y = 1; y < NY; y++) {
					tempV = V1[x][y];
					V1[x][y] = V3[x][y - 1];
					V3[x][y - 1] = tempV;
				}
			}

			//boundary
			for (int x = 0; x < NX; x++) {
				V3[x][NY - 1] = rYmax * V3[x][NY - 1];
				V1[x][0] = rYmin * V1[x][0];
			}
			for (int y = 0; y < NY; y++) {
				V4[NX - 1][y] = rXmax * V4[NX - 1][y];
				V2[0][y] = rXmin * V2[0][y];
			}


			output << n * dt << "  " << V2[Eout[0]][Eout[1]] + V4[Eout[0]][Eout[1]] << endl; // Writing the output voltage to file - Comment out for timing //Code provided by steve

			//Output the progress to the terminal
			if (n % 100 == 0)
				cout << n << endl;

		}
		//Closing Data loggging Files
		output.close();
		gaussian_time.close();

		//Calculate how long it took to complete a TLM simulation for a set number of NT
		cout << "Done";
		end = clock(); //End Timing
		double TLM_Execution_Time = ((end - start) / (double)CLOCKS_PER_SEC);
		std::cout << TLM_Execution_Time << '\n';

		//Writing the computation time to file - Comment out for timing
		computation_time << NX << "  " << NY << "  " << NT << "  " << TLM_Execution_Time << endl;
	}
	computation_time.close(); //Closing Timing File
	cin.get(); //Keep Terminal Open until Enter is pressed

	//Free Device Memory
	freeDeviceMemory(d_V1, NX);
	freeDeviceMemory(d_V2, NX);
	freeDeviceMemory(d_V3, NX);
	freeDeviceMemory(d_V4, NX);

	// Free host memory
	for (int x = 0; x < NX; x++) {
		delete[] V1[x];
		delete[] V2[x];
		delete[] V3[x];
		delete[] V4[x];
	}

	//Resetting Cuda Device - Ensures Proper operation for Nsight and Visual Profiler
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


//************* TLM Function Definitions *************

//TLM Source Definition
void TLMsource(double& E0, int& n, double& dt, double& delay, double& width, ofstream& gaussian_time, double** V1, double** V2, double** V3, double** V4, int* Ein)
{
	E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
	// log value of gaussian voltage to file
	gaussian_time << n * dt << "  " << E0 << endl; //Writing the Source Voltage to a file  - Comment out for timing
	V1[Ein[0]][Ein[1]] = V1[Ein[0]][Ein[1]] + E0;
	V2[Ein[0]][Ein[1]] = V2[Ein[0]][Ein[1]] - E0;
	V3[Ein[0]][Ein[1]] = V3[Ein[0]][Ein[1]] - E0;
	V4[Ein[0]][Ein[1]] = V4[Ein[0]][Ein[1]] + E0;

}

//TLM Scatter Definition
__global__ void TLMscatter(double*, double*, int, double)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


}

//TLM Connect Definition
__global__ void TLMconnect(double*, double*, int)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

}

//TLM Boundry Definition
__global__ void TLMboundry(double& a, double& b, double& cc)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

}

//************* Utility Function Definitions *************

//Declare 2D Array on Host
double** declare_array2D(int NX, int NY) {
	double** V = new double* [NX];
	for (int x = 0; x < NX; x++) {
		V[x] = new double[NY];
	}

	//Setting to Zero
	for (int x = 0; x < NX; x++) {
		for (int y = 0; y < NY; y++) {
			V[x][y] = 0;
		}
	}
	return V;
}

// Function to allocate and copy 2D array to the device
double** allocateAndCopyToDevice(double** V, int NX, int NY) {
	double** d_V;

	// Allocate device memory
	cudaMalloc((void**)&d_V, NX * sizeof(double*));
	for (int i = 0; i < NX; i++) {
		cudaMalloc((void**)&(d_V[i]), NY * sizeof(double));
	}

	// Copy data from host to device
	cudaMemcpy(d_V, V, NX * sizeof(double*), cudaMemcpyHostToDevice);
	for (int i = 0; i < NX; i++) {
		cudaMemcpy(d_V[i], V[i], NY * sizeof(double), cudaMemcpyHostToDevice);
	}

	return d_V;
}

// Function to free device memory
void freeDeviceMemory(double** d_V, int NX) {
	for (int i = 0; i < NX; i++) {
		cudaFree(d_V[i]);
	}
	cudaFree(d_V);
}