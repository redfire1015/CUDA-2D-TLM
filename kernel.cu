//**************************************//
//			James Holdsworth			//
//			  January 2024				//
//			  2D CUDA TLM				//
//**************************************//

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//C/C++ Includes
//IO
#include <stdio.h>
#include <iostream>
#include <iomanip>  // for setprecision
//File Handling
#include <string>
#include <fstream>
//Timing 
#include <ctime>

//Program Defines 
#define M_PI 3.1415926589793	// PI
#define c 299792458				// Speed of Light in a Vacuum 
#define mu0 M_PI*4e-7			// magnetic permeability in a Vacuum H/m
#define eta0 c*mu0				// wave impedance

#define defNX 100
#define defNY 100
#define defNT 1024				//Sets the starting number of NT


using namespace std;

//*************	Utility Function Declarations *************
/*
 * Function to Check for CUDA Errors and synchronise CUDA
 * @param None
 */
void cudaCheckAndSync();	//Function to check for CUDA Errors and Synchronise Device

//************* TLM Function Declarations *************
/*
 * Kernel that applies a Source Voltage to a specified input node (Ein)
 * @param Pointer to the device memory representing voltage array 1.
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 3.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Value of NX (Size of grid in X direction)
 * @param Value of NY (Size of grid in Y direction)
 * @param Value representing the X coord of the input probe.
 * @param Value representing the Y coord of the input probe.
 * @param Value representing the source voltage.
 */
__global__ void TLMsource(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY, const double E0, const int EinX, const int EinY);	// excitation function

/*
 * Kernel that 'scatters' an input impulse based on an applied source voltage
 * @param Pointer to the device memory representing voltage array 1.
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 3.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Value of NX (Size of grid in X direction).
 * @param Value of NY (Size of grid in Y direction).
 */
__global__ void TLMscatter(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY);	// TLM scatter process

/*
 * Kernel to connect the scattered impulses, Also applies boundary conditions
 * @param Pointer to the device memory representing voltage array 1.
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 3.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Value of NX (Size of grid in X direction).
 * @param Value of NY (Size of grid in Y direction).
 */
__global__ void TLMconnect(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY);		// TLM connect process

/*
 * Kernel that applied boundary conditions and determines the voltage at the output node (Eout)
 * @param Pointer to the device memory representing voltage array 1.
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 3.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Value of NX (Size of grid in X direction).
 * @param Value of NY (Size of grid in Y direction).
 * @param n The current time-step index
 * @param Value representing the X coord of the output probe.
 * @param Value representing the Y coord of the output probe.
 * @param Pointer to the device memory representing minimum X boundary reflection.
 * @param Pointer to the device memory representing maximum X boundary reflection.
 * @param Pointer to the device memory representing minimum Y boundary reflection.
 * @param Pointer to the device memory representing maximum Y boundary reflection.
 */
__global__ void TLMBoundaryOutput(double* d_V1, double* d_V2, double* d_V3, double* d_V4, double* dev_vout, const int NX, const int NY, const double rXmin, const double rXmax, const double rYmin, const double rYmax, const int n, const int EoutX, const int EoutY); // TLM Boundary process and Output Calculation


int main()
{
	cudaError_t cudaStatus;

	//Good practice to set device to use for kernel code
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to set Device to Device 0");
	}


	int NX = defNX; //Number of Nodes in X Direction
	int NY = defNY; //Number of Nodes in Y Direction
	int NT = defNT; //Number of Time steps
	double dl = 1;	//Set the node line segment length in meters

	//Calculated Variables
	double dt = dl / (sqrt(2.) * c);	//Set the time step duration

	//boundary coefficients
	double rXmin = -1;
	double rXmax = -1;
	double rYmin = -1;
	double rYmax = -1;


	double width = 20 * dt * sqrt(2.);	//Gaussian Input Signal Width
	double delay = 100 * dt * sqrt(2.); // The time delay before starting the input excitation
	int Ein[] = { 10,10 };	//Location of the input Voltage
	int Eout[] = { 15,15 }; //Location of the output Voltage Probe

	//CPU  variables
	double E0 = 0; //Variable for the input signal value
	double* h_Vout = new double[NT](); //Array to Store data from GPU and set to 0

	//Init arrays for GPU 
	double* d_V1 = nullptr;
	double* d_V2 = nullptr;
	double* d_V3 = nullptr;
	double* d_V4 = nullptr;
	double* d_Vout = nullptr;

	//Setup Blocks and Threads
	int threadsPerBlock = 256;
	int numBlocks = std::min(((NX * NY) + threadsPerBlock - 1) / threadsPerBlock, 2147483647); // Guarantees at least 1 Block (Max Blocks on newer cards is (2^31)-1) (Old Cards it is 65535 or 2^16-1)

	//Allocating Device Memory
	cudaCheckAndSync();
	cudaStatus = cudaMalloc((void**)&d_V1, (NX * NY * sizeof(double))); // Memory allocate for V1 array on device
	cudaStatus = cudaMalloc((void**)&d_V2, (NX * NY * sizeof(double))); // Memory allocate for V2 array on device
	cudaStatus = cudaMalloc((void**)&d_V3, (NX * NY * sizeof(double))); // Memory allocate for V3 array on device
	cudaStatus = cudaMalloc((void**)&d_V4, (NX * NY * sizeof(double))); // Memory allocate for V4 array on device
	cudaStatus = cudaMalloc((void**)&d_Vout, (NT * sizeof(double))); // Memory allocate for results array
	cudaCheckAndSync();

	//Setup Timing
	clock_t start, end;
	start = clock();

	//File streaming - Output Data
	ofstream output("2D_GPU_Voltage.out"); //Output of the voltage  
	ofstream gaussian_time("2D_GPU_gaussian_excitation.out");  // log excitation function to file 

	// Start of TLM algorithm
	// Loop over total time NT in steps of dt
	for (int n = 0; n < NT; n++) {
		//Source
		E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
		gaussian_time << n * dt << "  " << E0 << endl; //Writing the Source Voltage to a file  - Comment out for timing

		TLMsource << <1, 4 >> > (d_V1, d_V2, d_V3, d_V4, NX, NY, E0, Ein[0], Ein[1]); //Only 4 Operations so 1 Block with 4 Threads
		cudaCheckAndSync();

		//Scatter
		TLMscatter << <numBlocks, threadsPerBlock >> > (d_V1, d_V2, d_V3, d_V4, NX, NY); //Many operations so varying blocks and threads
		cudaCheckAndSync();

		//Connect
		TLMconnect << <numBlocks, threadsPerBlock >> > (d_V1, d_V2, d_V3, d_V4, NX, NY); //Many operations so varying blocks and threads
		cudaCheckAndSync();

		//Output
		TLMBoundaryOutput << <numBlocks, threadsPerBlock >> > (d_V1, d_V2, d_V3, d_V4, d_Vout, NX, NY, rXmin, rXmax, rYmin, rYmax, n, Eout[0], Eout[1]); //Only 1 Operation so 1 Block with 1 Thread
		cudaCheckAndSync();

		//Print progress to the terminal
		//Will Slow down processing (marginally) as time taken to print to screen
		//Comment Out for Timing
		if (n % 100 == 0)
			cout << n << endl;
	}

	//Once completed the calculations, copy the voltage output array back to the CPU to be written to file
	cudaStatus = cudaMemcpy(h_Vout, d_Vout, (sizeof(double) * NT), cudaMemcpyDeviceToHost); // Memory Copy back to CPU

	// Write Output Voltage at Eout to file
	for (int i = 0; i < NT; ++i) {
		output << i * dt << "  " << h_Vout[i] << endl; // Comment Out for Timing
	}

	//Closing Data logging Files
	output.close();
	gaussian_time.close();

	cout << "Done";
	end = clock(); //End Timing
	double TLM_Execution_Time = ((end - start) / (double)CLOCKS_PER_SEC); //Calculate Execution time
	cout << TLM_Execution_Time << '\n'; //Print time

	cin.get(); //Keep Terminal Open until Enter is pressed

	// Free allocated memory from GPU
	cudaFree(d_V1);
	cudaFree(d_V2);
	cudaFree(d_V3);
	cudaFree(d_V4);
	cudaFree(d_Vout);

	//Free CPU memory
	delete[]h_Vout;


	//Resetting CUDA Device - Ensures Proper operation for Nsight and Visual Profiler
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

//************* TLM Function Definitions *************

//TLM Source Definition
//Places a source stimulation at a given point wrt time.
__global__ void TLMsource(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY, const double E0, const int EinX, const int EinY) {

	// Unique Thread ID
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0) d_V1[(EinX * NY) + EinY] += E0;
	if (tid == 1) d_V2[(EinX * NY) + EinY] -= E0;
	if (tid == 2) d_V3[(EinX * NY) + EinY] -= E0;
	if (tid == 3) d_V4[(EinX * NY) + EinY] += E0;
}

//TLM Scatter Definition
//Calculates how the input wave interacts with nodes
__global__ void TLMscatter(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY) {

	// Local Thread Variable
	double V = 0; //Voltage variable (Unique to Each Thread)

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Unique Thread ID
	unsigned int stride = blockDim.x * gridDim.x;	//Strides to take inside the for loop based upon total available threads and blocks

	for (unsigned long long i = tid; i < (NX * NY); i += stride) {
		double I = ((d_V1[i] + d_V4[i] - d_V2[i] - d_V3[i]) / 2); // Calculate I

		V = 2 * d_V1[i] - I;
		d_V1[i] = V - d_V1[i];

		V = 2 * d_V2[i] + I;
		d_V2[i] = V - d_V2[i];

		V = 2 * d_V3[i] + I;
		d_V3[i] = V - d_V3[i];

		V = 2 * d_V4[i] - I;
		d_V4[i] = V - d_V4[i];
		//Don't need to synchronise here as threads only rely on independent values!
	}
}

//TLM Connect Definition
//Connect TLM nodes and update Voltages based on interactions between nodes 
__global__ void TLMconnect(double* d_V1, double* d_V2, double* d_V3, double* d_V4, const int NX, const int NY) {

	double tempV = 0; // Temp voltage

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Unique Thread ID
	unsigned int stride = blockDim.x * gridDim.x; //Strides to take inside the for loop based upon total available threads and blocks

	// Connect function performed on V2 and V4
	for (unsigned long long i = (tid + NY); i < (NX * NY); i += stride) { // Skip any x=0 values (use offset of NY)
		tempV = d_V2[i];
		d_V2[i] = d_V4[i - NY]; //Swap adjacent Values remembering array is flattened (adjacent X values are now 'NY' indexes apart)
		d_V4[i - NY] = tempV;
	}

	//Don't need to Synchronise here as each loop operated on different arrays (V2,V4 first then V1,V3)

	// Connect function performed on V1 and V3
	for (unsigned long long i = tid + 1; i < (NX * NY); i += stride) { // Skip any Y=0 values (Offset by 1, then check for each iteration)
		if (i % NY == 0) { // When i = a multiple of NY, skip (Skips all NY = 0 in the flattened arrays)
			// Do Nothing
		}
		else
		{
			tempV = d_V1[i];
			d_V1[i] = d_V3[i - 1]; //Swap adjacent Values remembering array is flattened (Y values are still above and below each-other)
			d_V3[i - 1] = tempV;
		}
	}
}

// TLM Boundary and Output function Definition
// Apply Boundary Conditions and calculate output voltage
__global__ void TLMBoundaryOutput(double* d_V1, double* d_V2, double* d_V3, double* d_V4, double* dev_vout, const int NX, const int NY, const double rXmin, const double rXmax, const double rYmin, const double rYmax, const int n, const int EoutX, const int EoutY) {

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Gets thread ID
	unsigned int stride = blockDim.x * gridDim.x; //Strides to take inside the for loop based upon total available threads and blocks

	// Calculate Boundary Conditions For V1 and V3
	for (unsigned long long i = tid; i < NX; i += stride) {
		d_V3[(i * NY) + (NY - 1)] *= rYmax;
		d_V1[(i * NY)] *= rYmin;
	}

	//No Sync Needed as operating on V1,V3 then V2,V4

	// Calculate Boundary Conditions For V2 and V4
	for (unsigned long long i = tid; i < NY; i += stride) {
		d_V4[((NX - 1) * NY) + i] *= rXmax;
		d_V2[i] *= rXmin;
	}

	__syncthreads(); // Synchronise Here as going to use V2 and V4 arrays 

	//Calculate output voltage and store on device array
	if (tid == 0)  dev_vout[n] = d_V2[EoutX * NY + EoutY] + d_V4[EoutX * NY + EoutY]; //Only run on first thread

}

//************* Utility Function Definitions *************

void cudaCheckAndSync()	//Function to check for CUDA Errors and Synchronise Device
{
	//Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cudaGetLastError() failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	//Check for Errors when Synchronising CudaDevice
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize() failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}