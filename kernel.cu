//**************************************//
//			James Holdsworth			//
//			  January 2024				//
//			  2D CUDA TLM				//
//**************************************//

//Cuda Includes
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
#define c 299792458			// speed of light in a vacuum
#define PI 3.1415926589793  // PI
#define mu0 PI*4e-7         // magnetic permeability in a vacuum H/m
#define eta0 c*mu0          // wave impedance in free space 

#define defNX 100
#define defNY 100
#define defNT 8192		//Sets the starting number of NT

//Essentially 2D -> 1D array means:
// X = x*NY 
// Y = y 

using namespace std;

//*************	Utility Function Declarations *************
/**
 * A kernel to apply a Source Voltage to a supplied input node (Ein)
 * @param None
 */
void cudaCheckAndSync();	//Function to check for CUDA Errors and Synchronise Device

//************* TLM Function Declarations *************
/**
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
__global__ void TLMsource(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const int EinX, const int EinY, const double E0);	// excitation function

/**
 * Kernel that 'scatters' an input impulse based on an applied source voltage
 * @param Pointer to the device memory representing voltage array 1.
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 3.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Value of NX (Size of grid in X direction).
 * @param Value of NY (Size of grid in Y direction).
 * @param Pointer to the device memory representing the Transmission line impedance.
 */
__global__ void TLMscatter(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const double _Z);	// TLM scatter process

/**
 * Kernel to connect the scattered impulses, Also applies boundary conditions
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
__global__ void TLMconnect(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const int n, const int EoutX, const int EoutY, const double rXmin, const double rXmax, const double rYmin, const double rYmax);		// TLM connect process, including boundary conditions

/**
 *
 * Kernel that determines the voltage at the output node (Eout)
 * @param Pointer to the device memory representing voltage array 2.
 * @param Pointer to the device memory representing voltage array 4.
 * @param Pointer to the device memory representing the voltage output array.
 * @param Value of NX (Size of grid in X direction).
 * @param Value of NY (Size of grid in Y direction).
 * @param n The current time-step index
 * @param Value representing the X coord of the output probe.
 * @param Value representing the Y coord of the output probe.
 */
__global__ void TLMoutput(double* dev_V2, double* dev_V4, double* dev_vout, const int NX, const int NY, const int n, const int EoutX, const int EoutY);


int main()
{
	cudaError_t cudaStatus;

	//Good practice to set device to use for kernel code
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to set Device to Device 0");
	}

	//Changeable Host Variables
	int NX = defNX; //Number of Nodes in X Direction
	int NY = defNY; //Number of Nodes in Y Direction
	int NT = defNT; //Number of Time steps
	double dl = 1;						//Set the node line segment length in meters

	//Calculated Host Variables
	double dt = dl / (sqrt(2.) * c);	//Set the time step duration
	double Z = eta0 / sqrt(2.); //Value for Impedance

	//boundary coefficients
	double rXmin = -1;
	double rXmax = -1;
	double rYmin = -1;
	double rYmax = -1;

	//input / output
	double width = 20 * dt * sqrt(2.);	//Gaussian Width
	double delay = 100 * dt * sqrt(2.); // set time delay before starting excitation
	int Ein[] = { 10,10 };	//Location of the input
	int Eout[] = { 15,15 }; //Location of the Voltage Probe

	//CPU (Host) variables
	double E0 = 0;
	double* h_Vout = new double[NT](); //Array to Store data from GPU, Dynamic to make iterating easier

	//2D mesh for GPU variables
	double* dev_V1;
	double* dev_V2;
	double* dev_V3;
	double* dev_V4;
	double* dev_Vout;

	//Setup Blocks and Threads
	int numThreads = 1024;
	int numBlocks = std::min(((NX * NY) + numThreads - 1) / numThreads, 2147483647); // Guarantees at least 1 Block (Max Blocks on newer cards is (2^31)-1

	//Allocating Device Memory
	cudaCheckAndSync();
	cudaStatus = cudaMalloc((void**)&dev_V1, (NX * NY * sizeof(double))); // Memory allocate for V1 array on device
	cudaStatus = cudaMalloc((void**)&dev_V2, (NX * NY * sizeof(double))); // Memory allocate for V2 array on device
	cudaStatus = cudaMalloc((void**)&dev_V3, (NX * NY * sizeof(double))); // Memory allocate for V3 array on device
	cudaStatus = cudaMalloc((void**)&dev_V4, (NX * NY * sizeof(double))); // Memory allocate for V4 array on device
	cudaStatus = cudaMalloc((void**)&dev_Vout, (NT * sizeof(double))); // Memory allocate for results array
	cudaCheckAndSync();

	//Setup Timing
	clock_t start, end;
	start = clock();

	//File streaming - Output Data
	ofstream output("2D_GPU_Voltage.out"); //Output of the voltage  - Comment out for timing 
	ofstream gaussian_time("2D_GPU_gaussian_excitation.out");  // log excitation function to file - Comment out for timing

	// Start of TLM algorithm
	// Loop over total time NT in steps of dt
	for (int n = 0; n < NT; n++) {
		//Source
		E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));
		gaussian_time << n * dt << "  " << E0 << endl; //Writing the Source Voltage to a file  - Comment out for timing
		TLMsource << <1, 4 >> > (dev_V1, dev_V2, dev_V3, dev_V4, NX, NY, Ein[0], Ein[1], E0); //Only 4 Operations so 1 Block with 4 Threads
		cudaCheckAndSync();

		//Scatter
		TLMscatter << <numBlocks, numThreads >> > (dev_V1, dev_V2, dev_V3, dev_V4, NX, NY, Z); //Many operations so varying blocks and threads
		cudaCheckAndSync();

		//Connect
		TLMconnect << <numBlocks, numThreads >> > (dev_V1, dev_V2, dev_V3, dev_V4, NX, NY, n, Eout[0], Eout[1], rXmin, rXmax, rYmin, rYmax); //Many operations so varying blocks and threads
		cudaCheckAndSync();

		//Output
		TLMoutput << <1, 1 >> > (dev_V2, dev_V4, dev_Vout, NX, NY, n, Eout[0], Eout[1]); //Only 1 Operation so 1 Block with 1 Thread
		cudaCheckAndSync();

		//Print progress to the terminal
		//Will Slow down processing (marginally) as time taken to print to screen
		if (n % 100 == 0)
			cout << n << endl;
	}

	//Once completed the calculations, copy the voltage output array back to the CPU to be written to file
	cudaStatus = cudaMemcpy(h_Vout, dev_Vout, (NT * sizeof(double)), cudaMemcpyDeviceToHost); // Memory Copy back to CPU

	// Output timing and voltage at Eout point
	for (int i = 0; i < NT; ++i) {
		output << i * dt << "  " << h_Vout[i] << std::endl; // Writes output to file 
	}

	//Closing Data logging Files
	output.close();
	gaussian_time.close();


	cout << "Done";
	end = clock(); //End Timing
	double TLM_Execution_Time = ((end - start) / (double)CLOCKS_PER_SEC); //Calculate Execution time
	std::cout << TLM_Execution_Time << '\n'; //Print time

	cin.get(); //Keep Terminal Open until Enter is pressed

	// Free allocated memory from GPU
	cudaFree(dev_V1);
	cudaFree(dev_V2);
	cudaFree(dev_V3);
	cudaFree(dev_V4);
	cudaFree(dev_Vout);

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
__global__ void TLMsource(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const int EinX, const int EinY, const double E0) {

	// Unique Thread ID
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//Only 4 Operations so going to limit them to run on 4 threads
	if (tid == 0) dev_V1[(EinX * NY) + EinY] = dev_V1[EinX * NY + EinY] + E0;
	if (tid == 1) dev_V2[(EinX * NY) + EinY] = dev_V2[EinX * NY + EinY] - E0;
	if (tid == 2) dev_V3[(EinX * NY) + EinY] = dev_V3[EinX * NY + EinY] - E0;
	if (tid == 3) dev_V4[(EinX * NY) + EinY] = dev_V4[EinX * NY + EinY] + E0;
	//Synchronisation takes place in main
}

//TLM Scatter Definition
//Calculates how the input wave interacts with nodes
__global__ void TLMscatter(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const double _Z) {

	// Local Thread Variable
	double V = 0; //Voltage Value


	// Thread identities
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Unique Thread ID
	unsigned int stride = blockDim.x * gridDim.x;	//Strides to take inside the for loop based upon total available threads and blocks

	for (size_t i = tid; i < NX * NY; i += stride) {
		// Tidied up
		double I = ((dev_V1[i] + dev_V4[i] - dev_V2[i] - dev_V3[i]) / 2); // Calculate coefficient
		//I = (2 * V1[(x * NY) + y] + 2 * V4[(x * NY) + y] - 2 * V2[(x * NY) + y] - 2 * V3[(x * NY) + y]) / (4 * Z);

		V = 2 * dev_V1[i] - I;         //port1
		dev_V1[i] = V - dev_V1[i];

		V = 2 * dev_V2[i] + I;         //port2
		dev_V2[i] = V - dev_V2[i];

		V = 2 * dev_V3[i] + I;         //port3
		dev_V3[i] = V - dev_V3[i];

		V = 2 * dev_V4[i] - I;         //port4
		dev_V4[i] = V - dev_V4[i];
		//Don't need to synchronise here as threads only rely on current values!
		//Reduces overhead
	}
	//Synchronisation takes place in main
}

//TLM Connect Definition
//Connect TLM nodes and update Voltages based on interactions between nodes 
__global__ void TLMconnect(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, const int NX, const int NY, const int n, const int EoutX, const int EoutY, const double rXmin, const double rXmax, const double rYmin, const double rYmax) { // boundary variables

	// Local Thread Variable
	double tempV = 0; // Temp voltage variable used to swap values

	// Thread identities
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Unique Thread ID
	unsigned int stride = blockDim.x * gridDim.x; //Strides to take inside the for loop based upon total available threads and blocks

	// Connect: Loop for ports 2 and 4
	for (size_t i = (tid + NY); i < (NX * NY); i += stride) { // Skip any x=0 values
		tempV = dev_V2[i];
		dev_V2[i] = dev_V4[i - NY];
		dev_V4[i - NY] = tempV;
	}


	// Connect: Loop for ports 1 and 3
	for (size_t i = tid + 1; i < (NX * NY); i += stride) { // Skip any Y=0 values
		// Skip when finding y = 0
		if (i % NY != 0) {
			tempV = dev_V1[i];
			dev_V1[i] = dev_V3[i - 1];
			dev_V3[i - 1] = tempV;
		}
	}
	__syncthreads(); // Sync between loops

	// Calculate Boundary Conditions For V1 and V3
	for (size_t x = tid; x < NX; x += stride) {
		dev_V3[x * NY + NY - 1] = rYmax * dev_V3[x * NY + NY - 1];
		dev_V1[x * NY] = rYmin * dev_V1[x * NY];
	}


	// Calculate Boundary Conditions For V2 and V4
	for (size_t y = tid; y < NY; y += stride) {
		dev_V4[(NX - 1) * NY + y] = rXmax * dev_V4[(NX - 1) * NY + y];
		dev_V2[y] = rXmin * dev_V2[y];
	}
	//Synchronisation takes place in main
}

// TLM output function
// Kernel call that only performs a singular addition
// Could be done in CPU code!
__global__ void TLMoutput(double* dev_V2, double* dev_V4, double* dev_vout, const int NX, const int NY, const int n, const int EoutX, const int EoutY) {

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; //Gets thread ID - Only called on a single thread on a single block but good practise in case kernel called with more than 1 thread

	if (tid == 0) { //Only run on first thread
		dev_vout[n] = dev_V2[EoutX * NY + EoutY] + dev_V4[EoutX * NY + EoutY]; //Calculate output voltage and store on device array
	}
	//Synchronisation takes place in main
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