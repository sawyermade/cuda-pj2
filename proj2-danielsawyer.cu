/* ==================================================================
	Programmer: Daniel Sawyer (danielsawyer@mail.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc proj2-danielsawyer.cu -o proj2 in the rc machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

//MY INCLUDES
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
int blockSize;

/* These are for an old way of tracking time */
struct timezone Idunno;
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}
//overloaded to show GPU time
double report_running_time(int blah) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("\nRunning time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}
//overloaded taking 1 arg
void output_histogram(bucket* histogram1){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram1[i].d_cnt);
		total_cnt += histogram1[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}
//overloaded taking 2 args
void output_histogram(bucket* histogram1, bucket* histogram2){
	int i; 
	long long total_cnt = 0, total_cnt2 = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", abs(histogram1[i].d_cnt - histogram2[i].d_cnt));
		total_cnt += histogram1[i].d_cnt;
		total_cnt2 += histogram2[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%llu \n", abs(total_cnt - total_cnt2));
		else printf("| ");
	}
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

__device__ double pdist(atom &L, atom &R) {

	double x = L.x_pos - R.x_pos;
	double y = L.y_pos - R.y_pos;
	double z = L.z_pos - R.z_pos;

	return sqrt(x*x + y*y + z*z);
}


/**********	KERNEL STUFF **********/

//NAIVE
__global__ void Naive(atom* dev_points, bucket *dev_Hist, int d_PDH_acnt, int d_PDH_res) {

	float dist = 0.0;
	int h_ind = 0;
	atom currentPt;
	unsigned long long int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < d_PDH_acnt) {
		
		currentPt = dev_points[tid];

		for(int i = tid+1; i < d_PDH_acnt; i++) {

			dist = pdist(currentPt, dev_points[i]);

			h_ind = (int)dist / d_PDH_res;

			atomicAdd((unsigned long long int*)&dev_Hist[h_ind].d_cnt, 1);
		}
	}
}

//CUDA KERNEL ALGO 3
__global__ void PDH_Algo3(atom *d_atom_list, bucket *d_histogram, long long d_PDH_acnt, double d_PDH_res, int nbuckets, int nblocks) {

	//vars
	register double dist;
	register int i, j, h_pos, tx = threadIdx.x, dx = blockDim.x, bx = blockIdx.x;
	register long long tid = threadIdx.x + blockDim.x * blockIdx.x;
	register atom L;

	//shared vars
	extern __shared__ atom smem[];
	atom* R = (atom*)smem;
	bucket* s_histogram = (bucket*)&smem[dx];
	
	//initialize histogram to 0 and load L in parallel
	for(i = tx; i < nbuckets; i += dx)
		s_histogram[i].d_cnt = 0;

	L = d_atom_list[tid];
	__syncthreads();

	//runs through other blocks
	for(i = bx + 1; i < nblocks; i++) {

		//loads tile of atoms into shared mem
		R[tx] = d_atom_list[tx + i*dx];
		__syncthreads();

		//runs through the shared mem, saves to hist
		if(i*dx < d_PDH_acnt)
		for(j = 0; j < dx; j++) {
			
			if(j + i*dx < d_PDH_acnt) {

				dist = pdist(L, R[j]);

				h_pos = (int)(dist/d_PDH_res);

				atomicAdd((unsigned long long int*)&s_histogram[h_pos].d_cnt,1);
			}
		}
		__syncthreads();
	}

	//loads currents block into shared
	R[tx] = L;
	__syncthreads();

	//calcs distances for current block
	if(tid < d_PDH_acnt)
	for(j = tx + 1; j < dx; j++) {

		if(j + bx*dx < d_PDH_acnt) {

			dist = pdist(L, R[j]);

			h_pos = (int)(dist/d_PDH_res);

			atomicAdd((unsigned long long int*)&s_histogram[h_pos].d_cnt,1);
		}
	}
	__syncthreads();

	//saves back to global
	for(i = tx; i < nbuckets; i += dx)
		atomicAdd((unsigned long long int*)&d_histogram[i].d_cnt,s_histogram[i].d_cnt);
}

/**********	KERNEL STUFF **********/

//preps for kernel call
float CudaPrep(bucket * histogram2) {

	//sizes of atom and bucket arrays
	int size_atom = sizeof(atom)*PDH_acnt;
	int size_hist = sizeof(bucket)*num_buckets;

	//grid and block sizes
	int dev = 0;
	cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
	
	dim3 threads(blockSize);
	//dim3 threads(deviceProp.warpSize);
	dim3 grid(ceil((float)PDH_acnt/threads.x));

	//Device Vars
	bucket *d_histogram;
	atom *d_atom_list;
	int num_blocks = ceil((float)PDH_acnt/threads.x);

	//Allocate device memory
	cudaMalloc((void **) &d_histogram, size_hist);
	cudaMalloc((void**) &d_atom_list, size_atom);

	//Copy to device
	cudaMemcpy(d_atom_list, atom_list, size_atom, cudaMemcpyHostToDevice);
	cudaMemset(d_histogram, 0, size_hist);

	//kernel execution time crap
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	//run cuda kernel
	//Naive<<<grid, threads>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);
	PDH_Algo3<<<grid, threads, num_buckets*sizeof(bucket) + threads.x*sizeof(atom)>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets, num_blocks);

	//kernel execution stop
	//cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy new gpu histogram back to host from device
	cudaMemcpy(histogram2, d_histogram, size_hist, cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(d_histogram); cudaFree(d_atom_list);

	return elapsedTime;
}

int main(int argc, char **argv)
{
	if(argc < 4) {

		printf("\nError, too few arguments. Usage: ./proj2 {#of_samples} {bucket_width} {block_size}.\n");
		return -1;
	}

	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	blockSize = atof(argv[3]);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram();


	/* NEW SHIT */

	//New histogram that will come from the device
	bucket *histogram2 = (bucket*)malloc(sizeof(bucket)*num_buckets);
	memset(histogram2, 0, size_hist);

	//start time
	gettimeofday(&startTime, &Idunno);

	//run on GPU
	float elapsedTime = CudaPrep(histogram2);

	//check runtime
	report_running_time(1);

	//print device histogram
	output_histogram(histogram2);

	//Difference between cpu and gpu
	printf("\nCPU vs GPU Histogram Differences\n");
	output_histogram(histogram, histogram2);

	//Free memory.
	free(histogram); free(atom_list);

	printf("\n******** Total Running Time of Kernel = %0.5f ms *******\n", elapsedTime);
	printf("\n******** Total Running Time of Kernel = %0.5f sec *******\n", elapsedTime/1000);
	
	return 0;
}
