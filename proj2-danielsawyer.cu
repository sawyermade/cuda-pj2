/* ==================================================================
	Programmer: Daniel Sawyer (danielsawyer@mail.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc proj1-danielsawyer.cu -o SDH in the rc machines
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
			printf("\n T:%lld \n", abs(total_cnt - total_cnt2));
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

//CUDA KERNEL FOR SDH
__global__ void PDH_Cuda(atom *d_atom_list, bucket *d_histogram, long long d_PDH_acnt, double d_PDH_res) {

	double dist;
	int i, j, h_pos;

	i = threadIdx.x + blockDim.x * blockIdx.x;
	//if(i == 0) printf("\nTHE I VALUE = 0\n");
	//if(i == 0) printf("\nwarpSize = %d\n", warpSize);
	for(j = i+1; j < d_PDH_acnt; ++j) {

		dist = sqrt( (d_atom_list[i].x_pos - d_atom_list[j].x_pos)*(d_atom_list[i].x_pos - d_atom_list[j].x_pos) +
					 (d_atom_list[i].y_pos - d_atom_list[j].y_pos)*(d_atom_list[i].y_pos - d_atom_list[j].y_pos) +
					 (d_atom_list[i].z_pos - d_atom_list[j].z_pos)*(d_atom_list[i].z_pos - d_atom_list[j].z_pos) );
		h_pos = (int)(dist/d_PDH_res);		
		atomicAdd((unsigned long long int*)&d_histogram[h_pos].d_cnt,1);
		//d_histogram[h_pos].d_cnt++;
		//__syncthreads();
	}
}

void CudaPrep(bucket * histogram2) {

	//sizes of atom and bucket arrays
	int size_atom = sizeof(atom)*PDH_acnt;
	int size_hist = sizeof(bucket)*num_buckets;

	//grid and block sizes
	int dev = 0;
	cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
	//printf("\nWARP = %d\n", deviceProp.warpSize);
	dim3 threads(deviceProp.warpSize);
	dim3 grid(ceil((float)PDH_acnt/threads.x));

	//Device Vars
	bucket *d_histogram;
	atom *d_atom_list;

	//Allocate device memory
	cudaMalloc((void **) &d_histogram, size_hist);
	cudaMalloc((void**) &d_atom_list, size_atom);

	//Copy to device
	cudaMemcpy(d_atom_list, atom_list, size_atom, cudaMemcpyHostToDevice);
	cudaMemset(d_histogram, 0, size_hist);

	//run cuda kernel
	PDH_Cuda<<<grid,threads>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);

	//copy new gpu histogram back to host from device
	cudaMemcpy(histogram2, d_histogram, size_hist, cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(d_histogram); cudaFree(d_atom_list);
}

int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	//printf("args are %d and %f\n", PDH_acnt, PDH_res);

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
	//memset(histogram2, 0, size_hist);

	//start time
	gettimeofday(&startTime, &Idunno);

	//run on GPU
	CudaPrep(histogram2);

	//check runtime
	report_running_time(1);

	//print device histogram
	output_histogram(histogram2);

	//Difference between cpu and gpu
	printf("\nCPU vs GPU Histogram Differences\n");
	output_histogram(histogram, histogram2);

	//Free memory.
	free(histogram); free(atom_list);
	
	return 0;
}
