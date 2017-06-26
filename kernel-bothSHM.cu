__global__ void PDH_Cuda_2(atom *d_atom_list, bucket *d_histogram, long long d_PDH_acnt, double d_PDH_res, int bsize, int nblocks, int nbuckets) {

	//code goes here.
	register double dist;
	register int i, j, h_pos;
	register atom part;
	//register atom part2;
	
	extern __shared__ bucket s_histogram[];
	__shared__ atom s_atom_list[32];

	
	part = d_atom_list[threadIdx.x + blockDim.x * blockIdx.x];
	for(i = threadIdx.x; i < nbuckets; i += blockDim.x) {
		s_histogram[i].d_cnt = 0;
	}
	__syncthreads();
	
	if(threadIdx.x + blockDim.x * blockIdx.x < d_PDH_acnt) {
	//#pragma unroll
	for(i = blockIdx.x + 1; i < nblocks; ++i) {
		
		s_atom_list[threadIdx.x] = d_atom_list[i*blockDim.x + threadIdx.x];
		__syncthreads();

		if(i*blockDim.x < d_PDH_acnt)
		for(j = 0; j < blockDim.x; ++j) {
			
			if(i*blockDim.x + j < d_PDH_acnt) {
				//part2 = s_atom_list[j];
				dist = sqrt(
					(part.x_pos - s_atom_list[j].x_pos)*(part.x_pos - s_atom_list[j].x_pos) +
					(part.y_pos - s_atom_list[j].y_pos)*(part.y_pos - s_atom_list[j].y_pos) +
					(part.z_pos - s_atom_list[j].z_pos)*(part.z_pos - s_atom_list[j].z_pos));
					// (part.x_pos - part2.x_pos)*(part.x_pos - part2.x_pos) +
					// (part.y_pos - part2.y_pos)*(part.y_pos - part2.y_pos) +
					// (part.z_pos - part2.z_pos)*(part.z_pos - part2.z_pos));

				//if(j == 2)
					//printf("\ndist = %f\n", dist);
				//__syncthreads();
				h_pos = (int)(dist/d_PDH_res);
				//printf("\nh_pos = %d\n", h_pos);

				atomicAdd((unsigned long long int*)&s_histogram[h_pos].d_cnt,1);
				// __syncthreads(); //maybe?
			}
			//__syncthreads(); //maybe?
			//printf()
		}
		__syncthreads(); //maybe?
	}}

	s_atom_list[threadIdx.x] = part;
	__syncthreads();

	if(threadIdx.x + blockDim.x * blockIdx.x < d_PDH_acnt)
	for(i = threadIdx.x + 1; i < blockDim.x; ++i) {
		
		if(blockDim.x*blockIdx.x + i < d_PDH_acnt) {
			//part2 = s_atom_list[i];
			dist = sqrt(
				(part.x_pos - s_atom_list[i].x_pos)*(part.x_pos - s_atom_list[i].x_pos) +
				(part.y_pos - s_atom_list[i].y_pos)*(part.y_pos - s_atom_list[i].y_pos) +
				(part.z_pos - s_atom_list[i].z_pos)*(part.z_pos - s_atom_list[i].z_pos));
				// (part.x_pos - part2.x_pos)*(part.x_pos - part2.x_pos) +
				// (part.y_pos - part2.y_pos)*(part.y_pos - part2.y_pos) +
				// (part.z_pos - part2.z_pos)*(part.z_pos - part2.z_pos));

			h_pos = (int)(dist/d_PDH_res);

			atomicAdd((unsigned long long int*)&s_histogram[h_pos].d_cnt,1);
		}
	}
	__syncthreads();

	for(i = threadIdx.x; i < nbuckets; i += blockDim.x) {
		atomicAdd((unsigned long long int*)&d_histogram[i].d_cnt,s_histogram[i].d_cnt);
	}
}