#include <stdio.h>
#include <math.h>

/* IN : 
    bias
    input_neurons   // Input neurons
    input_weights   // Convolution weights
    output_neurons  // Output neurons

*/

//convolution
__global__ void first_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out)
					 + (out_fm*row)
					 + col;
	//Stride
    int x_str = 0, y_str = 0;
    x_str = 3*(row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = 3*(col*str-pad);
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/(3*in_fm);
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str/3;
	}

	float product = 0.0;
	for(int i = 0; i < loopr; i++){
		for(int j = 0; j < loopc; j++){
			for(int k = 0; k < ker_channel; k++){
				product += in_nu[i*in_fm*ker_channel + j*ker_channel + k +  x_str + y_str] 
						*in_w[num_out*ker*ker*ker_channel + i*ker + j + k*ker*ker + x_pad*ker + y_pad];
			}
		}
	}
	product += bias[num_out];

	if(product < 0)
		product = 0;

	out_nu[out_position] = product;
}

__global__ void conv_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

	float product = 0.0;
	for(int i = 0; i < ker_channel; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				product += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
						*in_w[num_out*ker_channel*ker*ker + i*ker*ker + j*ker + k + x_pad*ker + y_pad];
			}
		}
	}
	product += bias[num_out];

	if(product < 0)
		product = 0;

	out_nu[out_position] = product;
}

__global__ void conv_split_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,
int in_fm,int out_fm,int str,int pad,int ker,int ker_channel,int start,int GPU_num)
{
	int num_out = blockIdx.x + start;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int x_pad = 0, y_pad = 0;
	int loopr = ker, loopc = ker;

	//Upper
	if(row*str < pad){
		x_pad = pad - row*str;
		loopr = ker - x_pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col*str < pad){
		y_pad = pad - col*str;
		loopc = ker - y_pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

    int start_num_kernel = 0;
    if(GPU_num == 0)
        start_num_kernel = 0;
    else
        start_num_kernel = ker_channel;

	float product = 0.0;
	for(int i = start_num_kernel; i < ker_channel << GPU_num; i++){
		for(int j = 0; j < loopr; j++){
			for(int k = 0; k < loopc; k++){
				product += in_nu[in_fm*in_fm*i + in_fm*j + k + x_str + y_str] 
						*in_w[num_out*ker_channel*ker*ker + (i-start_num_kernel)*ker*ker + j*ker + k + x_pad*ker + y_pad];
			}
		}
	}
	product += bias[num_out];

	if(product < 0)
		product = 0;

	out_nu[out_position] = product;
}

__global__ void norm_jjb(float *in_nu,float *out_nu,
float alpha,float beta,int local_size,int out_fm)
{
	int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    int input_position = (out_fm*row) + col;

    int nStart = 0, nEnd = 0;
    nStart=(num_out-2) > 1 ? (num_out-2) : 1 ;
    nEnd=(num_out+2) < gridDim.x ? (num_out+2) : gridDim.x ;

    float sum = 0.0;
    float result = 0.0;  
    for(int i = (nStart-1); i < (nEnd-1); i++){
        sum += pow((in_nu[i*out_fm*out_fm + input_position]),2);
    }
    result = (in_nu[out_position]) / (pow( 1 + ((alpha/local_size) * sum),beta));
    sum = 0.0;
    out_nu[out_position] = result;
}

__global__ void max_jjb(float *in_nu,float *out_nu,
int in_fm,int out_fm,int str,int pad,int ker)
{
    int num_out = blockIdx.x;
	int row_out_block = blockIdx.y;
	int col_out_block = blockIdx.z;
	int row_out_thread = threadIdx.x;
	int col_out_thread = threadIdx.y;

    int row = ((blockDim.x*row_out_block)+(row_out_thread));
    int col = ((blockDim.y*col_out_block)+(col_out_thread));  

	int out_position = (out_fm*out_fm*num_out) + (out_fm*row) + col;

    //Stride
    int x_str = 0, y_str = 0;
    x_str = (row*str-pad)*in_fm;
    x_str = x_str < 0 ? 0 : x_str;
    y_str = col*str-pad;
    y_str = y_str < 0 ? 0 : y_str;

	//Padding
	int loopr = ker, loopc = ker;

	//Upper
	if(row < pad){
		loopr = ker - pad;
	}
	//Bottom
	if(row >= out_fm - pad){
		loopr = in_fm - x_str/in_fm;
	}
	//Left
	if(col < pad){
		loopc = ker - pad;
	}
	//Right
	if(col >= out_fm - pad){
		loopc = in_fm -  y_str;
	}

    float max = 0.0;
    for(int i = 0; i < loopr; i++){
        for(int j = 0; j < loopc; j++){
            if(max < (in_nu[num_out*in_fm*in_fm + i*in_fm + j + x_str + y_str]))
                max = in_nu[num_out*in_fm*in_fm + i*in_fm + j + x_str + y_str];
        }
    }

    out_nu[out_position] = max;
}

//fully connected
__global__ void fc_jjb(float *bias,float *in_nu,float *in_w,float *out_nu,
int input, bool relu)
{
    int num_out = blockIdx.x;
	int weight = num_out * input;

	float result = 0.0;
	for(int i = 0; i < input; i++){
		result += in_nu[i] * in_w[weight+i];
	}
	result += bias[num_out];

	//ReLU
    if(relu == true){
        if(result < 0)
            result = 0;
    }

	out_nu[num_out] = result;
}