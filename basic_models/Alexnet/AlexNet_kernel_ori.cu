#include <stdio.h>
#include <math.h>

/* IN : 
    bias
    input_neurons   // Input neurons
    input_weights   // Convolution weights
    output_neurons  // Output neurons
    start_num_out   // Convolution을 나눠서 하기 위해 output개수를 두 파트로 나눠주는 num
    num_out         // Number of output feature map
    conv_index      // Convolution part number
    out_row         // Output row
    out_col         // Output column
    stride_width    // Stride width
    pad             // Padding 
    in_row          // Input row
    in_col          // Input column
    kernel          // Size of kernel(filter) 
    num_kernel      // Number of kernels(filters)
    split_row       // split threadblock if it is bigger than 32
    split_col       // split threadblock if it is bigger than 32
*/

//convolution
__global__ void first_conv(float *bias,float *input_neurons,float *input_weights,float *output_neurons, int input_size, int channel, int out_row, int out_col, int stride_width, int kernel, int split_row, int split_col)
{
    float result = 0.0;
    int rowstride = 0,colstride = 0;
    int output = blockIdx.x;
    int row = threadIdx.x + split_row;
    int col = threadIdx.y + split_col;
    rowstride = col * stride_width * channel;
    colstride = channel * row * stride_width * input_size;

    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < kernel; i++){
        for(int j = 0; j < kernel; j++){
            for(int k = 0; k < channel; k++){
                result += (input_neurons[i*input_size*channel + j*channel + k + rowstride + colstride] 
                            * input_weights[i*kernel + j + k*kernel*kernel + (output*kernel*kernel*channel)]);
            }
        }
    }
    
    result += bias[output];
    if(result < 0) /* RELU Layer */
        result = 0; // max(0,x)
    output_neurons[output*out_row*out_col + row*out_col + col] = result;
    result = 0.0;
}

__global__ void conv(float *bias, float *input_neurons, float *input_weights, float *output_neurons, int out_row, int out_col, int stride_width, int pad, int kernel, int num_kernel, int start_index, int GPU_num, int split_row, int split_col)
{
    //number of output feature map
    int out_fe_num = blockIdx.x + start_index;    //0 ~ output feature map 개수-1
    //row of output feature map
    int out_fe_row = threadIdx.x + split_row;   //0 ~ output feature map row-1
    //column of output feature map
    int out_fe_col = threadIdx.y + split_col;   //0 ~ output feature map column-1

    int start_row_left = 0;
    int start_row_right = 0;
    int start_col_upper = 0;
    int start_col_bottom = 0;
    int kernel_row = 0;
    int kernel_col = 0;
    int start_num_kernel = 0;

    kernel_row = kernel;
    kernel_col = kernel;

    //padding부분의 연산을 제외해주기 위해 변수 설정(입,출력 값들이 text에 어떻게 저장되는 지를 고려하여 변수 설정)
    if(out_fe_row < pad){
        start_row_left = pad - out_fe_row;
        kernel_row = kernel - start_row_left;
    }
    else
        start_row_right = (out_fe_row - pad) * out_row;

    if(out_fe_col < pad){
        start_col_upper = pad - out_fe_col;
        kernel_col = kernel - start_col_upper;
    }
    else
        start_col_bottom = out_fe_col * stride_width;

    if(out_fe_row >= out_row - pad)
        kernel_row = out_row + pad - out_fe_row;

    if(out_fe_col >= out_col - pad)
        kernel_col = out_col + pad - out_fe_col;

    if(GPU_num == 0)
        start_num_kernel = 0;
    else
        start_num_kernel = num_kernel;

    //compute convolution
    float result = 0.0;
    for(int i = start_num_kernel; i < (num_kernel << GPU_num); i++){
        for(int j = 0; j < kernel_row; j++){
            for(int k = 0; k < kernel_col; k++){
                //sum of multiplication of each pixel
                //행렬 성분 index의 경우 tango에서 제공한 txt파일에 나열된 값들을 기반으로 제작
                //weight 값 및 bias 값을 따로 구하여 파일로 제작한다면 아래 수식을 수정해야함
                result += (input_neurons[i*out_row*out_col + j*out_col + k + start_row_right + start_col_bottom] 
                            * input_weights[out_fe_num*kernel*kernel*num_kernel + (i-start_num_kernel)*kernel*kernel + j*kernel + k + kernel*start_row_left + start_col_upper]);
            }
        }
    }
    //plus bias
    result += bias[out_fe_num];

    //ReLU
    if(result < 0)
        result = 0;
    output_neurons[out_fe_num*out_row*out_col + out_fe_row*out_col + out_fe_col] = result;
    // if(result > 0)
    //     printf("%f\t",result);
    result = 0.0;
    if(out_fe_col >= pad)
        start_col_bottom += stride_width;
}

//max pooling
__global__ void maxpool(float *input_neurons, float *output_neurons, int num_out, int out_row, int out_col, int stride_width, int kernel, int in_row, int in_col, int split_row, int split_col)
{
    //number of output feature map
    int out_fe_num = blockIdx.x;    //0 ~ output feature map 개수-1
    //row of output feature map
    int out_fe_row = threadIdx.x + split_row;   //0 ~ output feature map row-1
    //column of output feature map
    int out_fe_col = threadIdx.y + split_col;   //0 ~ output feature map column-1 

    int row_default = 0;
    int col_default = 0;

    row_default = out_fe_col * stride_width;
    col_default = out_fe_row * stride_width * in_col;

    float max = 0.0;
    for(int i = 0; i < kernel; i++){
        for(int j = 0; j < kernel; j++){
            if(max < (input_neurons[out_fe_num*in_row*in_col + i*in_col + j + row_default + col_default]))
                max = input_neurons[out_fe_num*in_row*in_col + i*in_col + j + row_default + col_default];
        }
    }
    output_neurons[out_fe_num*out_row*out_col + out_fe_row*out_col + out_fe_col] = max;
    max = 0.0;
    row_default += stride_width;
}

//normalization
__global__ void norm(float *input_neurons, float *output_neurons, float alpha, float beta, int local_size, int num_out, int out_row, int out_col, int split_row, int split_col)
{
    //number of output feature map
    int out_fe_num = blockIdx.x;    //0 ~ output feature map 개수-1
    //row of output feature map
    int out_fe_row = threadIdx.x + split_row;   //0 ~ output feature map row-1
    //column of output feature map
    int out_fe_col = threadIdx.y + split_col;   //0 ~ output feature map column-1  

    int nStart = 0, nEnd = 0;
    nStart=(out_fe_num-2) > 1 ? (out_fe_num-2) : 1 ;
    nEnd=(out_fe_num+2) < num_out ? (out_fe_num+2) : num_out ;

    float sum = 0.0;
    float result = 0.0;  
    for(int i = (nStart-1); i < (nEnd-1); i++){
        sum += pow((input_neurons[i*out_row*out_col + out_fe_row*out_col + out_fe_col]),2);
    }
    result = (input_neurons[out_fe_num*out_row*out_col + out_fe_row*out_col + out_fe_col]) / (pow( 1 + ((alpha/local_size) * sum),beta));
    sum = 0.0;
    output_neurons[out_fe_num*out_row*out_col + out_fe_row*out_col + out_fe_col] = result;
}

//fully connected
__global__ void fc(float *bias, float *input_neurons, float *input_weights, float *output_neurons, int output, int input, bool relu)
{
    //number of output feature map
    int out_fe_num = blockIdx.x;    //0 ~ output feature map 개수-1   
    int weight = out_fe_num * input;

    //Compute fully connected
    float result = 0.0;
    for(int i = 0; i < input; i++){
        result += input_neurons[i] * input_weights[weight+i];
    }
    result += bias[out_fe_num];

    //ReLU
    if(relu == true){
        if(result < 0)
            result = 0;
    }
    
    output_neurons[out_fe_num] = result;
    result = 0.0;
}