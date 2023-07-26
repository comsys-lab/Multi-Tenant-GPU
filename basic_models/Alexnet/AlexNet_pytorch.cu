#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "AlexNet_kernel.cu"

/* Define Feature Map Sizes and Filter Sizes of AlexNet */
#define INPUT_SIZE 224*224*3

#define L1_OUT 64
#define L2_OUT 192
#define L3_OUT 384
#define L4_OUT 256
#define L5_OUT 256
#define L6_OUT 4096
#define L7_OUT 4096
#define L8_OUT 1000

#define L1_FILTER_SIZE 11*11*3
#define L2_FILTER_SIZE 5*5*64
#define L3_FILTER_SIZE 3*3*192
#define L4_FILTER_SIZE 3*3*384
#define L5_FILTER_SIZE 3*3*256

#define L1_FM_SIZE 55*55
#define L2_FM_SIZE 27*27
#define L3_FM_SIZE 13*13
#define L4_FM_SIZE 13*13
#define L5_FM_SIZE 13*13
#define L5_AFTER_POOL_FM_SIZE 6*6

extern "C"
void Alexnet();
unsigned g_verbose;
unsigned NUM;

/* Read Parameters from Txt Files. */
void read_parameter(const char *pFileName,float *layer_parameters)
{
	FILE *fp = fopen(pFileName, "rb");
	int count = 0;
	double temp_num;
	printf(" File FOUND : %s\n",pFileName);
	while(fscanf(fp, "%lf", &temp_num) == 1){
		layer_parameters[count] = temp_num;
		count++;
	}
	printf("Final Count : %d\n", count);
	fclose(fp);
}

/* Main Function to Execute AlexNet */
int main(int argc, char** argv)
{
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
		for (i=2; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;
	if (commandline_error || !NUM) {
		printf("Usage: ./AN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}
	Alexnet();
}

/* Function to Read Input Parameters */
void read_input(float *Layer1_Neurons_CPU){
	read_parameter("data_alex_pytorch/input_cat1.txt", Layer1_Neurons_CPU);
}

/* Function to Read All Bias */
void read_bias(float *Layer1_bias_CPU,float *Layer2_bias_CPU,float *Layer3_bias_CPU,float *Layer4_bias_CPU,float *Layer5_bias_CPU,float *Layer6_bias_CPU,float *Layer7_bias_CPU,float *Layer8_bias_CPU){
	read_parameter("data_alex_pytorch/bias1.txt", Layer1_bias_CPU);
	read_parameter("data_alex_pytorch/bias2.txt", Layer2_bias_CPU);
	read_parameter("data_alex_pytorch/bias3.txt", Layer3_bias_CPU);
	read_parameter("data_alex_pytorch/bias4.txt", Layer4_bias_CPU);
	read_parameter("data_alex_pytorch/bias5.txt", Layer5_bias_CPU);
	read_parameter("data_alex_pytorch/bias6.txt", Layer6_bias_CPU);
	read_parameter("data_alex_pytorch/bias7.txt", Layer7_bias_CPU);
	read_parameter("data_alex_pytorch/bias8.txt", Layer8_bias_CPU);
}

/* Function to Read All Weights */
void read_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU){
	read_parameter("data_alex_pytorch/conv1.txt", Layer1_Weights_CPU);
	read_parameter("data_alex_pytorch/conv2.txt", Layer2_Weights_CPU);
	read_parameter("data_alex_pytorch/conv3.txt", Layer3_Weights_CPU);
	read_parameter("data_alex_pytorch/conv4.txt", Layer4_Weights_CPU);
	read_parameter("data_alex_pytorch/conv5.txt", Layer5_Weights_CPU);
	read_parameter("data_alex_pytorch/fc6.txt", Layer6_Weights_CPU);
	read_parameter("data_alex_pytorch/fc7.txt", Layer7_Weights_CPU);
	read_parameter("data_alex_pytorch/fc8.txt", Layer8_Weights_CPU);
}

/* AlexNet */
void Alexnet(){

	int deviceCount;                                                         
	cudaGetDeviceCount(&deviceCount);                
	if (deviceCount == 0) {                                                  
		fprintf(stderr, "There is no device.\n");                            
		exit(EXIT_FAILURE);                                                  
	}                                                                        
	int dev;                                                                 
	for (dev = 0; dev < deviceCount; ++dev) {                                
		cudaDeviceProp deviceProp;                                           
		cudaGetDeviceProperties(&deviceProp, dev);   
		if (deviceProp.major >= 1)                                           
			break;                                                           
	}                                                                        
	if (dev == deviceCount) {                                                
		fprintf(stderr, "There is no device supporting CUDA.\n");            
		exit(EXIT_FAILURE);                                                  
	}                                                                        
	else                                                                     
		cudaSetDevice(dev);

	/* Allocate Memory on CPU and Fill All Parameters */
	float *Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
	read_input(Layer1_Neurons_CPU);
	
	float *Layer1_bias_CPU = (float*) malloc (L1_OUT * sizeof(float));
	float *Layer2_bias_CPU = (float*) malloc (L2_OUT * sizeof(float));
	float *Layer3_bias_CPU = (float*) malloc (L3_OUT * sizeof(float));
	float *Layer4_bias_CPU = (float*) malloc (L4_OUT * sizeof(float));
	float *Layer5_bias_CPU = (float*) malloc (L5_OUT * sizeof(float));
	float *Layer6_bias_CPU = (float*) malloc (L6_OUT * sizeof(float));
	float *Layer7_bias_CPU = (float*) malloc (L7_OUT * sizeof(float));
	float *Layer8_bias_CPU = (float*) malloc (L8_OUT * sizeof(float));
	read_bias(Layer1_bias_CPU, Layer2_bias_CPU, Layer3_bias_CPU, Layer4_bias_CPU, Layer5_bias_CPU, Layer6_bias_CPU, Layer7_bias_CPU, Layer8_bias_CPU);


	float *Layer1_Weights_CPU = (float*) malloc (L1_OUT*L1_FILTER_SIZE * sizeof(float));
	float *Layer2_Weights_CPU = (float*) malloc (L2_OUT*L2_FILTER_SIZE * sizeof(float));
	float *Layer3_Weights_CPU = (float*) malloc (L3_OUT*L3_FILTER_SIZE * sizeof(float));
	float *Layer4_Weights_CPU = (float*) malloc (L4_OUT*L4_FILTER_SIZE * sizeof(float));
	float *Layer5_Weights_CPU = (float*) malloc (L5_OUT*L5_FILTER_SIZE * sizeof(float));
	float *Layer6_Weights_CPU = (float*) malloc (L6_OUT*L5_OUT*L5_AFTER_POOL_FM_SIZE * sizeof(float));
	float *Layer7_Weights_CPU = (float*) malloc (L7_OUT*L6_OUT * sizeof(float));
	float *Layer8_Weights_CPU = (float*) malloc (L8_OUT*L7_OUT * sizeof(float));
    read_weights(Layer1_Weights_CPU, Layer2_Weights_CPU, Layer3_Weights_CPU, Layer4_Weights_CPU, Layer5_Weights_CPU, Layer6_Weights_CPU, Layer7_Weights_CPU, Layer8_Weights_CPU);

	/* Allocate Memory on GPU */
	float *Layer1_bias, *Layer2_bias, *Layer3_bias, *Layer4_bias, *Layer5_bias, *Layer6_bias, *Layer7_bias, *Layer8_bias;
	float *Layer1_Neurons, *Layer2_Neurons, *Layer3_Neurons, *Layer4_Neurons, *Layer5_Neurons, *Layer6_Neurons, *Layer7_Neurons, *Layer8_Neurons; 
	float *Layer1_Weights, *Layer2_Weights, *Layer3_Weights, *Layer4_Weights, *Layer5_Weights, *Layer6_Weights, *Layer7_Weights, *Layer8_Weights;

	float *Layer1_pool, *Layer2_pool, *Layer5_pool;
	float *Layer1_norm, *Layer2_norm; 
	float *Result_Neurons;

	// Copy from CPU to GPU

	//* First Layer *//
	cudaMalloc((void**) &Layer1_bias, L1_OUT * sizeof(float)); //64
	cudaMalloc((void**) &Layer1_Neurons, INPUT_SIZE * sizeof(float)); //224*224*3
	cudaMalloc((void**) &Layer1_Weights, (L1_OUT*L1_FILTER_SIZE) * sizeof(float)); //64*11*11*3 = 23232
	cudaMalloc((void**) &Layer1_norm, (L1_OUT*L1_FM_SIZE) * sizeof(float)); //64*55*55

	cudaMemcpy(Layer1_Neurons, Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_bias, Layer1_bias_CPU, L1_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Weights, Layer1_Weights_CPU, (L1_OUT*L1_FILTER_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Layer1_Block(64,5,5);
	dim3 Layer1_Thread(11,11);
	cudaEvent_t start1, end1;
	float inference_time1;
	cudaEventCreate(&start1);
	cudaEventCreate(&end1);
	cudaEventRecord(start1,0);
	first_jjb<<<Layer1_Block,Layer1_Thread>>>(Layer1_bias,Layer1_Neurons,Layer1_Weights,Layer1_norm,224,55,4,2,11,3);
	cudaEventRecord(end1, 0);
	cudaEventSynchronize(end1);
	cudaEventElapsedTime(&inference_time1, start1, end1);
	printf("Elapsed time: %f ms\n", inference_time1);
	cudaEventDestroy(start1);
	cudaEventDestroy(end1);
	cudaMalloc((void**) &Layer1_pool, (L1_OUT*L1_FM_SIZE) * sizeof(float)); //64*55*55

	/* Normalization of First Layer */
	dim3 Norm11_Block(64,5,5);
	dim3 Norm11_Thread(11,11);
	norm_jjb<<<Norm11_Block,Norm11_Thread>>>(Layer1_norm,Layer1_pool,0.0001,0.75,5,55);

	cudaMalloc((void**) &Layer2_Neurons, (L1_OUT*L2_FM_SIZE) * sizeof(float)); //64*27*27

	/* Maxpooling of First Layer */
	dim3 Pool1_Block(64,1,1);
	dim3 Pool1_Thread(27,27);
	max_jjb<<<Pool1_Block,Pool1_Thread>>>(Layer1_pool,Layer2_Neurons,55,27,2,0,3);

	//* Second Layer *//
	cudaMalloc((void**) &Layer2_bias, L2_OUT * sizeof(float)); //192
	cudaMalloc((void**) &Layer2_Weights, (L2_OUT*L2_FILTER_SIZE) * sizeof(float)); //192*5*5*64 = 307200
	cudaMalloc((void**) &Layer2_norm, (L2_OUT*L2_FM_SIZE) * sizeof(float)); //192*27*27

	cudaMemcpy(Layer2_bias, Layer2_bias_CPU, L2_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer2_Weights, Layer2_Weights_CPU, (L2_OUT*L2_FILTER_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	/* Convolution of Second Layer */
	dim3 Layer2_Block(192,1,1);
	dim3 Layer2_Thread(27,27); 
	// GPU 1
	conv_jjb<<<Layer2_Block,Layer2_Thread>>>(Layer2_bias,Layer2_Neurons,Layer2_Weights,Layer2_norm,27,27,1,2,5,64);
	
	cudaMalloc((void**) &Layer2_pool, (L2_OUT*L2_FM_SIZE) * sizeof(float)); //192*27*27

	/* Normalization of Second Layer */
	dim3 Norm2_Block(192,1,1);
	dim3 Norm2_Thread(27,27);
	norm_jjb<<<Norm2_Block,Norm2_Thread>>>(Layer2_norm,Layer2_pool,0.0001,0.75,5,27);
	
	cudaMalloc((void**) &Layer3_Neurons, (L2_OUT*L3_FM_SIZE) * sizeof(float)); //192*13*13

	/* Maxpooling of Second Layer */
	dim3 Pool2_Block(192,1,1);
	dim3 Pool2_Thread(13,13);
	max_jjb<<<Pool2_Block,Pool2_Thread>>>(Layer2_pool, Layer3_Neurons,27,13,2,0,3);

	//* Third Layer *//
	cudaMalloc((void**) &Layer3_bias, L3_OUT * sizeof(float)); //384
	cudaMalloc((void**) &Layer3_Weights, (L3_OUT*L3_FILTER_SIZE) * sizeof(float)); //384*3*3*192 = 663552
	cudaMalloc((void**) &Layer4_Neurons, (L3_OUT*L4_FM_SIZE) * sizeof(float)); //384*13*13

	cudaMemcpy(Layer3_bias, Layer3_bias_CPU, L3_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer3_Weights, Layer3_Weights_CPU, (L3_OUT*L3_FILTER_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	/* Convolution of Third Layer */
	dim3 Layer3_Block(384,1,1);
	dim3 Layer3_Thread(13,13); 
	conv_jjb<<<Layer3_Block,Layer3_Thread>>>(Layer3_bias, Layer3_Neurons, Layer3_Weights, Layer4_Neurons,13,13,1,1,3,192);

	//* Fourth Layer *//
	cudaMalloc((void**) &Layer4_bias, L4_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer4_Weights, (L4_OUT*L4_FILTER_SIZE) * sizeof(float)); //256*3*3*384 = 884736
	cudaMalloc((void**) &Layer5_Neurons, (L4_OUT*L5_FM_SIZE) * sizeof(float)); //256*13*13

	cudaMemcpy(Layer4_bias, Layer4_bias_CPU, L4_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer4_Weights, Layer4_Weights_CPU, (L4_OUT*L4_FILTER_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	/* Convolution of Fourth Layer */
	dim3 Layer4_Block(256,1,1);
	dim3 Layer4_Thread(13,13); 
	// GPU 1
	conv_jjb<<<Layer4_Block,Layer4_Thread>>>(Layer4_bias, Layer4_Neurons, Layer4_Weights, Layer5_Neurons,13,13,1,1,3,384);

	//* Fifth Layer *//
	cudaMalloc((void**) &Layer5_bias, L5_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer5_Weights, (L5_OUT*L5_FILTER_SIZE) * sizeof(float)); //256*3*3*256 = 442368
	cudaMalloc((void**) &Layer5_pool, (L5_OUT*L5_FM_SIZE) * sizeof(float)); //256*13*13

	cudaMemcpy(Layer5_bias, Layer5_bias_CPU, L5_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer5_Weights, Layer5_Weights_CPU, (L5_OUT*L5_FILTER_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	/* Convolution of Fifth Layer */
	dim3 Layer5_Block(256,1,1);
	dim3 Layer5_Thread(13,13); 
	// GPU 1
	conv_jjb<<<Layer5_Block,Layer5_Thread>>>(Layer5_bias, Layer5_Neurons, Layer5_Weights, Layer5_pool,13,13,1,1,3,256);

	cudaMalloc((void**) &Layer6_Neurons, (L5_OUT*L5_AFTER_POOL_FM_SIZE) * sizeof(float)); //256*6*6

	/* Maxpooling of Fifth Layer */
	dim3 Pool3_Block(256,1,1);
	dim3 Pool3_Thread(6,6);
	max_jjb<<<Pool3_Block,Pool3_Thread>>>(Layer5_pool, Layer6_Neurons,13,6,2,0,3);

	//* Sixth Layer *//
	cudaMalloc((void**) &Layer6_bias, L6_OUT * sizeof(float)); //4096
	cudaMalloc((void**) &Layer6_Weights, (L6_OUT*L5_OUT*L5_AFTER_POOL_FM_SIZE) * sizeof(float)); //4096*256*6*6 = 37748736
	cudaMalloc((void**) &Layer7_Neurons, L6_OUT * sizeof(float)); //4096

	cudaMemcpy(Layer6_bias, Layer6_bias_CPU, L6_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer6_Weights, Layer6_Weights_CPU, (L6_OUT*L5_OUT*L5_AFTER_POOL_FM_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	/* First Fully Connected Layer */
	dim3 Layer6_Block(4096,1,1);
	dim3 Layer6_Thread(1,1);
	fc_jjb<<<Layer6_Block,Layer6_Thread>>>(Layer6_bias, Layer6_Neurons, Layer6_Weights, Layer7_Neurons, (6*6*256), true);

	//* Seventh Layer *//
	cudaMalloc((void**) &Layer7_bias, L7_OUT * sizeof(float)); //4096
	cudaMalloc((void**) &Layer7_Weights, (L7_OUT*L6_OUT) * sizeof(float)); //4096*4096 = 16777216
	cudaMalloc((void**) &Layer8_Neurons, L7_OUT * sizeof(float)); //4096

	cudaMemcpy(Layer7_bias, Layer7_bias_CPU, L7_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer7_Weights, Layer7_Weights_CPU, (L7_OUT*L6_OUT) * sizeof(float), cudaMemcpyHostToDevice);

	/* Second Fully Connected Layer */
	dim3 Layer7_Block(4096,1,1);
	dim3 Layer7_Thread(1,1);
	fc_jjb<<<Layer7_Block,Layer7_Thread>>>(Layer7_bias, Layer7_Neurons, Layer7_Weights, Layer8_Neurons, 4096, true);

	//* Eighth Layer *//
	cudaMalloc((void**) &Layer8_bias, L8_OUT * sizeof(float)); //1000
	cudaMalloc((void**) &Layer8_Weights, (L8_OUT*L7_OUT) * sizeof(float)); //1000*4096 = 4096000
	cudaMalloc((void**) &Result_Neurons, L8_OUT * sizeof(float)); //1000

	cudaMemcpy(Layer8_bias, Layer8_bias_CPU, L8_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer8_Weights, Layer8_Weights_CPU, (L8_OUT*L7_OUT) * sizeof(float), cudaMemcpyHostToDevice);

	/* Third Fully Connected Layer */
	dim3 Layer8_Block(1000,1,1);
	dim3 Layer8_Thread(1,1);
	fc_jjb<<<Layer8_Block,Layer8_Thread>>>(Layer8_bias, Layer8_Neurons, Layer8_Weights, Result_Neurons, 4096, false);

	float *Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
	cudaMemcpy(Result_Neurons_CPU, Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);

	float max1 = 0.0, max2 = 0.0, max3 = 0.0;
	int index1 = 0, index2 = 0, index3 = 0; 
	for(int i = 0; i < 1000; i++){
		if(max1 < Result_Neurons_CPU[i]){
			max1 = Result_Neurons_CPU[i];	
			index1 = i;
		}
		else if(max2 < Result_Neurons_CPU[i] && Result_Neurons_CPU[i] != max1){
			max2 = Result_Neurons_CPU[i];	
			index2 = i;
		}
		else if(max3 < Result_Neurons_CPU[i] && Result_Neurons_CPU[i] != max2){
			max3 = Result_Neurons_CPU[i];	
			index3 = i;
		}
	}
	
	int line_count1 = 0, line_count2 = 0, line_count3 = 0;
	char buffer[1000];
	FILE *list1 = fopen("imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list1) != NULL){
		line_count1++;
		if(line_count1 == (index1+1)){
			printf("\nClass ID: %d\nClass Name: %sProbability: %f\n", index1, buffer, max1);
			break;
		}
	}
	fclose(list1);

	FILE *list2 = fopen("imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list2) != NULL){
		line_count2++;
		if(line_count2 == (index2+1)){
			printf("\nClass ID: %d\nClass Name: %sProbability: %f\n", index2, buffer, max2);
			break;
		}
	}
	fclose(list2);

	FILE *list3 = fopen("imagenet1000_clsidx_to_labels.txt","rt");
	while(fgets(buffer, 1000, list3) != NULL){
		line_count3++;
		if(line_count3 == (index3+1)){
			printf("\nClass ID: %d\nClass Name: %sProbability: %f\n\n", index3, buffer, max3);
			break;
		}
	}
	fclose(list3);

	//printf("INDEX = %d\n",index);
	//printf("%f\n",Result_Neurons_CPU[index]);

	cudaFree(Layer1_bias);
	cudaFree(Layer2_bias);
	cudaFree(Layer3_bias);
	cudaFree(Layer4_bias);
	cudaFree(Layer5_bias);
	cudaFree(Layer6_bias);
	cudaFree(Layer7_bias);
	cudaFree(Layer8_bias);

	cudaFree(Layer1_Neurons);
	cudaFree(Layer2_Neurons);
	cudaFree(Layer3_Neurons);
	cudaFree(Layer4_Neurons);
	cudaFree(Layer5_Neurons);
	cudaFree(Layer6_Neurons);
	cudaFree(Layer7_Neurons);
	cudaFree(Layer8_Neurons);

	cudaFree(Layer1_Weights);
	cudaFree(Layer2_Weights);
	cudaFree(Layer3_Weights);
	cudaFree(Layer4_Weights);
	cudaFree(Layer5_Weights);
	cudaFree(Layer6_Weights);
	cudaFree(Layer7_Weights);
	cudaFree(Layer8_Weights);

	cudaFree(Layer1_pool);
	cudaFree(Layer2_pool);
	cudaFree(Layer5_pool);

	cudaFree(Layer1_norm);
	cudaFree(Layer2_norm);

	cudaFree(Result_Neurons);


	free(Layer1_Neurons_CPU);

	free(Layer1_bias_CPU);
	free(Layer2_bias_CPU);
	free(Layer3_bias_CPU);
	free(Layer4_bias_CPU);
	free(Layer5_bias_CPU);
	free(Layer6_bias_CPU);
	free(Layer7_bias_CPU);
	free(Layer8_bias_CPU);

    free(Layer1_Weights_CPU);
    free(Layer2_Weights_CPU);
    free(Layer3_Weights_CPU);
    free(Layer4_Weights_CPU);
    free(Layer5_Weights_CPU);
    free(Layer6_Weights_CPU);
    free(Layer7_Weights_CPU);
    free(Layer8_Weights_CPU);

	exit(0);
}