#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Vgg16_kernel.cu"

/* Define Feature Map Sizes and Filter Sizes of AlexNet */
#define INPUT_SIZE 224*224*3 //150,528

/* Layer 1,2 */
#define BLOCK1_OUT 64
#define BLOCK1_FM_SIZE 224*224
/* Layer 3,4 */
#define BLOCK2_OUT 128
#define BLOCK2_FM_SIZE 112*112
/* Layer 5,6,7 */
#define BLOCK3_OUT 256
#define BLOCK3_FM_SIZE 56*56
/* Layer 8,9,10 */
#define BLOCK4_OUT 512
#define BLOCK4_FM_SIZE 28*28
/* Layer 11,12,13 */
#define BLOCK5_OUT 512
#define BLOCK5_FM_SIZE 14*14

#define L14_OUT 4096
#define L15_OUT 4096
#define L16_OUT 1000

#define FILTER_SIZE 3*3

extern "C"
void Vgg16();
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

/* Main Function to Execute Vgg16 */
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
		printf("Usage: ./VN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}
	Vgg16();
}

/* Function to Read Input Parameters */
void read_input(float *Layer1_Neurons_CPU){
    read_parameter("data_pytorch/input_cat.txt", Layer1_Neurons_CPU);
}

/* Function to Read All Bias */
void read_bias(float *Layer1_bias_CPU,float *Layer2_bias_CPU,float *Layer3_bias_CPU,float *Layer4_bias_CPU,
                float *Layer5_bias_CPU,float *Layer6_bias_CPU,float *Layer7_bias_CPU,float *Layer8_bias_CPU,
                float *Layer9_bias_CPU,float *Layer10_bias_CPU,float *Layer11_bias_CPU,float *Layer12_bias_CPU,
                float *Layer13_bias_CPU,float *Layer14_bias_CPU,float *Layer15_bias_CPU,float *Layer16_bias_CPU){
	read_parameter("data_pytorch/bias1.txt", Layer1_bias_CPU);
	read_parameter("data_pytorch/bias2.txt", Layer2_bias_CPU);
	read_parameter("data_pytorch/bias3.txt", Layer3_bias_CPU);
	read_parameter("data_pytorch/bias4.txt", Layer4_bias_CPU);
	read_parameter("data_pytorch/bias5.txt", Layer5_bias_CPU);
	read_parameter("data_pytorch/bias6.txt", Layer6_bias_CPU);
	read_parameter("data_pytorch/bias7.txt", Layer7_bias_CPU);
	read_parameter("data_pytorch/bias8.txt", Layer8_bias_CPU);
    read_parameter("data_pytorch/bias9.txt", Layer9_bias_CPU);
	read_parameter("data_pytorch/bias10.txt", Layer10_bias_CPU);
	read_parameter("data_pytorch/bias11.txt", Layer11_bias_CPU);
	read_parameter("data_pytorch/bias12.txt", Layer12_bias_CPU);
	read_parameter("data_pytorch/bias13.txt", Layer13_bias_CPU);
	read_parameter("data_pytorch/bias14.txt", Layer14_bias_CPU);
	read_parameter("data_pytorch/bias15.txt", Layer15_bias_CPU);
	read_parameter("data_pytorch/bias16.txt", Layer16_bias_CPU);
}

/* Function to Read All Weights */
void read_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,
                    float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU,
                    float *Layer9_Weights_CPU,float *Layer10_Weights_CPU,float *Layer11_Weights_CPU,float *Layer12_Weights_CPU,
                    float *Layer13_Weights_CPU,float *Layer14_Weights_CPU,float *Layer15_Weights_CPU,float *Layer16_Weights_CPU){
	read_parameter("data_pytorch/conv1.txt", Layer1_Weights_CPU);
	read_parameter("data_pytorch/conv2.txt", Layer2_Weights_CPU);
	read_parameter("data_pytorch/conv3.txt", Layer3_Weights_CPU);
	read_parameter("data_pytorch/conv4.txt", Layer4_Weights_CPU);
	read_parameter("data_pytorch/conv5.txt", Layer5_Weights_CPU);
	read_parameter("data_pytorch/conv6.txt", Layer6_Weights_CPU);
	read_parameter("data_pytorch/conv7.txt", Layer7_Weights_CPU);
	read_parameter("data_pytorch/conv8.txt", Layer8_Weights_CPU);
 	read_parameter("data_pytorch/conv9.txt", Layer9_Weights_CPU);
	read_parameter("data_pytorch/conv10.txt", Layer10_Weights_CPU);
	read_parameter("data_pytorch/conv11.txt", Layer11_Weights_CPU);
	read_parameter("data_pytorch/conv12.txt", Layer12_Weights_CPU);
	read_parameter("data_pytorch/conv13.txt", Layer13_Weights_CPU);
	read_parameter("data_pytorch/fc14.txt", Layer14_Weights_CPU);
	read_parameter("data_pytorch/fc15.txt", Layer15_Weights_CPU);
	read_parameter("data_pytorch/fc16.txt", Layer16_Weights_CPU);
}

/* Vgg16 */
void Vgg16(){

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
	
	float *Layer1_bias_CPU = (float*) malloc (BLOCK1_OUT * sizeof(float)); //64
	float *Layer2_bias_CPU = (float*) malloc (BLOCK1_OUT * sizeof(float)); //64
	float *Layer3_bias_CPU = (float*) malloc (BLOCK2_OUT * sizeof(float)); //128
	float *Layer4_bias_CPU = (float*) malloc (BLOCK2_OUT * sizeof(float)); //128
	float *Layer5_bias_CPU = (float*) malloc (BLOCK3_OUT * sizeof(float)); //256
	float *Layer6_bias_CPU = (float*) malloc (BLOCK3_OUT * sizeof(float)); //256
	float *Layer7_bias_CPU = (float*) malloc (BLOCK3_OUT * sizeof(float)); //256
	float *Layer8_bias_CPU = (float*) malloc (BLOCK4_OUT * sizeof(float)); //512
    float *Layer9_bias_CPU = (float*) malloc (BLOCK4_OUT * sizeof(float)); //512
	float *Layer10_bias_CPU = (float*) malloc (BLOCK4_OUT * sizeof(float)); //512
	float *Layer11_bias_CPU = (float*) malloc (BLOCK5_OUT * sizeof(float)); //512
	float *Layer12_bias_CPU = (float*) malloc (BLOCK5_OUT * sizeof(float)); //512
	float *Layer13_bias_CPU = (float*) malloc (BLOCK5_OUT * sizeof(float)); //512
	float *Layer14_bias_CPU = (float*) malloc (L14_OUT * sizeof(float)); //4096
	float *Layer15_bias_CPU = (float*) malloc (L15_OUT * sizeof(float)); //4096
	float *Layer16_bias_CPU = (float*) malloc (L16_OUT * sizeof(float)); //1000
	read_bias(Layer1_bias_CPU, Layer2_bias_CPU, Layer3_bias_CPU, Layer4_bias_CPU,
                Layer5_bias_CPU, Layer6_bias_CPU, Layer7_bias_CPU, Layer8_bias_CPU,
                Layer9_bias_CPU, Layer10_bias_CPU, Layer11_bias_CPU, Layer12_bias_CPU,
                Layer13_bias_CPU, Layer14_bias_CPU, Layer15_bias_CPU, Layer16_bias_CPU);

	float *Layer1_Weights_CPU = (float*) malloc (BLOCK1_OUT*(FILTER_SIZE*3) * sizeof(float)); //64*3*3*3 = 1,728
	float *Layer2_Weights_CPU = (float*) malloc (BLOCK1_OUT*(FILTER_SIZE*64) * sizeof(float)); //64*3*3*64 = 36,864
	float *Layer3_Weights_CPU = (float*) malloc (BLOCK2_OUT*(FILTER_SIZE*64) * sizeof(float)); //128*3*3*64 = 73,728
	float *Layer4_Weights_CPU = (float*) malloc (BLOCK2_OUT*(FILTER_SIZE*128) * sizeof(float)); //128*3*3*128 = 147,456
	float *Layer5_Weights_CPU = (float*) malloc (BLOCK3_OUT*(FILTER_SIZE*128) * sizeof(float)); //256*3*3*128 = 294,912
	float *Layer6_Weights_CPU = (float*) malloc (BLOCK3_OUT*(FILTER_SIZE*256) * sizeof(float)); //256*3*3*256 = 589,824
	float *Layer7_Weights_CPU = (float*) malloc (BLOCK3_OUT*(FILTER_SIZE*256) * sizeof(float)); //256*3*3*256 = 589,824
	float *Layer8_Weights_CPU = (float*) malloc (BLOCK4_OUT*(FILTER_SIZE*256) * sizeof(float)); //512*3*3*256 = 1,179,648
    float *Layer9_Weights_CPU = (float*) malloc (BLOCK4_OUT*(FILTER_SIZE*512) * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Layer10_Weights_CPU = (float*) malloc (BLOCK4_OUT*(FILTER_SIZE*512) * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Layer11_Weights_CPU = (float*) malloc (BLOCK5_OUT*(FILTER_SIZE*512) * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Layer12_Weights_CPU = (float*) malloc (BLOCK5_OUT*(FILTER_SIZE*512) * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Layer13_Weights_CPU = (float*) malloc (BLOCK5_OUT*(FILTER_SIZE*512) * sizeof(float)); //512*3*3*512 = 2,359,296
	float *Layer14_Weights_CPU = (float*) malloc (L14_OUT*BLOCK5_OUT*(7*7) * sizeof(float)); //4096*512*7*7 = 102,760,448
	float *Layer15_Weights_CPU = (float*) malloc (L15_OUT*L14_OUT * sizeof(float)); //4096*4096 = 16,777,216
	float *Layer16_Weights_CPU = (float*) malloc (L16_OUT*L15_OUT * sizeof(float)); //1000*4096 = 4,096,000
    read_weights(Layer1_Weights_CPU, Layer2_Weights_CPU, Layer3_Weights_CPU, Layer4_Weights_CPU,
                    Layer5_Weights_CPU, Layer6_Weights_CPU, Layer7_Weights_CPU, Layer8_Weights_CPU,
                    Layer9_Weights_CPU, Layer10_Weights_CPU, Layer11_Weights_CPU, Layer12_Weights_CPU,
                    Layer13_Weights_CPU, Layer14_Weights_CPU, Layer15_Weights_CPU, Layer16_Weights_CPU);
    
    /* Allocate Memory on GPU */
	float *Layer1_bias, *Layer2_bias, *Layer3_bias, *Layer4_bias, *Layer5_bias, *Layer6_bias, *Layer7_bias, *Layer8_bias;
    float *Layer9_bias, *Layer10_bias, *Layer11_bias, *Layer12_bias, *Layer13_bias, *Layer14_bias, *Layer15_bias, *Layer16_bias;

	float *Layer1_Neurons, *Layer2_Neurons, *Layer3_Neurons, *Layer4_Neurons, *Layer5_Neurons, *Layer6_Neurons, *Layer7_Neurons, *Layer8_Neurons;
    float *Layer9_Neurons, *Layer10_Neurons, *Layer11_Neurons, *Layer12_Neurons, *Layer13_Neurons, *Layer14_Neurons, *Layer15_Neurons, *Layer16_Neurons;

	float *Layer1_Weights, *Layer2_Weights, *Layer3_Weights, *Layer4_Weights, *Layer5_Weights, *Layer6_Weights, *Layer7_Weights, *Layer8_Weights;
    float *Layer9_Weights, *Layer10_Weights, *Layer11_Weights, *Layer12_Weights, *Layer13_Weights, *Layer14_Weights, *Layer15_Weights, *Layer16_Weights;

    float *Layer2_Pool, *Layer4_Pool, *Layer7_Pool, *Layer10_Pool, *Layer13_Pool;

	float *Result_Neurons;

    /* 1st Layer */
    cudaMalloc((void**) &Layer1_bias, BLOCK1_OUT * sizeof(float)); //64
	cudaMalloc((void**) &Layer1_Neurons, INPUT_SIZE * sizeof(float)); //224*224*3
	cudaMalloc((void**) &Layer1_Weights, (BLOCK1_OUT*(FILTER_SIZE*3)) * sizeof(float)); //64*3*3*3 = 1728
	cudaMalloc((void**) &Layer2_Neurons, (BLOCK1_OUT*BLOCK1_FM_SIZE) * sizeof(float)); //64*224*224

	cudaMemcpy(Layer1_Neurons, Layer1_Neurons_CPU, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_bias, Layer1_bias_CPU, BLOCK1_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Weights, Layer1_Weights_CPU, (BLOCK1_OUT*(FILTER_SIZE*3)) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Block_Thread(7,7);

    dim3 Block1_Block(64,32,32);
	cudaEvent_t start1, end1;
	float inference_time1;
	cudaEventCreate(&start1);
	cudaEventCreate(&end1);
	cudaEventRecord(start1,0);
	first_jjb<<<Block1_Block,Block_Thread>>>(Layer1_bias,Layer1_Neurons,Layer1_Weights,Layer2_Neurons,224,224,1,1,3,3);
	cudaEventRecord(end1, 0);
	cudaEventSynchronize(end1);
	cudaEventElapsedTime(&inference_time1, start1, end1);
	printf("Elapsed time: %f ms\n", inference_time1);
	cudaEventDestroy(start1);
	cudaEventDestroy(end1);
	float *Layer2_Neurons_CPU = (float *) malloc ((64*224*224) * sizeof(float));
	cudaMemcpy(Layer2_Neurons_CPU, Layer2_Neurons, (64*224*224) * sizeof(float), cudaMemcpyDeviceToHost);
	
    /* 2nd Layer */
    cudaMalloc((void**) &Layer2_bias, BLOCK1_OUT * sizeof(float)); //64
	cudaMalloc((void**) &Layer2_Weights, (BLOCK1_OUT*(FILTER_SIZE*64)) * sizeof(float)); //64*3*3*64 = 36864
    cudaMalloc((void**) &Layer2_Pool, (BLOCK1_OUT*BLOCK1_FM_SIZE) * sizeof(float)); //64*224*224

    cudaMemcpy(Layer2_bias, Layer2_bias_CPU, BLOCK1_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer2_Weights, Layer2_Weights_CPU, (BLOCK1_OUT*(FILTER_SIZE*64)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block1_Block,Block_Thread>>>(Layer2_bias,Layer2_Neurons,Layer2_Weights,Layer2_Pool,224,224,1,1,3,64);

    cudaMalloc((void**) &Layer3_Neurons, (BLOCK1_OUT*BLOCK2_FM_SIZE) * sizeof(float)); //64*112*112

	dim3 Block1_Pool_Block(64,16,16);
    max_jjb<<<Block1_Pool_Block,Block_Thread>>>(Layer2_Pool,Layer3_Neurons,224,112,2,0,2);

    /* 3rd Layer */
    cudaMalloc((void**) &Layer3_bias, BLOCK2_OUT * sizeof(float)); //128
	cudaMalloc((void**) &Layer3_Weights, (BLOCK2_OUT*(FILTER_SIZE*64)) * sizeof(float)); //128*3*3*64 = 73728
	cudaMalloc((void**) &Layer4_Neurons, (BLOCK2_OUT*BLOCK2_FM_SIZE) * sizeof(float)); //128*112*112

	cudaMemcpy(Layer3_bias, Layer3_bias_CPU, BLOCK2_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer3_Weights, Layer3_Weights_CPU, (BLOCK2_OUT*(FILTER_SIZE*64)) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Block2_Block(128,16,16);
	conv_jjb<<<Block2_Block,Block_Thread>>>(Layer3_bias,Layer3_Neurons,Layer3_Weights,Layer4_Neurons,112,112,1,1,3,64);

	/* 4th Layer */
    cudaMalloc((void**) &Layer4_bias, BLOCK2_OUT * sizeof(float)); //128
	cudaMalloc((void**) &Layer4_Weights, (BLOCK2_OUT*(FILTER_SIZE*128)) * sizeof(float)); //128*3*3*128 = 147456
    cudaMalloc((void**) &Layer4_Pool, (BLOCK2_OUT*BLOCK2_FM_SIZE) * sizeof(float)); //128*112*112

    cudaMemcpy(Layer4_bias, Layer4_bias_CPU, BLOCK1_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer4_Weights, Layer4_Weights_CPU, (BLOCK1_OUT*(FILTER_SIZE*128)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block2_Block,Block_Thread>>>(Layer4_bias,Layer4_Neurons,Layer4_Weights,Layer4_Pool,112,112,1,1,3,128);

	cudaMalloc((void**) &Layer5_Neurons, (BLOCK2_OUT*BLOCK3_FM_SIZE) * sizeof(float)); //128*56*56

	dim3 Block2_Pool_Block(128,8,8);
    max_jjb<<<Block2_Pool_Block,Block_Thread>>>(Layer4_Pool,Layer5_Neurons,112,56,2,0,2);

    /* 5th Layer */
    cudaMalloc((void**) &Layer5_bias, BLOCK3_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer5_Weights, (BLOCK3_OUT*(FILTER_SIZE*128)) * sizeof(float)); //256*3*3*128 = 294912
	cudaMalloc((void**) &Layer6_Neurons, (BLOCK3_OUT*BLOCK3_FM_SIZE) * sizeof(float)); //256*56*56

	cudaMemcpy(Layer5_bias, Layer5_bias_CPU, BLOCK3_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer5_Weights, Layer5_Weights_CPU, (BLOCK3_OUT*(FILTER_SIZE*128)) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Block3_Block(256,8,8);
	conv_jjb<<<Block3_Block,Block_Thread>>>(Layer5_bias,Layer5_Neurons,Layer5_Weights,Layer6_Neurons,56,56,1,1,3,128);

    /* 6th Layer */
    cudaMalloc((void**) &Layer6_bias, BLOCK3_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer6_Weights, (BLOCK3_OUT*(FILTER_SIZE*256)) * sizeof(float)); //256*3*3*256 = 589824
    cudaMalloc((void**) &Layer7_Neurons, (BLOCK3_OUT*BLOCK3_FM_SIZE) * sizeof(float)); //256*56*56

    cudaMemcpy(Layer6_bias, Layer6_bias_CPU, BLOCK3_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer6_Weights, Layer6_Weights_CPU, (BLOCK3_OUT*(FILTER_SIZE*256)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block3_Block,Block_Thread>>>(Layer6_bias,Layer6_Neurons,Layer6_Weights,Layer7_Neurons,56,56,1,1,3,256);

    /* 7th Layer */
    cudaMalloc((void**) &Layer7_bias, BLOCK3_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer7_Weights, (BLOCK3_OUT*(FILTER_SIZE*256)) * sizeof(float)); //256*3*3*256 = 589824
    cudaMalloc((void**) &Layer7_Pool, (BLOCK3_OUT*BLOCK3_FM_SIZE) * sizeof(float)); //256*56*56

    cudaMemcpy(Layer7_bias, Layer7_bias_CPU, BLOCK3_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer7_Weights, Layer7_Weights_CPU, (BLOCK3_OUT*(FILTER_SIZE*256)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block3_Block,Block_Thread>>>(Layer7_bias,Layer7_Neurons,Layer7_Weights,Layer7_Pool,56,56,1,1,3,256);

    cudaMalloc((void**) &Layer8_Neurons, (BLOCK3_OUT*BLOCK4_FM_SIZE) * sizeof(float)); //256*28*28

	dim3 Block3_Pool_Block(256,4,4);
    max_jjb<<<Block3_Pool_Block,Block_Thread>>>(Layer7_Pool,Layer8_Neurons,56,28,2,0,2);

    /* 8th Layer */
    cudaMalloc((void**) &Layer8_bias, BLOCK4_OUT * sizeof(float)); //512
	cudaMalloc((void**) &Layer8_Weights, (BLOCK4_OUT*(FILTER_SIZE*256)) * sizeof(float)); //512*3*3*256 = 1179648
	cudaMalloc((void**) &Layer9_Neurons, (BLOCK4_OUT*BLOCK4_FM_SIZE) * sizeof(float)); //512*28*28

	cudaMemcpy(Layer8_bias, Layer8_bias_CPU, BLOCK4_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer8_Weights, Layer8_Weights_CPU, (BLOCK4_OUT*(FILTER_SIZE*256)) * sizeof(float), cudaMemcpyHostToDevice);
	
    dim3 Block4_Block(512,4,4);
	conv_jjb<<<Block4_Block,Block_Thread>>>(Layer8_bias,Layer8_Neurons,Layer8_Weights,Layer9_Neurons,28,28,1,1,3,256);

    /* 9th Layer */
    cudaMalloc((void**) &Layer9_bias, BLOCK4_OUT * sizeof(float)); //512
	cudaMalloc((void**) &Layer9_Weights, (BLOCK4_OUT*(FILTER_SIZE*512)) * sizeof(float)); //512*3*3*512 = 2359296
    cudaMalloc((void**) &Layer10_Neurons, (BLOCK4_OUT*BLOCK4_FM_SIZE) * sizeof(float)); //512*28*28

    cudaMemcpy(Layer9_bias, Layer9_bias_CPU, BLOCK4_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer9_Weights, Layer9_Weights_CPU, (BLOCK4_OUT*(FILTER_SIZE*512)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(Layer9_bias,Layer9_Neurons,Layer9_Weights,Layer10_Neurons,28,28,1,1,3,512);

    /* 10th Layer */
    cudaMalloc((void**) &Layer10_bias, BLOCK4_OUT * sizeof(float)); //512
	cudaMalloc((void**) &Layer10_Weights, (BLOCK4_OUT*(FILTER_SIZE*512)) * sizeof(float)); //512*3*3*512 = 2359296
    cudaMalloc((void**) &Layer10_Pool, (BLOCK4_OUT*BLOCK4_FM_SIZE) * sizeof(float)); //512*28*28

    cudaMemcpy(Layer10_bias, Layer10_bias_CPU, BLOCK4_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer10_Weights, Layer10_Weights_CPU, (BLOCK4_OUT*(FILTER_SIZE*512)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(Layer10_bias,Layer10_Neurons,Layer10_Weights,Layer10_Pool,28,28,1,1,3,512);

    cudaMalloc((void**) &Layer11_Neurons, (BLOCK4_OUT*BLOCK5_FM_SIZE) * sizeof(float)); //512*14*14

	dim3 Block4_Pool_Block(512,2,2);
    max_jjb<<<Block4_Pool_Block,Block_Thread>>>(Layer10_Pool,Layer11_Neurons,28,14,2,0,2);

    /* 11th Layer */
    cudaMalloc((void**) &Layer11_bias, BLOCK5_OUT * sizeof(float)); //512
	cudaMalloc((void**) &Layer11_Weights, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float)); //512*3*3*512 = 2359296
	cudaMalloc((void**) &Layer12_Neurons, (BLOCK5_OUT*BLOCK5_FM_SIZE) * sizeof(float)); //512*14*14

	cudaMemcpy(Layer11_bias, Layer11_bias_CPU, BLOCK5_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer11_Weights, Layer11_Weights_CPU, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Block5_Block(512,2,2);
	conv_jjb<<<Block5_Block,Block_Thread>>>(Layer11_bias,Layer11_Neurons,Layer11_Weights,Layer12_Neurons,14,14,1,1,3,512);

    /* 12th Layer */
    cudaMalloc((void**) &Layer12_bias, BLOCK5_OUT * sizeof(float)); //512
	cudaMalloc((void**) &Layer12_Weights, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float)); //512*3*3*512 = 2359296
    cudaMalloc((void**) &Layer13_Neurons, (BLOCK5_OUT*BLOCK5_FM_SIZE) * sizeof(float)); //512*14*14

    cudaMemcpy(Layer12_bias, Layer12_bias_CPU, BLOCK5_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer12_Weights, Layer12_Weights_CPU, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(Layer12_bias,Layer12_Neurons,Layer12_Weights,Layer13_Neurons,14,14,1,1,3,512);

    /* 13th Layer */
    cudaMalloc((void**) &Layer13_bias, BLOCK5_OUT * sizeof(float)); //256
	cudaMalloc((void**) &Layer13_Weights, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float)); //512*3*3*512 = 2359296
    cudaMalloc((void**) &Layer13_Pool, (BLOCK5_OUT*BLOCK5_FM_SIZE) * sizeof(float)); //512*14*14

    cudaMemcpy(Layer13_bias, Layer13_bias_CPU, BLOCK5_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer13_Weights, Layer13_Weights_CPU, (BLOCK5_OUT*(FILTER_SIZE*512)) * sizeof(float), cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(Layer13_bias,Layer13_Neurons,Layer13_Weights,Layer13_Pool,14,14,1,1,3,512);

    cudaMalloc((void**) &Layer14_Neurons, (BLOCK5_OUT*(7*7)) * sizeof(float)); //512*7*7

	dim3 Block5_Pool_Block(512,1,1);
    max_jjb<<<Block5_Pool_Block,Block_Thread>>>(Layer13_Pool,Layer14_Neurons,14,7,2,0,2);

    /* 14th Layer */
	cudaMalloc((void**) &Layer14_bias, L14_OUT * sizeof(float)); //4096
	cudaMalloc((void**) &Layer14_Weights, (L14_OUT*(BLOCK5_OUT*(7*7))) * sizeof(float)); //4096*512*7*7
	cudaMalloc((void**) &Layer15_Neurons, L14_OUT * sizeof(float)); //4096

	cudaMemcpy(Layer14_bias, Layer14_bias_CPU, L14_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer14_Weights, Layer14_Weights_CPU, (L14_OUT*(BLOCK5_OUT*(7*7))) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 FC1_Block(4096,1,1);
	dim3 FC1_Thread(1,1);
	fc_jjb<<<FC1_Block,FC1_Thread>>>(Layer14_bias, Layer14_Neurons, Layer14_Weights, Layer15_Neurons, (7*7*512), true);

    /* 15th Layer */
	cudaMalloc((void**) &Layer15_bias, L15_OUT * sizeof(float)); //4096
	cudaMalloc((void**) &Layer15_Weights, (L15_OUT*L14_OUT) * sizeof(float)); //4096*4096
	cudaMalloc((void**) &Layer16_Neurons, L15_OUT * sizeof(float)); //4096

	cudaMemcpy(Layer15_bias, Layer15_bias_CPU, L15_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer15_Weights, Layer15_Weights_CPU, (L15_OUT*L14_OUT) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 FC2_Block(4096,1,1);
	dim3 FC2_Thread(1,1);
	fc_jjb<<<FC2_Block,FC2_Thread>>>(Layer15_bias, Layer15_Neurons, Layer15_Weights, Layer16_Neurons, 4096, true);

    /* 16th Layer */
	cudaMalloc((void**) &Layer16_bias, L16_OUT * sizeof(float)); //1000
	cudaMalloc((void**) &Layer16_Weights, (L16_OUT*L15_OUT) * sizeof(float)); //1000*4096
	cudaMalloc((void**) &Result_Neurons, L16_OUT * sizeof(float)); //1000

	cudaMemcpy(Layer16_bias, Layer16_bias_CPU, L16_OUT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer16_Weights, Layer16_Weights_CPU, (L16_OUT*L15_OUT) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 FC3_Block(1000,1,1);
	dim3 FC3_Thread(1,1);
	fc_jjb<<<FC3_Block,FC3_Thread>>>(Layer16_bias, Layer16_Neurons, Layer16_Weights, Result_Neurons, 4096, false);

	float *Result_Neurons_CPU = (float *) malloc ((1000) * sizeof(float));
	cudaMemcpy(Result_Neurons_CPU, Result_Neurons, (1000) * sizeof(float), cudaMemcpyDeviceToHost);
	// for(int i = 0; i < 1000; i++)
	// 	printf("%f\n",Result_Neurons_CPU[i]);

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

	cudaFree(Layer1_bias);
	cudaFree(Layer2_bias);
	cudaFree(Layer3_bias);
	cudaFree(Layer4_bias);
	cudaFree(Layer5_bias);
	cudaFree(Layer6_bias);
	cudaFree(Layer7_bias);
	cudaFree(Layer8_bias);
	cudaFree(Layer9_bias);
	cudaFree(Layer10_bias);
	cudaFree(Layer11_bias);
	cudaFree(Layer12_bias);
	cudaFree(Layer13_bias);
	cudaFree(Layer14_bias);
	cudaFree(Layer15_bias);
	cudaFree(Layer16_bias);

	cudaFree(Layer1_Neurons);
	cudaFree(Layer2_Neurons);
	cudaFree(Layer3_Neurons);
	cudaFree(Layer4_Neurons);
	cudaFree(Layer5_Neurons);
	cudaFree(Layer6_Neurons);
	cudaFree(Layer7_Neurons);
	cudaFree(Layer8_Neurons);
	cudaFree(Layer9_Neurons);
	cudaFree(Layer10_Neurons);
	cudaFree(Layer11_Neurons);
	cudaFree(Layer12_Neurons);
	cudaFree(Layer13_Neurons);
	cudaFree(Layer14_Neurons);
	cudaFree(Layer15_Neurons);
	cudaFree(Layer16_Neurons);

	cudaFree(Layer1_Weights);
	cudaFree(Layer2_Weights);
	cudaFree(Layer3_Weights);
	cudaFree(Layer4_Weights);
	cudaFree(Layer5_Weights);
	cudaFree(Layer6_Weights);
	cudaFree(Layer7_Weights);
	cudaFree(Layer8_Weights);
	cudaFree(Layer9_Weights);
	cudaFree(Layer10_Weights);
	cudaFree(Layer11_Weights);
	cudaFree(Layer12_Weights);
	cudaFree(Layer13_Weights);
	cudaFree(Layer14_Weights);
	cudaFree(Layer15_Weights);
	cudaFree(Layer16_Weights);

	cudaFree(Layer2_Pool);
	cudaFree(Layer4_Pool);
	cudaFree(Layer7_Pool);
	cudaFree(Layer10_Pool);
	cudaFree(Layer13_Pool);

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
	free(Layer9_bias_CPU);
	free(Layer10_bias_CPU);
	free(Layer11_bias_CPU);
	free(Layer12_bias_CPU);
	free(Layer13_bias_CPU);
	free(Layer14_bias_CPU);
	free(Layer15_bias_CPU);
	free(Layer16_bias_CPU);

    free(Layer1_Weights_CPU);
    free(Layer2_Weights_CPU);
    free(Layer3_Weights_CPU);
    free(Layer4_Weights_CPU);
    free(Layer5_Weights_CPU);
    free(Layer6_Weights_CPU);
    free(Layer7_Weights_CPU);
    free(Layer8_Weights_CPU);
	free(Layer9_Weights_CPU);
    free(Layer10_Weights_CPU);
    free(Layer11_Weights_CPU);
    free(Layer12_Weights_CPU);
    free(Layer13_Weights_CPU);
    free(Layer14_Weights_CPU);
    free(Layer15_Weights_CPU);
    free(Layer16_Weights_CPU);

	exit(0);
}