#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ResNet18_kernel.cu"

/* Define Feature Map Sizes and Filter Sizes of Resnet18 */
#define INPUT_SIZE 224*224*3

extern "C"
void Resnet18();
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
		printf("Usage: ./RN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}
	Resnet18();
}

/* Function to Read Input Parameters */
void read_input(float *Layer1_Neurons_CPU){
    read_parameter("data_resnet18/input_cat.txt", Layer1_Neurons_CPU);
}

/* Function to Read All Weights */
void read_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,
                    float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU,
                    float *Layer9_Weights_CPU,float *Layer10_Weights_CPU,float *Layer11_Weights_CPU,float *Layer12_Weights_CPU,
                    float *Layer13_Weights_CPU,float *Layer14_Weights_CPU,float *Layer15_Weights_CPU,float *Layer16_Weights_CPU,
					float *Layer17_Weights_CPU,float *Block3_Weights_CPU,float *Block4_Weights_CPU,float *Block5_Weights_CPU){
	read_parameter("data_resnet18/conv_data/conv1.txt", Layer1_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv2.txt", Layer2_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv3.txt", Layer3_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv4.txt", Layer4_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv5.txt", Layer5_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv6.txt", Layer6_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv7.txt", Layer7_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv8.txt", Layer8_Weights_CPU);
 	read_parameter("data_resnet18/conv_data/conv9.txt", Layer9_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv10.txt", Layer10_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv11.txt", Layer11_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv12.txt", Layer12_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv13.txt", Layer13_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv14.txt", Layer14_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv15.txt", Layer15_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv16.txt", Layer16_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv17.txt", Layer17_Weights_CPU);

	read_parameter("data_resnet18/conv_data/conv_block3.txt", Block3_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv_block4.txt", Block4_Weights_CPU);
	read_parameter("data_resnet18/conv_data/conv_block5.txt", Block5_Weights_CPU);
}

/* Function to Read All Gamma */
void read_gamma(float *Layer1_Gamma_CPU,float *Layer2_Gamma_CPU,float *Layer3_Gamma_CPU,float *Layer4_Gamma_CPU,
                    float *Layer5_Gamma_CPU,float *Layer6_Gamma_CPU,float *Layer7_Gamma_CPU,float *Layer8_Gamma_CPU,
                    float *Layer9_Gamma_CPU,float *Layer10_Gamma_CPU,float *Layer11_Gamma_CPU,float *Layer12_Gamma_CPU,
                    float *Layer13_Gamma_CPU,float *Layer14_Gamma_CPU,float *Layer15_Gamma_CPU,float *Layer16_Gamma_CPU,
					float *Layer17_Gamma_CPU,float *Block3_Gamma_CPU,float *Block4_Gamma_CPU,float *Block5_Gamma_CPU){
	read_parameter("data_resnet18/gamma_data/gamma1.txt", Layer1_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma2.txt", Layer2_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma3.txt", Layer3_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma4.txt", Layer4_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma5.txt", Layer5_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma6.txt", Layer6_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma7.txt", Layer7_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma8.txt", Layer8_Gamma_CPU);
 	read_parameter("data_resnet18/gamma_data/gamma9.txt", Layer9_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma10.txt", Layer10_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma11.txt", Layer11_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma12.txt", Layer12_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma13.txt", Layer13_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma14.txt", Layer14_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma15.txt", Layer15_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma16.txt", Layer16_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma17.txt", Layer17_Gamma_CPU);

	read_parameter("data_resnet18/gamma_data/gamma_block3.txt", Block3_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma_block4.txt", Block4_Gamma_CPU);
	read_parameter("data_resnet18/gamma_data/gamma_block5.txt", Block5_Gamma_CPU);
}

/* Function to Read All Beta */
void read_beta(float *Layer1_Beta_CPU,float *Layer2_Beta_CPU,float *Layer3_Beta_CPU,float *Layer4_Beta_CPU,
                    float *Layer5_Beta_CPU,float *Layer6_Beta_CPU,float *Layer7_Beta_CPU,float *Layer8_Beta_CPU,
                    float *Layer9_Beta_CPU,float *Layer10_Beta_CPU,float *Layer11_Beta_CPU,float *Layer12_Beta_CPU,
                    float *Layer13_Beta_CPU,float *Layer14_Beta_CPU,float *Layer15_Beta_CPU,float *Layer16_Beta_CPU,
					float *Layer17_Beta_CPU,float *Block3_Beta_CPU,float *Block4_Beta_CPU,float *Block5_Beta_CPU){
	read_parameter("data_resnet18/beta_data/beta1.txt", Layer1_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta2.txt", Layer2_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta3.txt", Layer3_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta4.txt", Layer4_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta5.txt", Layer5_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta6.txt", Layer6_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta7.txt", Layer7_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta8.txt", Layer8_Beta_CPU);
 	read_parameter("data_resnet18/beta_data/beta9.txt", Layer9_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta10.txt", Layer10_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta11.txt", Layer11_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta12.txt", Layer12_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta13.txt", Layer13_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta14.txt", Layer14_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta15.txt", Layer15_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta16.txt", Layer16_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta17.txt", Layer17_Beta_CPU);

	read_parameter("data_resnet18/beta_data/beta_block3.txt", Block3_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta_block4.txt", Block4_Beta_CPU);
	read_parameter("data_resnet18/beta_data/beta_block5.txt", Block5_Beta_CPU);
}

/* Function to Read All Means */
void read_mean(float *mean1_CPU,float *mean2_CPU,float *mean3_CPU,float *mean4_CPU,
                    float *mean5_CPU,float *mean6_CPU,float *mean7_CPU,float *mean8_CPU,
                    float *mean9_CPU,float *mean10_CPU,float *mean11_CPU,float *mean12_CPU,
                    float *mean13_CPU,float *mean14_CPU,float *mean15_CPU,float *mean16_CPU,
					float *mean17_CPU,float *Block3_mean_CPU,float *Block4_mean_CPU,float *Block5_mean_CPU){
	read_parameter("data_resnet18/mean_data/mean1.txt", mean1_CPU);
	read_parameter("data_resnet18/mean_data/mean2.txt", mean2_CPU);
	read_parameter("data_resnet18/mean_data/mean3.txt", mean3_CPU);
	read_parameter("data_resnet18/mean_data/mean4.txt", mean4_CPU);
	read_parameter("data_resnet18/mean_data/mean5.txt", mean5_CPU);
	read_parameter("data_resnet18/mean_data/mean6.txt", mean6_CPU);
	read_parameter("data_resnet18/mean_data/mean7.txt", mean7_CPU);
	read_parameter("data_resnet18/mean_data/mean8.txt", mean8_CPU);
 	read_parameter("data_resnet18/mean_data/mean9.txt", mean9_CPU);
	read_parameter("data_resnet18/mean_data/mean10.txt", mean10_CPU);
	read_parameter("data_resnet18/mean_data/mean11.txt", mean11_CPU);
	read_parameter("data_resnet18/mean_data/mean12.txt", mean12_CPU);
	read_parameter("data_resnet18/mean_data/mean13.txt", mean13_CPU);
	read_parameter("data_resnet18/mean_data/mean14.txt", mean14_CPU);
	read_parameter("data_resnet18/mean_data/mean15.txt", mean15_CPU);
	read_parameter("data_resnet18/mean_data/mean16.txt", mean16_CPU);
	read_parameter("data_resnet18/mean_data/mean17.txt", mean17_CPU);

	read_parameter("data_resnet18/mean_data/mean_block3.txt", Block3_mean_CPU);
	read_parameter("data_resnet18/mean_data/mean_block4.txt", Block4_mean_CPU);
	read_parameter("data_resnet18/mean_data/mean_block5.txt", Block5_mean_CPU);
}

/* Function to Read All Variances */
void read_var(float *var1_CPU,float *var2_CPU,float *var3_CPU,float *var4_CPU,
                    float *var5_CPU,float *var6_CPU,float *var7_CPU,float *var8_CPU,
                    float *var9_CPU,float *var10_CPU,float *var11_CPU,float *var12_CPU,
                    float *var13_CPU,float *var14_CPU,float *var15_CPU,float *var16_CPU,
					float *var17_CPU,float *Block3_var_CPU,float *Block4_var_CPU,float *Block5_var_CPU){
	read_parameter("data_resnet18/var_data/var1.txt", var1_CPU);
	read_parameter("data_resnet18/var_data/var2.txt", var2_CPU);
	read_parameter("data_resnet18/var_data/var3.txt", var3_CPU);
	read_parameter("data_resnet18/var_data/var4.txt", var4_CPU);
	read_parameter("data_resnet18/var_data/var5.txt", var5_CPU);
	read_parameter("data_resnet18/var_data/var6.txt", var6_CPU);
	read_parameter("data_resnet18/var_data/var7.txt", var7_CPU);
	read_parameter("data_resnet18/var_data/var8.txt", var8_CPU);
 	read_parameter("data_resnet18/var_data/var9.txt", var9_CPU);
	read_parameter("data_resnet18/var_data/var10.txt", var10_CPU);
	read_parameter("data_resnet18/var_data/var11.txt", var11_CPU);
	read_parameter("data_resnet18/var_data/var12.txt", var12_CPU);
	read_parameter("data_resnet18/var_data/var13.txt", var13_CPU);
	read_parameter("data_resnet18/var_data/var14.txt", var14_CPU);
	read_parameter("data_resnet18/var_data/var15.txt", var15_CPU);
	read_parameter("data_resnet18/var_data/var16.txt", var16_CPU);
	read_parameter("data_resnet18/var_data/var17.txt", var17_CPU);

	read_parameter("data_resnet18/var_data/var_block3.txt", Block3_var_CPU);
	read_parameter("data_resnet18/var_data/var_block4.txt", Block4_var_CPU);
	read_parameter("data_resnet18/var_data/var_block5.txt", Block5_var_CPU);
}

/* Function to Read FC Parameters */
void read_fc_param(float *FC_bias_CPU,float *FC_Weights_CPU){
	read_parameter("data_resnet18/fc_data/fc1_bias.txt", FC_bias_CPU);
	read_parameter("data_resnet18/fc_data/fc1_weight.txt", FC_Weights_CPU);
}

/* Resnet18 */
void Resnet18(){

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

	float *Layer1_Weights_CPU = (float*) malloc ((7*7*3*64) * sizeof(float)); // = 9,408
	float *Layer2_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Layer3_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Layer4_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Layer5_Weights_CPU = (float*) malloc ((3*3*64*64) * sizeof(float)); // = 36,864
	float *Layer6_Weights_CPU = (float*) malloc ((3*3*64*128) * sizeof(float)); // = 73,728
	float *Layer7_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
	float *Layer8_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
    float *Layer9_Weights_CPU = (float*) malloc ((3*3*128*128) * sizeof(float)); // = 147,456
	float *Layer10_Weights_CPU = (float*) malloc ((3*3*128*256) * sizeof(float)); // = 294,912
	float *Layer11_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Layer12_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Layer13_Weights_CPU = (float*) malloc ((3*3*256*256) * sizeof(float)); // = 589,824
	float *Layer14_Weights_CPU = (float*) malloc ((3*3*256*512) * sizeof(float)); // = 1,179,648
	float *Layer15_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296
	float *Layer16_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296
	float *Layer17_Weights_CPU = (float*) malloc ((3*3*512*512) * sizeof(float)); // = 2,359,296

	float *Block3_Weights_CPU = (float*) malloc ((1*1*64*128) * sizeof(float)); // = 8,192
	float *Block4_Weights_CPU = (float*) malloc ((1*1*128*256) * sizeof(float)); // = 32,768
	float *Block5_Weights_CPU = (float*) malloc ((1*1*256*512) * sizeof(float)); // = 131,072
    read_weights(Layer1_Weights_CPU, Layer2_Weights_CPU, Layer3_Weights_CPU, Layer4_Weights_CPU,
                    Layer5_Weights_CPU, Layer6_Weights_CPU, Layer7_Weights_CPU, Layer8_Weights_CPU,
                    Layer9_Weights_CPU, Layer10_Weights_CPU, Layer11_Weights_CPU, Layer12_Weights_CPU,
                    Layer13_Weights_CPU, Layer14_Weights_CPU, Layer15_Weights_CPU, Layer16_Weights_CPU,
					Layer17_Weights_CPU, Block3_Weights_CPU, Block4_Weights_CPU, Block5_Weights_CPU);

    float *Layer1_Gamma_CPU = (float*) malloc (64 * sizeof(float));
	float *Layer2_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer3_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer4_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer5_Gamma_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer6_Gamma_CPU = (float*) malloc (128 * sizeof(float));
	float *Layer7_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Layer8_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Layer9_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Layer10_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer11_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer12_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer13_Gamma_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer14_Gamma_CPU = (float*) malloc (512 * sizeof(float));
	float *Layer15_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Layer16_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Layer17_Gamma_CPU = (float*) malloc (512 * sizeof(float));

	float *Block3_Gamma_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Block4_Gamma_CPU = (float*) malloc (256 * sizeof(float));
	float *Block5_Gamma_CPU = (float*) malloc (512 * sizeof(float)); 
    read_gamma(Layer1_Gamma_CPU, Layer2_Gamma_CPU, Layer3_Gamma_CPU, Layer4_Gamma_CPU,
                    Layer5_Gamma_CPU, Layer6_Gamma_CPU, Layer7_Gamma_CPU, Layer8_Gamma_CPU,
                    Layer9_Gamma_CPU, Layer10_Gamma_CPU, Layer11_Gamma_CPU, Layer12_Gamma_CPU,
                    Layer13_Gamma_CPU, Layer14_Gamma_CPU, Layer15_Gamma_CPU, Layer16_Gamma_CPU,
					Layer17_Gamma_CPU, Block3_Gamma_CPU, Block4_Gamma_CPU, Block5_Gamma_CPU);

	float *Layer1_Beta_CPU = (float*) malloc (64 * sizeof(float));
	float *Layer2_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer3_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer4_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer5_Beta_CPU = (float*) malloc (64 * sizeof(float)); 
	float *Layer6_Beta_CPU = (float*) malloc (128 * sizeof(float));
	float *Layer7_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Layer8_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
    float *Layer9_Beta_CPU = (float*) malloc (128 * sizeof(float)); 
	float *Layer10_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer11_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer12_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer13_Beta_CPU = (float*) malloc (256 * sizeof(float)); 
	float *Layer14_Beta_CPU = (float*) malloc (512 * sizeof(float));
	float *Layer15_Beta_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Layer16_Beta_CPU = (float*) malloc (512 * sizeof(float)); 
	float *Layer17_Beta_CPU = (float*) malloc (512 * sizeof(float));

	float *Block3_Beta_CPU = (float*) malloc (128 * sizeof(float));
	float *Block4_Beta_CPU = (float*) malloc (256 * sizeof(float));
	float *Block5_Beta_CPU = (float*) malloc (512 * sizeof(float));
    read_beta(Layer1_Beta_CPU, Layer2_Beta_CPU, Layer3_Beta_CPU, Layer4_Beta_CPU,
                    Layer5_Beta_CPU, Layer6_Beta_CPU, Layer7_Beta_CPU, Layer8_Beta_CPU,
                    Layer9_Beta_CPU, Layer10_Beta_CPU, Layer11_Beta_CPU, Layer12_Beta_CPU,
                    Layer13_Beta_CPU, Layer14_Beta_CPU, Layer15_Beta_CPU, Layer16_Beta_CPU,
					Layer17_Beta_CPU, Block3_Beta_CPU, Block4_Beta_CPU, Block5_Beta_CPU);

	float *mean1_CPU = (float*) malloc (64 * sizeof(float));
	float *mean2_CPU = (float*) malloc (64 * sizeof(float)); 
	float *mean3_CPU = (float*) malloc (64 * sizeof(float)); 
	float *mean4_CPU = (float*) malloc (64 * sizeof(float)); 
	float *mean5_CPU = (float*) malloc (64 * sizeof(float)); 
	float *mean6_CPU = (float*) malloc (128 * sizeof(float));
	float *mean7_CPU = (float*) malloc (128 * sizeof(float)); 
	float *mean8_CPU = (float*) malloc (128 * sizeof(float)); 
    float *mean9_CPU = (float*) malloc (128 * sizeof(float)); 
	float *mean10_CPU = (float*) malloc (256 * sizeof(float)); 
	float *mean11_CPU = (float*) malloc (256 * sizeof(float)); 
	float *mean12_CPU = (float*) malloc (256 * sizeof(float)); 
	float *mean13_CPU = (float*) malloc (256 * sizeof(float)); 
	float *mean14_CPU = (float*) malloc (512 * sizeof(float));
	float *mean15_CPU = (float*) malloc (512 * sizeof(float)); 
	float *mean16_CPU = (float*) malloc (512 * sizeof(float)); 
	float *mean17_CPU = (float*) malloc (512 * sizeof(float));

	float *Block3_mean_CPU = (float*) malloc (128 * sizeof(float));
	float *Block4_mean_CPU = (float*) malloc (256 * sizeof(float));
	float *Block5_mean_CPU = (float*) malloc (512 * sizeof(float));
    read_mean(mean1_CPU, mean2_CPU, mean3_CPU, mean4_CPU,
                    mean5_CPU, mean6_CPU, mean7_CPU, mean8_CPU,
                    mean9_CPU, mean10_CPU, mean11_CPU, mean12_CPU,
                    mean13_CPU, mean14_CPU, mean15_CPU, mean16_CPU,
					mean17_CPU, Block3_mean_CPU, Block4_mean_CPU, Block5_mean_CPU);
	
	float *var1_CPU = (float*) malloc (64 * sizeof(float));
	float *var2_CPU = (float*) malloc (64 * sizeof(float)); 
	float *var3_CPU = (float*) malloc (64 * sizeof(float)); 
	float *var4_CPU = (float*) malloc (64 * sizeof(float)); 
	float *var5_CPU = (float*) malloc (64 * sizeof(float)); 
	float *var6_CPU = (float*) malloc (128 * sizeof(float));
	float *var7_CPU = (float*) malloc (128 * sizeof(float)); 
	float *var8_CPU = (float*) malloc (128 * sizeof(float)); 
    float *var9_CPU = (float*) malloc (128 * sizeof(float)); 
	float *var10_CPU = (float*) malloc (256 * sizeof(float)); 
	float *var11_CPU = (float*) malloc (256 * sizeof(float)); 
	float *var12_CPU = (float*) malloc (256 * sizeof(float)); 
	float *var13_CPU = (float*) malloc (256 * sizeof(float)); 
	float *var14_CPU = (float*) malloc (512 * sizeof(float));
	float *var15_CPU = (float*) malloc (512 * sizeof(float)); 
	float *var16_CPU = (float*) malloc (512 * sizeof(float)); 
	float *var17_CPU = (float*) malloc (512 * sizeof(float));

	float *Block3_var_CPU = (float*) malloc (128 * sizeof(float));
	float *Block4_var_CPU = (float*) malloc (256 * sizeof(float));
	float *Block5_var_CPU = (float*) malloc (512 * sizeof(float));
    read_var(var1_CPU, var2_CPU, var3_CPU, var4_CPU,
                    var5_CPU, var6_CPU, var7_CPU, var8_CPU,
                    var9_CPU, var10_CPU, var11_CPU, var12_CPU,
                    var13_CPU, var14_CPU, var15_CPU, var16_CPU,
					var17_CPU, Block3_var_CPU, Block4_var_CPU, Block5_var_CPU);

	float *FC_bias_CPU = (float*) malloc (1000* sizeof(float));
	float *FC_Weights_CPU = (float*) malloc ((512*1000) * sizeof(float));
	read_fc_param(FC_bias_CPU, FC_Weights_CPU);


    //* Block A (1 layer)*//

	dim3 Block_Thread(7,7);
	dim3 Single_Thread(1,1);

	dim3 Block1_Block(64,16,16);

	//1st layer
	float *Layer1_Neurons, *Layer1_Weights, *Layer1_Bn, *mean1, *var1, *Layer1_Gamma, *Layer1_Beta, *Layer1_Pool, *Layer2_Neurons;
	cudaMalloc((void**) &Layer1_Neurons, sizeof(float) * INPUT_SIZE); //224*224*3
	cudaMalloc((void**) &Layer1_Weights, sizeof(float) * (7*7*3*64));
	cudaMalloc((void**) &Layer1_Bn, sizeof(float) * (112*112*64));
	cudaMalloc((void**) &mean1, sizeof(float) * 64);
	cudaMalloc((void**) &var1, sizeof(float) * 64);
	cudaMalloc((void**) &Layer1_Gamma, sizeof(float) * 64);
	cudaMalloc((void**) &Layer1_Beta, sizeof(float) * 64);
	cudaMalloc((void**) &Layer1_Pool, sizeof(float) * (112*112*64));
	cudaMalloc((void**) &Layer2_Neurons, sizeof(float) * (56*56*64));

	cudaMemcpy(Layer1_Neurons, Layer1_Neurons_CPU, sizeof(float) * INPUT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Weights, Layer1_Weights_CPU, sizeof(float) * (7*7*3*64), cudaMemcpyHostToDevice);
	cudaMemcpy(mean1, mean1_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(var1, var1_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Gamma, Layer1_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer1_Beta, Layer1_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	
	cudaEvent_t start1, end1;
	float inference_time1;
	cudaEventCreate(&start1);
	cudaEventCreate(&end1);
	cudaEventRecord(start1,0);
	first_jjb<<<Block1_Block,Block_Thread>>>(NULL,Layer1_Neurons,Layer1_Weights,Layer1_Bn,224,112,2,3,7,3,false,true);
	cudaEventRecord(end1, 0);
	cudaEventSynchronize(end1);
	cudaEventElapsedTime(&inference_time1, start1, end1);
	printf("Elapsed time: %f ms\n", inference_time1);
	cudaEventDestroy(start1);
	cudaEventDestroy(end1);
	batchnorm_jjb<<<Block1_Block,Block_Thread>>>(Layer1_Bn,Layer1_Pool,mean1,var1,Layer1_Gamma,Layer1_Beta,112,true);

	dim3 Block1_Pool_Block(64,8,8);
	max_jjb<<<Block1_Pool_Block,Block_Thread>>>(Layer1_Pool,Layer2_Neurons,112,56,2,1,3);	

	//* Block B (2,3,4,5 layer) *//
	/* 2 Identity Block */ 

	dim3 Block2_Block(64,8,8);

	// 2nd layer
	float *Layer2_Weights, *Layer2_Bn, *mean2, *var2, *Layer2_Gamma, *Layer2_Beta, *Layer3_Neurons;
	cudaMalloc((void**) &Layer2_Weights, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Layer2_Bn, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &mean2, sizeof(float) * 64);
	cudaMalloc((void**) &var2, sizeof(float) * 64);
	cudaMalloc((void**) &Layer2_Gamma, sizeof(float) * 64);
	cudaMalloc((void**) &Layer2_Beta, sizeof(float) * 64);
	cudaMalloc((void**) &Layer3_Neurons, sizeof(float) * (56*56*64));

	cudaMemcpy(Layer2_Weights, Layer2_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(mean2, mean2_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(var2, var2_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer2_Gamma, Layer2_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer2_Beta, Layer2_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);

	
	conv_jjb<<<Block2_Block,Block_Thread>>>(NULL,Layer2_Neurons,Layer2_Weights,Layer2_Bn,56,56,1,1,3,64,false,false);
	batchnorm_jjb<<<Block2_Block,Block_Thread>>>(Layer2_Bn,Layer3_Neurons,mean2,var2,Layer2_Gamma,Layer2_Beta,56,true);

	// 3rd layer
	float *Layer3_Weights, *Layer3_Bn, *mean3, *var3, *Layer3_Gamma, *Layer3_Beta, *Layer3_Basic, *Layer4_Neurons;
	cudaMalloc((void**) &Layer3_Weights, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Layer3_Bn, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &mean3, sizeof(float) * 64);
	cudaMalloc((void**) &var3, sizeof(float) * 64);
	cudaMalloc((void**) &Layer3_Gamma, sizeof(float) * 64);
	cudaMalloc((void**) &Layer3_Beta, sizeof(float) * 64);
	cudaMalloc((void**) &Layer3_Basic, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &Layer4_Neurons, sizeof(float) * (56*56*64));

	cudaMemcpy(Layer3_Weights, Layer3_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(mean3, mean3_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(var3, var3_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer3_Gamma, Layer3_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer3_Beta, Layer3_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);

	conv_jjb<<<Block2_Block,Block_Thread>>>(NULL,Layer3_Neurons,Layer3_Weights,Layer3_Bn,56,56,1,1,3,64,false,false);
	batchnorm_jjb<<<Block2_Block,Block_Thread>>>(Layer3_Bn,Layer3_Basic,mean3,var3,Layer3_Gamma,Layer3_Beta,56,false);

	basic_block<<<Block2_Block,Block_Thread>>>(Layer2_Neurons,Layer3_Basic,Layer4_Neurons,56,true);

	// 4th layer
	float *Layer4_Weights, *Layer4_Bn, *mean4, *var4, *Layer4_Gamma, *Layer4_Beta, *Layer5_Neurons;

	cudaMalloc((void**) &Layer4_Weights, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Layer4_Bn, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &mean4, sizeof(float) * 64);
	cudaMalloc((void**) &var4, sizeof(float) * 64);
	cudaMalloc((void**) &Layer4_Gamma, sizeof(float) * 64);
	cudaMalloc((void**) &Layer4_Beta, sizeof(float) * 64);
	cudaMalloc((void**) &Layer5_Neurons, sizeof(float) * (56*56*64));

	cudaMemcpy(Layer4_Weights, Layer4_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(mean4, mean4_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(var4, var4_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer4_Gamma, Layer4_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer4_Beta, Layer4_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);

	conv_jjb<<<Block2_Block,Block_Thread>>>(NULL,Layer4_Neurons,Layer4_Weights,Layer4_Bn,56,56,1,1,3,64,false,false);
	batchnorm_jjb<<<Block2_Block,Block_Thread>>>(Layer4_Bn,Layer5_Neurons,mean4,var4,Layer4_Gamma,Layer4_Beta,56,true);

	//5th layer
	float *Layer5_Weights, *Layer5_Bn, *mean5, *var5, *Layer5_Gamma, *Layer5_Beta, *Layer5_Basic, *Layer6_Neurons;
	cudaMalloc((void**) &Layer5_Weights, sizeof(float) * (3*3*64*64));
	cudaMalloc((void**) &Layer5_Bn, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &mean5, sizeof(float) * 64);
	cudaMalloc((void**) &var5, sizeof(float) * 64);
	cudaMalloc((void**) &Layer5_Gamma, sizeof(float) * 64);
	cudaMalloc((void**) &Layer5_Beta, sizeof(float) * 64);
	cudaMalloc((void**) &Layer5_Basic, sizeof(float) * (56*56*64));
	cudaMalloc((void**) &Layer6_Neurons, sizeof(float) * (56*56*64));

	cudaMemcpy(Layer5_Weights, Layer5_Weights_CPU, sizeof(float) * (3*3*64*64), cudaMemcpyHostToDevice);
	cudaMemcpy(mean5, mean5_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(var5, var5_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer5_Gamma, Layer5_Gamma_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer5_Beta, Layer5_Beta_CPU, sizeof(float) * 64, cudaMemcpyHostToDevice);
	
	conv_jjb<<<Block2_Block,Block_Thread>>>(NULL,Layer5_Neurons,Layer5_Weights,Layer5_Bn,56,56,1,1,3,64,false,false);
	batchnorm_jjb<<<Block2_Block,Block_Thread>>>(Layer5_Bn,Layer5_Basic,mean5,var5,Layer5_Gamma,Layer5_Beta,56,false);

	basic_block<<<Block2_Block,Block_Thread>>>(Layer4_Neurons,Layer5_Basic,Layer6_Neurons,56,true);

	//* Block C (6,7,8,9 layer) *//
	/* 1 Convolution Block, 1 Identity Block */

	dim3 Block3_Block(128,4,4);

	//6th layer
	float *Layer6_Weights, *Layer6_Bn, *mean6, *var6, *Layer6_Gamma, *Layer6_Beta, *Layer7_Neurons;
	cudaMalloc((void**) &Layer6_Weights, sizeof(float) * (3*3*64*128));
	cudaMalloc((void**) &Layer6_Bn, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &mean6, sizeof(float) * 128);
	cudaMalloc((void**) &var6, sizeof(float) * 128);
	cudaMalloc((void**) &Layer6_Gamma, sizeof(float) * 128);
	cudaMalloc((void**) &Layer6_Beta, sizeof(float) * 128);
	cudaMalloc((void**) &Layer7_Neurons, sizeof(float) * (28*28*128));

	cudaMemcpy(Layer6_Weights, Layer6_Weights_CPU, sizeof(float) * (3*3*64*128), cudaMemcpyHostToDevice);
	cudaMemcpy(mean6, mean6_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(var6, var6_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer6_Gamma, Layer6_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer6_Beta, Layer6_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	
	conv_jjb<<<Block3_Block,Block_Thread>>>(NULL,Layer6_Neurons,Layer6_Weights,Layer6_Bn,56,28,2,1,3,64,false,false);
	batchnorm_jjb<<<Block3_Block,Block_Thread>>>(Layer6_Bn,Layer7_Neurons,mean6,var6,Layer6_Gamma,Layer6_Beta,28,true);

	//7th layer
	float *Layer7_Weights, *Layer7_Bn, *mean7, *var7, *Layer7_Gamma, *Layer7_Beta, *Layer7_Basic;
	cudaMalloc((void**) &Layer7_Weights, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Layer7_Bn, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &mean7, sizeof(float) * 128);
	cudaMalloc((void**) &var7, sizeof(float) * 128);
	cudaMalloc((void**) &Layer7_Gamma, sizeof(float) * 128);
	cudaMalloc((void**) &Layer7_Beta, sizeof(float) * 128);
	cudaMalloc((void**) &Layer7_Basic, sizeof(float) * (28*28*128));
	
	cudaMemcpy(Layer7_Weights, Layer7_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);
	cudaMemcpy(mean7, mean7_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(var7, var7_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer7_Gamma, Layer7_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer7_Beta, Layer7_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);

	conv_jjb<<<Block3_Block,Block_Thread>>>(NULL,Layer7_Neurons,Layer7_Weights,Layer7_Bn,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block3_Block,Block_Thread>>>(Layer7_Bn,Layer7_Basic,mean7,var7,Layer7_Gamma,Layer7_Beta,28,false);

	//Block B output
	float *Block3_Weights, *Block3_Bn, *Block3_mean, *Block3_var, *Block3_Gamma, *Block3_Beta, *Block3_Basic, *Layer8_Neurons;
	cudaMalloc((void**) &Block3_Weights, sizeof(float) * (1*1*64*128));
	cudaMalloc((void**) &Block3_Bn, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &Block3_mean, sizeof(float) * 128);
	cudaMalloc((void**) &Block3_var, sizeof(float) * 128);
	cudaMalloc((void**) &Block3_Gamma, sizeof(float) * 128);
	cudaMalloc((void**) &Block3_Beta, sizeof(float) * 128);
	cudaMalloc((void**) &Block3_Basic, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &Layer8_Neurons, sizeof(float) * (28*28*128));

	cudaMemcpy(Block3_Weights, Block3_Weights_CPU, sizeof(float) * (1*1*64*128), cudaMemcpyHostToDevice);
	cudaMemcpy(Block3_mean, Block3_mean_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Block3_var, Block3_var_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Block3_Gamma, Block3_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Block3_Beta, Block3_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	
	conv_jjb<<<Block3_Block,Block_Thread>>>(NULL,Layer6_Neurons,Block3_Weights,Block3_Bn,56,28,2,0,1,64,false,false); 
	batchnorm_jjb<<<Block3_Block,Block_Thread>>>(Block3_Bn,Block3_Basic,Block3_mean,Block3_var,Block3_Gamma,Block3_Beta,28,false);

	basic_block<<<Block3_Block,Block_Thread>>>(Layer7_Basic,Block3_Basic,Layer8_Neurons,28,true);

	//8th layer
	float *Layer8_Weights, *Layer8_Bn, *mean8, *var8, *Layer8_Gamma, *Layer8_Beta, *Layer9_Neurons;
	cudaMalloc((void**) &Layer8_Weights, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Layer8_Bn, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &mean8, sizeof(float) * 128);
	cudaMalloc((void**) &var8, sizeof(float) * 128);
	cudaMalloc((void**) &Layer8_Gamma, sizeof(float) * 128);
	cudaMalloc((void**) &Layer8_Beta, sizeof(float) * 128);
	cudaMalloc((void**) &Layer9_Neurons, sizeof(float) * (28*28*128));

	cudaMemcpy(Layer8_Weights, Layer8_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);
	cudaMemcpy(mean8, mean8_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(var8, var8_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer8_Gamma, Layer8_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer8_Beta, Layer8_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);

	conv_jjb<<<Block3_Block,Block_Thread>>>(NULL,Layer8_Neurons,Layer8_Weights,Layer8_Bn,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block3_Block,Block_Thread>>>(Layer8_Bn,Layer9_Neurons,mean8,var8,Layer8_Gamma,Layer8_Beta,28,true);

	//9th layer
	float *Layer9_Weights, *Layer9_Bn, *mean9, *var9, *Layer9_Gamma, *Layer9_Beta, *Layer9_Basic, *Layer10_Neurons;
	cudaMalloc((void**) &Layer9_Weights, sizeof(float) * (3*3*128*128));
	cudaMalloc((void**) &Layer9_Bn, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &mean9, sizeof(float) * 128);
	cudaMalloc((void**) &var9, sizeof(float) * 128);
	cudaMalloc((void**) &Layer9_Gamma, sizeof(float) * 128);
	cudaMalloc((void**) &Layer9_Beta, sizeof(float) * 128);
	cudaMalloc((void**) &Layer9_Basic, sizeof(float) * (28*28*128));
	cudaMalloc((void**) &Layer10_Neurons, sizeof(float) * (28*28*128));

	cudaMemcpy(Layer9_Weights, Layer9_Weights_CPU, sizeof(float) * (3*3*128*128), cudaMemcpyHostToDevice);
	cudaMemcpy(mean9, mean9_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(var9, var9_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer9_Gamma, Layer9_Gamma_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer9_Beta, Layer9_Beta_CPU, sizeof(float) * 128, cudaMemcpyHostToDevice);

	conv_jjb<<<Block3_Block,Block_Thread>>>(NULL,Layer9_Neurons,Layer9_Weights,Layer9_Bn,28,28,1,1,3,128,false,false);
	batchnorm_jjb<<<Block3_Block,Block_Thread>>>(Layer9_Bn,Layer9_Basic,mean9,var9,Layer9_Gamma,Layer9_Beta,28,false);

	basic_block<<<Block3_Block,Block_Thread>>>(Layer8_Neurons,Layer9_Basic,Layer10_Neurons,28,true);

	//* Block D (10,11,12,13 layer) *//
	/* 1 Convolution Block, 1 Identity Block */

	dim3 Block4_Block(256,2,2);

	//10th layer
	float *Layer10_Weights, *Layer10_Bn, *mean10, *var10, *Layer10_Gamma, *Layer10_Beta, *Layer11_Neurons;
	cudaMalloc((void**) &Layer10_Weights, sizeof(float) * (3*3*128*256));
	cudaMalloc((void**) &Layer10_Bn, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &mean10, sizeof(float) * 256);
	cudaMalloc((void**) &var10, sizeof(float) * 256);
	cudaMalloc((void**) &Layer10_Gamma, sizeof(float) * 256);
	cudaMalloc((void**) &Layer10_Beta, sizeof(float) * 256);
	cudaMalloc((void**) &Layer11_Neurons, sizeof(float) * (14*14*256));

	cudaMemcpy(Layer10_Weights, Layer10_Weights_CPU, sizeof(float) * (3*3*128*256), cudaMemcpyHostToDevice);
	cudaMemcpy(mean10, mean10_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(var10, var10_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer10_Gamma, Layer10_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer10_Beta, Layer10_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(NULL,Layer10_Neurons,Layer10_Weights,Layer10_Bn,28,14,2,1,3,128,false,false);
	batchnorm_jjb<<<Block4_Block,Block_Thread>>>(Layer10_Bn,Layer11_Neurons,mean10,var10,Layer10_Gamma,Layer10_Beta,14,true);

	//11th layer
	float *Layer11_Weights, *Layer11_Bn, *mean11, *var11, *Layer11_Gamma, *Layer11_Beta, *Layer11_Basic;
	cudaMalloc((void**) &Layer11_Weights, sizeof(float) * (3*3*256*256));
	cudaMalloc((void**) &Layer11_Bn, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &mean11, sizeof(float) * 256);
	cudaMalloc((void**) &var11, sizeof(float) * 256);
	cudaMalloc((void**) &Layer11_Gamma, sizeof(float) * 256);
	cudaMalloc((void**) &Layer11_Beta, sizeof(float) * 256);
	cudaMalloc((void**) &Layer11_Basic, sizeof(float) * (14*14*256));

	cudaMemcpy(Layer11_Weights, Layer11_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(mean11, mean11_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(var11, var11_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer11_Gamma, Layer11_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer11_Beta, Layer11_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	
	conv_jjb<<<Block4_Block,Block_Thread>>>(NULL,Layer11_Neurons,Layer11_Weights,Layer11_Bn,14,14,1,1,3,256,false,false);
	batchnorm_jjb<<<Block4_Block,Block_Thread>>>(Layer11_Bn,Layer11_Basic,mean11,var11,Layer11_Gamma,Layer11_Beta,14,false);

	//Block C output
	float *Block4_Weights, *Block4_Bn, *Block4_mean, *Block4_var, *Block4_Gamma, *Block4_Beta, *Block4_Basic, *Layer12_Neurons;
	cudaMalloc((void**) &Block4_Weights, sizeof(float) * (1*1*128*256));
	cudaMalloc((void**) &Block4_Bn, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &Block4_mean, sizeof(float) * 256);
	cudaMalloc((void**) &Block4_var, sizeof(float) * 256);
	cudaMalloc((void**) &Block4_Gamma, sizeof(float) * 256);
	cudaMalloc((void**) &Block4_Beta, sizeof(float) * 256);
	cudaMalloc((void**) &Block4_Basic, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &Layer12_Neurons, sizeof(float) * (14*14*256));

	cudaMemcpy(Block4_Weights, Block4_Weights_CPU, sizeof(float) * (1*1*128*256), cudaMemcpyHostToDevice);
	cudaMemcpy(Block4_mean, Block4_mean_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Block4_var, Block4_var_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Block4_Gamma, Block4_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Block4_Beta, Block4_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(NULL,Layer10_Neurons,Block4_Weights,Block4_Bn,28,14,2,0,1,128,false,false);
	batchnorm_jjb<<<Block4_Block,Block_Thread>>>(Block4_Bn,Block4_Basic,Block4_mean,Block4_var,Block4_Gamma,Block4_Beta,14,false);

	basic_block<<<Block4_Block,Block_Thread>>>(Layer11_Basic,Block4_Basic,Layer12_Neurons,14,true);

	//12th layer
	float *Layer12_Weights, *Layer12_Bn, *mean12, *var12, *Layer12_Gamma, *Layer12_Beta, *Layer13_Neurons;
	cudaMalloc((void**) &Layer12_Weights, sizeof(float) * (3*3*256*256));
	cudaMalloc((void**) &Layer12_Bn, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &mean12, sizeof(float) * 256);
	cudaMalloc((void**) &var12, sizeof(float) * 256);
	cudaMalloc((void**) &Layer12_Gamma, sizeof(float) * 256);
	cudaMalloc((void**) &Layer12_Beta, sizeof(float) * 256);
	cudaMalloc((void**) &Layer13_Neurons, sizeof(float) * (14*14*256));

	cudaMemcpy(Layer12_Weights, Layer12_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(mean12, mean12_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(var12, var12_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer12_Gamma, Layer12_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer12_Beta, Layer12_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(NULL,Layer12_Neurons,Layer12_Weights,Layer12_Bn,14,14,1,1,3,256,false,false);
	batchnorm_jjb<<<Block4_Block,Block_Thread>>>(Layer12_Bn,Layer13_Neurons,mean12,var12,Layer12_Gamma,Layer12_Beta,14,true);

	//13th layer
	float *Layer13_Weights, *Layer13_Bn, *mean13, *var13, *Layer13_Gamma, *Layer13_Beta, *Layer13_Basic, *Layer14_Neurons;
	cudaMalloc((void**) &Layer13_Weights, sizeof(float) * (3*3*256*256));
	cudaMalloc((void**) &Layer13_Bn, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &mean13, sizeof(float) * 256);
	cudaMalloc((void**) &var13, sizeof(float) * 256);
	cudaMalloc((void**) &Layer13_Gamma, sizeof(float) * 256);
	cudaMalloc((void**) &Layer13_Beta, sizeof(float) * 256);
	cudaMalloc((void**) &Layer13_Basic, sizeof(float) * (14*14*256));
	cudaMalloc((void**) &Layer14_Neurons, sizeof(float) * (14*14*256));

	cudaMemcpy(Layer13_Weights, Layer13_Weights_CPU, sizeof(float) * (3*3*256*256), cudaMemcpyHostToDevice);
	cudaMemcpy(mean13, mean13_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(var13, var13_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer13_Gamma, Layer13_Gamma_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer13_Beta, Layer13_Beta_CPU, sizeof(float) * 256, cudaMemcpyHostToDevice);

	conv_jjb<<<Block4_Block,Block_Thread>>>(NULL,Layer13_Neurons,Layer13_Weights,Layer13_Bn,14,14,1,1,3,256,false,false); 
	batchnorm_jjb<<<Block4_Block,Block_Thread>>>(Layer13_Bn,Layer13_Basic,mean13,var13,Layer13_Gamma,Layer13_Beta,14,false);

	basic_block<<<Block4_Block,Block_Thread>>>(Layer12_Neurons,Layer13_Basic,Layer14_Neurons,14,true);

	//* Block E (14,15,16,17 layer) *//
	/* 1 Convolution Block, 1 Identity Block */

	dim3 Block5_Block(512,1,1);

	//14th layer
	float *Layer14_Weights, *Layer14_Bn, *mean14, *var14, *Layer14_Gamma, *Layer14_Beta, *Layer15_Neurons;
	cudaMalloc((void**) &Layer14_Weights, sizeof(float) * (3*3*256*512));
	cudaMalloc((void**) &Layer14_Bn, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &mean14, sizeof(float) * 512);
	cudaMalloc((void**) &var14, sizeof(float) * 512);
	cudaMalloc((void**) &Layer14_Gamma, sizeof(float) * 512);
	cudaMalloc((void**) &Layer14_Beta, sizeof(float) * 512);
	cudaMalloc((void**) &Layer15_Neurons, sizeof(float) * (7*7*512));

	cudaMemcpy(Layer14_Weights, Layer14_Weights_CPU, sizeof(float) * (3*3*256*512), cudaMemcpyHostToDevice);
	cudaMemcpy(mean14, mean14_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(var14, var14_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer14_Gamma, Layer14_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer14_Beta, Layer14_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(NULL,Layer14_Neurons,Layer14_Weights,Layer14_Bn,14,7,2,1,3,256,false,false);
	batchnorm_jjb<<<Block5_Block,Block_Thread>>>(Layer14_Bn,Layer15_Neurons,mean14,var14,Layer14_Gamma,Layer14_Beta,7,true);

	//15th layer
	float *Layer15_Weights, *Layer15_Bn, *mean15, *var15, *Layer15_Gamma, *Layer15_Beta, *Layer15_Basic;
	cudaMalloc((void**) &Layer15_Weights, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Layer15_Bn, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &mean15, sizeof(float) * 512);
	cudaMalloc((void**) &var15, sizeof(float) * 512);
	cudaMalloc((void**) &Layer15_Gamma, sizeof(float) * 512);
	cudaMalloc((void**) &Layer15_Beta, sizeof(float) * 512);
	cudaMalloc((void**) &Layer15_Basic, sizeof(float) * (7*7*512));

	cudaMemcpy(Layer15_Weights, Layer15_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(mean15, mean15_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(var15, var15_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer15_Gamma, Layer15_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer15_Beta, Layer15_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(NULL,Layer15_Neurons,Layer15_Weights,Layer15_Bn,7,7,1,1,3,512,false,false);
	batchnorm_jjb<<<Block5_Block,Block_Thread>>>(Layer15_Bn,Layer15_Basic,mean15,var15,Layer15_Gamma,Layer15_Beta,7,false);

	//Block D output
	float *Block5_Weights, *Block5_Bn, *Block5_mean, *Block5_var, *Block5_Gamma, *Block5_Beta, *Block5_Basic, *Layer16_Neurons;
	cudaMalloc((void**) &Block5_Weights, sizeof(float) * (1*1*256*512));
	cudaMalloc((void**) &Block5_Bn, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &Block5_mean, sizeof(float) * 512);
	cudaMalloc((void**) &Block5_var, sizeof(float) * 512);
	cudaMalloc((void**) &Block5_Gamma, sizeof(float) * 521);
	cudaMalloc((void**) &Block5_Beta, sizeof(float) * 512);
	cudaMalloc((void**) &Block5_Basic, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &Layer16_Neurons, sizeof(float) * (7*7*512));

	cudaMemcpy(Block5_Weights, Block5_Weights_CPU, sizeof(float) * (1*1*256*512), cudaMemcpyHostToDevice);
	cudaMemcpy(Block5_mean, Block5_mean_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Block5_var, Block5_var_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Block5_Gamma, Block5_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Block5_Beta, Block5_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(NULL,Layer14_Neurons,Block5_Weights,Block5_Bn,14,7,2,0,1,256,false,false);
	batchnorm_jjb<<<Block5_Block,Block_Thread>>>(Block5_Bn,Block5_Basic,Block5_mean,Block5_var,Block5_Gamma,Block5_Beta,7,false);

	basic_block<<<Block5_Block,Block_Thread>>>(Layer15_Basic,Block5_Basic,Layer16_Neurons,7,true);

	//16th layer
	float *Layer16_Weights, *Layer16_Bn, *mean16, *var16, *Layer16_Gamma, *Layer16_Beta, *Layer17_Neurons;
	cudaMalloc((void**) &Layer16_Weights, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Layer16_Bn, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &mean16, sizeof(float) * 512);
	cudaMalloc((void**) &var16, sizeof(float) * 512);
	cudaMalloc((void**) &Layer16_Gamma, sizeof(float) * 512);
	cudaMalloc((void**) &Layer16_Beta, sizeof(float) * 512);
	cudaMalloc((void**) &Layer17_Neurons, sizeof(float) * (7*7*512));

	cudaMemcpy(Layer16_Weights, Layer16_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(mean16, mean16_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(var16, var16_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer16_Gamma, Layer16_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer16_Beta, Layer16_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(NULL,Layer16_Neurons,Layer16_Weights,Layer16_Bn,7,7,1,1,3,512,false,false);
	batchnorm_jjb<<<Block5_Block,Block_Thread>>>(Layer16_Bn,Layer17_Neurons,mean16,var16,Layer16_Gamma,Layer16_Beta,7,true);


	//17th layer
	float *Layer17_Weights, *Layer17_Bn, *mean17, *var17, *Layer17_Gamma, *Layer17_Beta, *Layer17_Basic, *Layer18_Neurons;
	cudaMalloc((void**) &Layer17_Weights, sizeof(float) * (3*3*512*512));
	cudaMalloc((void**) &Layer17_Bn, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &mean17, sizeof(float) * 512);
	cudaMalloc((void**) &var17, sizeof(float) * 512);
	cudaMalloc((void**) &Layer17_Gamma, sizeof(float) * 512);
	cudaMalloc((void**) &Layer17_Beta, sizeof(float) * 512);
	cudaMalloc((void**) &Layer17_Basic, sizeof(float) * (7*7*512));
	cudaMalloc((void**) &Layer18_Neurons, sizeof(float) * (7*7*512));

	cudaMemcpy(Layer17_Weights, Layer17_Weights_CPU, sizeof(float) * (3*3*512*512), cudaMemcpyHostToDevice);
	cudaMemcpy(mean17, mean17_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(var17, var17_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer17_Gamma, Layer17_Gamma_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);
	cudaMemcpy(Layer17_Beta, Layer17_Beta_CPU, sizeof(float) * 512, cudaMemcpyHostToDevice);

	conv_jjb<<<Block5_Block,Block_Thread>>>(NULL,Layer17_Neurons,Layer17_Weights,Layer17_Bn,7,7,1,1,3,512,false,false); 
	batchnorm_jjb<<<Block5_Block,Block_Thread>>>(Layer17_Bn,Layer17_Basic,mean17,var17,Layer17_Gamma,Layer17_Beta,7,false);

	basic_block<<<Block5_Block,Block_Thread>>>(Layer16_Neurons,Layer17_Basic,Layer18_Neurons,7,true);

	//* Fully Connected (18 layer) *//
	float *FC_bias, *FC_Weights, *FC_Neurons, *Result_Neurons;
	cudaMalloc((void**) &FC_bias, sizeof(float) * 1000);
	cudaMalloc((void**) &FC_Weights, sizeof(float) * (512*1000));
	cudaMalloc((void**) &FC_Neurons, sizeof(float) * 512);
	cudaMalloc((void**) &Result_Neurons, sizeof(float) * 1000);

	cudaMemcpy(FC_bias, FC_bias_CPU, sizeof(float) * 1000, cudaMemcpyHostToDevice);
	cudaMemcpy(FC_Weights, FC_Weights_CPU, sizeof(float) * (512*1000), cudaMemcpyHostToDevice);
	
	dim3 Avg_Block(512,1,1);
	globalavg_jjb<<<Avg_Block,Single_Thread>>>(Layer18_Neurons,FC_Neurons,7);

	dim3 FC_Block(1000,1,1);
	fc_jjb<<<FC_Block,Single_Thread>>>(FC_bias,FC_Neurons,FC_Weights,Result_Neurons,512,false);

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
	cudaFree(Layer17_Neurons);

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
	cudaFree(Layer17_Weights);
	cudaFree(Block3_Weights);
	cudaFree(Block4_Weights);
	cudaFree(Block5_Weights);

	cudaFree(Layer1_Gamma);
	cudaFree(Layer2_Gamma);
	cudaFree(Layer3_Gamma);
	cudaFree(Layer4_Gamma);
	cudaFree(Layer5_Gamma);
	cudaFree(Layer6_Gamma);
	cudaFree(Layer7_Gamma);
	cudaFree(Layer8_Gamma);
	cudaFree(Layer9_Gamma);
	cudaFree(Layer10_Gamma);
	cudaFree(Layer11_Gamma);
	cudaFree(Layer12_Gamma);
	cudaFree(Layer13_Gamma);
	cudaFree(Layer14_Gamma);
	cudaFree(Layer15_Gamma);
	cudaFree(Layer16_Gamma);
	cudaFree(Layer17_Gamma);
	cudaFree(Block3_Gamma);
	cudaFree(Block4_Gamma);
	cudaFree(Block5_Gamma);

	cudaFree(Layer1_Beta);
	cudaFree(Layer2_Beta);
	cudaFree(Layer3_Beta);
	cudaFree(Layer4_Beta);
	cudaFree(Layer5_Beta);
	cudaFree(Layer6_Beta);
	cudaFree(Layer7_Beta);
	cudaFree(Layer8_Beta);
	cudaFree(Layer9_Beta);
	cudaFree(Layer10_Beta);
	cudaFree(Layer11_Beta);
	cudaFree(Layer12_Beta);
	cudaFree(Layer13_Beta);
	cudaFree(Layer14_Beta);
	cudaFree(Layer15_Beta);
	cudaFree(Layer16_Beta);
	cudaFree(Layer17_Beta);
	cudaFree(Block3_Beta);
	cudaFree(Block4_Beta);
	cudaFree(Block5_Beta);

	cudaFree(Layer3_Basic);
	cudaFree(Layer5_Basic);
	cudaFree(Layer7_Basic);
	cudaFree(Layer9_Basic);
	cudaFree(Layer11_Basic);
	cudaFree(Layer13_Basic);
	cudaFree(Layer15_Basic);
	cudaFree(Layer17_Basic);
	cudaFree(Block3_Basic);
	cudaFree(Block4_Basic);
	cudaFree(Block5_Basic);

	cudaFree(Layer1_Bn);
	cudaFree(Layer2_Bn);
	cudaFree(Layer3_Bn);
	cudaFree(Layer4_Bn);
	cudaFree(Layer5_Bn);
	cudaFree(Layer6_Bn);
	cudaFree(Layer7_Bn);
	cudaFree(Layer8_Bn);
	cudaFree(Layer9_Bn);
	cudaFree(Layer10_Bn);
	cudaFree(Layer11_Bn);
	cudaFree(Layer12_Bn);
	cudaFree(Layer13_Bn);
	cudaFree(Layer14_Bn);
	cudaFree(Layer15_Bn);
	cudaFree(Layer16_Bn);
	cudaFree(Layer17_Bn);
	cudaFree(Block3_Bn);
	cudaFree(Block4_Bn);
	cudaFree(Block5_Bn);

	cudaFree(mean1);
	cudaFree(mean2);
	cudaFree(mean3);
	cudaFree(mean4);
	cudaFree(mean5);
	cudaFree(mean6);
	cudaFree(mean7);
	cudaFree(mean8);
	cudaFree(mean9);
	cudaFree(mean10);
	cudaFree(mean11);
	cudaFree(mean12);
	cudaFree(mean13);
	cudaFree(mean14);
	cudaFree(mean15);
	cudaFree(mean16);
	cudaFree(mean17);
	cudaFree(Block3_mean);
	cudaFree(Block4_mean);
	cudaFree(Block5_mean);

	cudaFree(var1);
	cudaFree(var2);
	cudaFree(var3);
	cudaFree(var4);
	cudaFree(var5);
	cudaFree(var6);
	cudaFree(var7);
	cudaFree(var8);
	cudaFree(var9);
	cudaFree(var10);
	cudaFree(var11);
	cudaFree(var12);
	cudaFree(var13);
	cudaFree(var14);
	cudaFree(var15);
	cudaFree(var16);
	cudaFree(var17);
	cudaFree(Block3_var);
	cudaFree(Block4_var);
	cudaFree(Block5_var);

	cudaFree(Layer1_Pool);

	cudaFree(FC_Neurons);
	cudaFree(FC_bias);
	cudaFree(FC_Weights);
	cudaFree(Result_Neurons);


	free(Layer1_Neurons_CPU);

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
	free(Layer17_Weights_CPU);
    free(Block3_Weights_CPU);
    free(Block4_Weights_CPU);
    free(Block5_Weights_CPU);

	free(Layer1_Gamma_CPU);
    free(Layer2_Gamma_CPU);
    free(Layer3_Gamma_CPU);
    free(Layer4_Gamma_CPU);
    free(Layer5_Gamma_CPU);
    free(Layer6_Gamma_CPU);
    free(Layer7_Gamma_CPU);
    free(Layer8_Gamma_CPU);
	free(Layer9_Gamma_CPU);
    free(Layer10_Gamma_CPU);
    free(Layer11_Gamma_CPU);
    free(Layer12_Gamma_CPU);
    free(Layer13_Gamma_CPU);
    free(Layer14_Gamma_CPU);
    free(Layer15_Gamma_CPU);
    free(Layer16_Gamma_CPU);
	free(Layer17_Gamma_CPU);
    free(Block3_Gamma_CPU);
    free(Block4_Gamma_CPU);
    free(Block5_Gamma_CPU);

	free(Layer1_Beta_CPU);
    free(Layer2_Beta_CPU);
    free(Layer3_Beta_CPU);
    free(Layer4_Beta_CPU);
    free(Layer5_Beta_CPU);
    free(Layer6_Beta_CPU);
    free(Layer7_Beta_CPU);
    free(Layer8_Beta_CPU);
	free(Layer9_Beta_CPU);
    free(Layer10_Beta_CPU);
    free(Layer11_Beta_CPU);
    free(Layer12_Beta_CPU);
    free(Layer13_Beta_CPU);
    free(Layer14_Beta_CPU);
    free(Layer15_Beta_CPU);
    free(Layer16_Beta_CPU);
	free(Layer17_Beta_CPU);
    free(Block3_Beta_CPU);
    free(Block4_Beta_CPU);
    free(Block5_Beta_CPU);

	free(mean1_CPU);
	free(mean2_CPU);
	free(mean3_CPU);
	free(mean4_CPU);
	free(mean5_CPU);
	free(mean6_CPU);
	free(mean7_CPU);
	free(mean8_CPU);
	free(mean9_CPU);
	free(mean10_CPU);
	free(mean11_CPU);
	free(mean12_CPU);
	free(mean13_CPU);
	free(mean14_CPU);
	free(mean15_CPU);
	free(mean16_CPU);
	free(mean17_CPU);

	free(Block3_mean_CPU);
	free(Block4_mean_CPU);
	free(Block5_mean_CPU);

	free(var1_CPU);
	free(var2_CPU);
	free(var3_CPU);
	free(var4_CPU);
	free(var5_CPU);
	free(var6_CPU);
	free(var7_CPU);
	free(var8_CPU);
	free(var9_CPU);
	free(var10_CPU);
	free(var11_CPU);
	free(var12_CPU);
	free(var13_CPU);
	free(var14_CPU);
	free(var15_CPU);
	free(var16_CPU);
	free(var17_CPU);

	free(Block3_var_CPU);
	free(Block4_var_CPU);
	free(Block5_var_CPU);

	exit(0);
}