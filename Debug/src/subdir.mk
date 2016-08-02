################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/YnActivation.c \
../src/YnBBox.c \
../src/YnBlas.c \
../src/YnCudaGpu.c \
../src/YnData.c \
../src/YnGemm.c \
../src/YnGemmGpu.c \
../src/YnImage.c \
../src/YnImageCv.c \
../src/YnLayer.c \
../src/YnLayerActivation.c \
../src/YnLayerActivationGpu.c \
../src/YnLayerAvgPool.c \
../src/YnLayerConnected.c \
../src/YnLayerConnectedGpu.c \
../src/YnLayerConvolutional.c \
../src/YnLayerConvolutionalGpu.c \
../src/YnList.c \
../src/YnMatrix.c \
../src/YnNetwork.c \
../src/YnOptionList.c \
../src/YnParser.c \
../src/YnStd.c \
../src/YnUtil.c 

OBJS += \
./src/YnActivation.o \
./src/YnBBox.o \
./src/YnBlas.o \
./src/YnCudaGpu.o \
./src/YnData.o \
./src/YnGemm.o \
./src/YnGemmGpu.o \
./src/YnImage.o \
./src/YnImageCv.o \
./src/YnLayer.o \
./src/YnLayerActivation.o \
./src/YnLayerActivationGpu.o \
./src/YnLayerAvgPool.o \
./src/YnLayerConnected.o \
./src/YnLayerConnectedGpu.o \
./src/YnLayerConvolutional.o \
./src/YnLayerConvolutionalGpu.o \
./src/YnList.o \
./src/YnMatrix.o \
./src/YnNetwork.o \
./src/YnOptionList.o \
./src/YnParser.o \
./src/YnStd.o \
./src/YnUtil.o 

C_DEPS += \
./src/YnActivation.d \
./src/YnBBox.d \
./src/YnBlas.d \
./src/YnCudaGpu.d \
./src/YnData.d \
./src/YnGemm.d \
./src/YnGemmGpu.d \
./src/YnImage.d \
./src/YnImageCv.d \
./src/YnLayer.d \
./src/YnLayerActivation.d \
./src/YnLayerActivationGpu.d \
./src/YnLayerAvgPool.d \
./src/YnLayerConnected.d \
./src/YnLayerConnectedGpu.d \
./src/YnLayerConvolutional.d \
./src/YnLayerConvolutionalGpu.d \
./src/YnList.d \
./src/YnMatrix.d \
./src/YnNetwork.d \
./src/YnOptionList.d \
./src/YnParser.d \
./src/YnStd.d \
./src/YnUtil.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


