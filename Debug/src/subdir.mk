################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/YnActivation.c \
../src/YnBBox.c \
../src/YnBlas.c \
../src/YnBlasGpu.c \
../src/YnCudaGpu.c \
../src/YnData.c \
../src/YnImage.c \
../src/YnImageCv.c \
../src/YnImageGpu.c \
../src/YnLayer.c \
../src/YnLayerActivation.c \
../src/YnLayerActivationGpu.c \
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
./src/YnBlasGpu.o \
./src/YnCudaGpu.o \
./src/YnData.o \
./src/YnImage.o \
./src/YnImageCv.o \
./src/YnImageGpu.o \
./src/YnLayer.o \
./src/YnLayerActivation.o \
./src/YnLayerActivationGpu.o \
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
./src/YnBlasGpu.d \
./src/YnCudaGpu.d \
./src/YnData.d \
./src/YnImage.d \
./src/YnImageCv.d \
./src/YnImageGpu.d \
./src/YnLayer.d \
./src/YnLayerActivation.d \
./src/YnLayerActivationGpu.d \
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


