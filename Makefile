GPU=1
OPENCV=1
DEBUG=0

ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20

VPATH=./src/
EXEC=ynet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread -lstdc++ 
COMMON= 
CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DYN_OPENCV
CFLAGS+= -DYN_OPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DYN_GPU -I/usr/local/cuda/include/ -I/usr/local/cuda/lib/
CFLAGS+= -DYN_GPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

OBJ= YnStd.o YnUtil.o YnGpu.o YnLayer.o YnNetwork.o YnActivation.o YnBBox.o YnBlas.o YnData.o YnGemm.o YnImage.o YnImageCv.o YnLayerActivation.o YnLayerAvgpool.o YnLayerConnected.o YnLayerConvolutional.o YnLayerCost.o YnLayerCrop.o YnLayerDeconvolutional.o YnLayerDetection.o YnLayerDropout.o YnLayerMaxpool.o YnLayerSoftmax.o YnList.o YnMatrix.o YnNet.o YnOptionList.o YnParser.o YnVoc.o
ifeq ($(GPU), 1) 
OBJ+= YnBlasGpu.o YnCudaGpu.o YnGemmGpu.o YnNetworkGpu.o YnActivationGpu.o YnImageGpu.o YnLayerActivationGpu.o YnLayerAvgpoolGpu.o YnLayerConnectedGpu.o YnLayerConvolutionalGpu.o YnLayerCostGpu.o YnLayerCropGpu.o YnLayerDeconvolutionalGpu.o YnLayerDetectionGpu.o YnLayerDropoutGpu.o YnLayerMaxpoolGpu.o YnLayerSoftmaxGpu.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard include/*.h, lib/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

