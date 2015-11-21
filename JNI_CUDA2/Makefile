# Makefile
# Generic Makefile for making cuda programs
#
BIN               := program
 
# flags
CUDA_INSTALL_PATH := /usr/local/cuda
#CUDA_SDK_PATH     := /home/pli/NVIDIA_GPU_Computing_SDK
JDK_PATH	  := $(JAVA_HOME)
#INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/C/common/inc -I$(JDK_PATH)/include -I$(JDK_PATH)/include/linux
INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -I$(JDK_PATH)/include -I$(JDK_PATH)/include/linux
LIBS              := -L$(CUDA_INSTALL_PATH)/lib64 -L.
CFLAGS            := -O2 -m64 -fPIC -MMD -MP
LDFLAGS           := -lrt -lm -lcudart -lcuda 
# compilers
JAVAC		  := javac
JAVAH		  := javah
NVCC              := nvcc
CC                := gcc
LINKER            := gcc
# files
J_SOURCES	  := $(wildcard *.java)
C_SOURCES         := $(wildcard *.c)
#CLASS_FILES	  := $(patsubst %.java, %.class, $(J_SOURCES))
CLASS_FILES	  := $(wildcard *.class)
CU_SOURCES        := $(wildcard *.cu)
HEADERS           := $(wildcard *.h)
C_OBJS            := $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))
ALL_OBJS	  := $(wildcard *.o)
 
$(BIN):	$(HEADERS)
	$(NVCC) -m64 -shared -Xcompiler -fPIC -o $(BIN).so $(LDFLAGS) $(INCLUDES) $(LIBS) $(CU_SOURCES)

java: clean $(J_SOURCES)
	$(JAVAC) $(J_SOURCES)
	$(JAVAH) -jni -classpath ./ Main #$(CLASS_FILES)

clean:
	rm -f $(BIN) Main.h *.d *.o *.so *~ *.o.d *.class
