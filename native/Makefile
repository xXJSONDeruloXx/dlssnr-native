CC ?= cc
VULKAN_CFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -fPIC

SOURCES = runtime.c model_package.c model_dll.c sha256.c model_decode.c dlssnr.c
HEADERS = runtime.h model_package.h model_dll.h sha256.h model_decode.h dlssnr.h

all: build/libdlssnr-native.so

build/libdlssnr-native.so: $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -shared -Wl,-soname,libdlssnr-native.so.1 -o $@ $(SOURCES) -lvulkan -lm

.PHONY: all
