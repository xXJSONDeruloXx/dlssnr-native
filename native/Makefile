CC ?= cc
VULKAN_CFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -fPIC

SOURCES = runtime.c model_package.c sha256.c dlssnr.c
HEADERS = runtime.h model_package.h sha256.h dlssnr.h

all: build/libdlssnr-native.so

build/libdlssnr-native.so: $(SOURCES) $(HEADERS)
	mkdir -p build
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -shared -Wl,-soname,libdlssnr-native.so.1 -o $@ $(SOURCES) -lvulkan

.PHONY: all
