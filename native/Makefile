CC ?= cc
VULKAN_CFLAGS ?=
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -fPIC

all: build/libdlssnr-native.so

build/libdlssnr-native.so: runtime.c runtime.h
	mkdir -p build
	$(CC) $(CFLAGS) $(VULKAN_CFLAGS) -shared -o $@ runtime.c -lvulkan

.PHONY: all
