.PHONY: all test check release
all:
	$(MAKE) -C native
	python3 native/probes/matrix_layout.py --shader native/build/matrix-layout.comp
	glslc --target-env=vulkan1.2 native/build/matrix-layout.comp -o native/build/matrix-layout.spv

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

check:
	python3 native/probes/runtime_check.py --library native/build/libdlssnr-native.so --spirv native/build/matrix-layout.spv

release:
	python3 package.py
