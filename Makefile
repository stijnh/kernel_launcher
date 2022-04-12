fmt:
	clang-format -i include/kernel_launcher/*.hpp src/*.cpp tests/*.cpp

test: build
	cd build && make tests
	build/tests/tests

build:
	mkdir build
	cd build && cmake ..

.PHONY: fmt test
