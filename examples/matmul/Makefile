CC=g++
CFLAGS=-std=c++14 -I../../thirdparty -I../../include -g -O2 -Wall -Wextra -pedantic
LIBS=-lcuda -lcudart -lnvrtc

BIN=main

all: $(BIN)

$(BIN): main.cpp matmul.cu
	$(CC) $< -o $@ $(CFLAGS) $(LIBS)

clean:
	rm -f $(BIN)

.PHONY: clean fmt
