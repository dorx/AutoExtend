CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
INCLUDES = -I./eigen-3.2.4
LIBS = 

neu extract-data : NeuNet.o main.o extract-data.cpp
	$(CC) $(CFLAGS) -o neu NeuNet.o main.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) extract-data.cpp -o extract-data $(INCLUDES) $(LIBS)

NeuNet.o : NeuNet.cpp NeuNet.h
	$(CC) $(CFLAGS) -c NeuNet.cpp $(INCLUDES) $(LIBS)

main.o : main.cpp NeuNet.o
	$(CC) $(CFLAGS) -c main.cpp $(INCLUDES) $(LIBS)

clean :
	rm -rf *.o neu extract-data
