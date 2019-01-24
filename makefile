GCC_FLAGS = -Wall

SOURCE = c_matrix_multiplication.c
DESTINATION = c_matrix_multiplication
LIBRARY_FLAGS = -lgsl -lgslcblas -lm

all:	$(SOURCE)
	gcc $(GCC_FLAGS) $(SOURCE)  -o $(DESTINATION) $(LIBRARY_FLAGS) 

clean:
	rm main
