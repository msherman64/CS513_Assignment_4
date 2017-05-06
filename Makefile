multiply: managed.cu
	nvcc -g -o multiply managed.cu
clean:
	rm multiply
