multiply: managed.cu
	nvcc -o multiply managed.cu
clean:
	rm multiply
