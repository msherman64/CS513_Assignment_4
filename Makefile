all: gen multiply
multiply: managed.cu
	nvcc -o multiply managed.cu
gen: gen_file.cu
	nvcc -o gen_file gen_file.cu
clean:
	rm multiply gen_file
