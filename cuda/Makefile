all:bandwidthTest deviceQuery

bandwidthTest:bandwidthTest.cu
	nvcc -ccbin g++-4.8 -I. -o bandwidthTest bandwidthTest.cu

deviceQuery: deviceQuery.cpp
	nvcc -ccbin g++-4.8 -I. -o deviceQuery deviceQuery.cpp


clean:
	rm -f bandwidthTest deviceQuery
