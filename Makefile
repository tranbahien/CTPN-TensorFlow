all:
	cython tools/bbox.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-I/usr/include/python2.7 -o tools/bbox.so tools/bbox.c
	rm -rf tools/bbox.c

	cython libs/cpu_nms.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
		-I/usr/include/python2.7 -o libs/cpu_nms.so libs/cpu_nms.c
	rm -rf libs/cpu_nms.c
