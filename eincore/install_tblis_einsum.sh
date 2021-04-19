g++ -c -O3 -I/usr/local/include/tblis/util  -I/usr/local/include/tblis as_einsum.cxx -o as_einsum.o -L/usr/local/lib/ -ltblis -march=native  -fopenmp
g++ as_einsum.o -shared -I/usr/local/include/tblis/util  -I/usr/local/include/tblis -o libeinsum_tblis.so -L/usr/local/lib/ -ltblis   -march=native -fopenmp
mv ./libeinsum_tblis.so /usr/local/lib
rm *.o