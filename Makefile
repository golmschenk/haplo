FC = gfortran
CC = gcc
CARGO = cargo

all: main.exe

clean:
	rm -f *.o *.exe
	rm -rf target

main.exe: main.o libhaplo
	${FC} main.o -o main.exe -lhaplo -L./target/release
	${CC} -c dummy.c

main.o: main.f90
	${FC} -c main.f90

libhaplo: src/lib.rs
	${CARGO} build --release
