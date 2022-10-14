FC = gfortran
CC = gcc
CARGO = cargo

all: main.exe

clean:
	rm -f *.o *.exe
	rm -rf target

main.exe: main.o libml4a
	${FC} main.o -o main.exe -lml4a -L./target/release
	${CC} -c dummy.c

main.o: main.f90
	${FC} -c main.f90

libml4a: ml4a/lib.rs
	${CARGO} build --release
