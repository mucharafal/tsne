all: compile run

obj:
	g++ tSNE.cpp -O2 -c -o tSNE.o
	g++ Array.cpp -O2 -c -o Array.o

link:
	g++ Array.o tSNE.o -O2 -o tSNE.out

compile: obj link

run:
	./tSNE.out