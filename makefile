all: compile run

obj:
	g++ tSNE.cpp -O2 -c -o tSNE.o

link:
	g++ tSNE.o -O2 -o tSNE.out

compile: obj link

run:
	./tSNE.out