
CXX=mpic++
CXXFLAGS+=-Wall -Wextra
LDFLAGS+=-lpthread -lmpi -lmpi_cxx

SRCS=\
     hello.cpp

OBJS=$(patsubst %.cpp,%.o,$(SRCS))

hello : $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

run : hello
	mpirun -np 4 hello

clean:
	rm -rf hello
