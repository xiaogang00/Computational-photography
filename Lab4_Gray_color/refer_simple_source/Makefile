EXEC=color2gray
SRC= main.cpp image_io.cpp
OBJ=$(SRC:.cpp=.o)
# COMPILE VARS
CFLAGS = -O3
LDFLAGS= -lm
CC=g++

########################################
default: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $(EXEC) $(OBJ) $(LDFLAGS)

########################################

clean:
	@rm -f *.o
