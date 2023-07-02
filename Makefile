CC=gcc
CFLAGS=-Wall

SDIR=./src
ODIR=./obj

CLSOBJ=./obj/mlpclassifier.o ./obj/utility.o
REGOBJ=./obj/mlpregression.o ./obj/utility.o

OBJ1=./obj/classifier.o ./obj/utility.o ./obj/readcfile.o  
OBJ2=./obj/regression.o ./obj/utility.o ./obj/readrfile.o

SRC2=./src/regression.c ./src/utility.c ./src/readrfile.c  
SRC1=./src/classifier.c ./src/utility.c ./src/readcfile.c

allc: $(CLSOBJ)
	gcc -shared -o ./lib/libMYClassifier.so $(CLSOBJ)
#	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$(pwd)/lib

allr: $(REGOBJ)
	gcc -shared -o ./lib/libMYRegressor.so $(REGOBJ)
#	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$(pwd)/lib

plots:
	gnuplot src/scripts.p

plotb:
	gnuplot src/scriptb.p

plotmb:
	gnuplot src/scriptmb.p


runc:
	./cls $(layers) $(ft) $(itr) $(lrate) $(nlist) $(actlist) $(sz) $(part) $(file) $(flag) $(batch)

runr:
	./reg $(layers) $(ft) $(itr) $(lrate) $(nlist) $(actlist) $(sz) $(part) $(file) $(flag) $(batch)

buildc: $(OBJ1) 
	$(CC) -L./lib -o cls $(SRC1) -Iinclude/ -lMYClassifier -lm

buildr: $(OBJ2)
	$(CC) -L./lib -o reg $(SRC2) -Iinclude/ -lMYRegressor -lm


$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) -c  -Werror -fpic -Iinclude/ $< -o $@

clean:
	rm -f obj/*.o buildc buildr *.csv ./lib/libMYClassifier.so ./lib/libMYRegressor.so cls reg

# make runc layers=3 ft=32 itr=1000 lrate=0.0001 nlist=50,2 actlist=sigmoid,ce sz=569 part=80 file=data.csv flag=1

#  export LD_LIBRARY_PATH=/home/baadalvm/Downloads/Assignment4/lib:$LD_LIBRARY_PATH