CFLAGS=-g 
LDFLAGS=-lm
TAR=`which tar` --exclude=.svn -czvf 
ARCHNAME=`date +%d-%m-%y-%H-%M`-`whoami`.tgz
ARCHCONTENT=*c *file include doc *.txt

compare: compare.o
	gcc -o $@ $^ $(LDFLAGS)

.o: .c
	gcc -c $< $(CFLAGS)
	
clean :
	rm -f *.o *~ compare

archive : 
	$(TAR) ../$(ARCHNAME) $(ARCHCONTENT) && mv ../$(ARCHNAME) .
	

