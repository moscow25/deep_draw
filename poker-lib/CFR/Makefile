# Note that if any header files are missing when you try to build, things fail
# in mysterious ways.  You get told there is "No rule to make target obj/foo.o".
HEADS =	src/constants.h src/cards.h src/canonical.h src/hand_evaluator.h \
	src/io.h src/hand_tree.h

# Can't seem to get things to work without putting library explicitly on
# link line.

# LDFLAGS = -llzo2

LIBRARIES = -lpthread # /usr/lib/x86_64-linux-gnu/liblzo2.a -lpthread

# -g -pg for profiling (and no -O2?) - change link line too
# -g -gstabs+ for debugging (and no -O2?)
# -O2 for optimization
# -DMEMACCT for memory accounting
# -DCYGWIN for cygwin
# To run with Google profiling, add /usr/local/lib/libprofiler.a and
# /usr/local/lib/libunwind.a to end of link line, and call ProfilerStart()
# and ProfilerStop().  (You will need to #include "gperftools/profiler.h".)
CFLAGS = -std=c++0x -Wall -O2

obj/%.o:	src/%.cpp $(HEADS)
		gcc $(CFLAGS) -c -o $@ $<

OBJS =	obj/constants.o obj/cards.o obj/canonical.o obj/hand_evaluator.o \
	obj/io.o obj/hand_tree.o

bin/build_hand_tree:	obj/build_hand_tree.o $(OBJS) $(HEADS)
	g++ $(LDFLAGS) $(CFLAGS) -o bin/build_hand_tree \
	obj/build_hand_tree.o $(OBJS) $(LIBRARIES)

bin/build_wins_and_losses:	obj/build_wins_and_losses.o $(OBJS) $(HEADS)
	g++ $(LDFLAGS) $(CFLAGS) -o bin/build_wins_and_losses \
	obj/build_wins_and_losses.o $(OBJS) $(LIBRARIES)

bin/test:	obj/test.o $(OBJS) $(HEADS)
	g++ $(LDFLAGS) $(CFLAGS) -o bin/test obj/test.o $(OBJS) \
	$(LIBRARIES)

