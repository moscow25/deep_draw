// This version supports low ranks other than 2 for Holdem.

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <string>
#include <vector>

#include "cards.h"
#include "constants.h"
#include "hand_evaluator.h"
#include "io.h"

using namespace std;

// This is not as general as it could be.  Holdem specific.
static void DealSevenCards(HandEvaluator *he) {
  Card min_card = MakeCard(2, 0);
  Card cards[7];
  Card c1, c2, c3, c4, c5, c6, c7;
  unsigned int *******tree = new unsigned int ******[(kMaxCard - min_card) + 1];
  for (c1 = min_card + 6; c1 <= kMaxCard; ++c1) {
    OutputCard(c1);
    printf("\n");
    fflush(stdout);
    cards[0] = c1;
    unsigned int i1 = c1 - min_card;
    unsigned int ******tree1 = new unsigned int *****[i1];
    tree[i1] = tree1;
    for (c2 = min_card + 5; c2 < c1; ++c2) {
      printf("  ");
      OutputCard(c2);
      printf("\n");
      fflush(stdout);
      cards[1] = c2;
      unsigned int i2 = c2 - min_card;
      unsigned int *****tree2 = new unsigned int ****[i2];
      tree1[i2] = tree2;
      for (c3 = min_card + 4; c3 < c2; ++c3) {
	cards[2] = c3;
	unsigned int i3 = c3 - min_card;
	unsigned int ****tree3 = new unsigned int ***[i3];
	tree2[i3] = tree3;
	for (c4 = min_card + 3; c4 < c3; ++c4) {
	  cards[3] = c4;
	  unsigned int i4 = c4 - min_card;
	  unsigned int ***tree4 = new unsigned int **[i4];
	  tree3[i4] = tree4;
	  for (c5 = min_card + 2; c5 < c4; ++c5) {
	    cards[4] = c5;
	    unsigned int i5 = c5 - min_card;
	    unsigned int **tree5 = new unsigned int *[i5];
	    tree4[i5] = tree5;
	    for (c6 = min_card + 1; c6 < c5; ++c6) {
	      cards[5] = c6;
	      unsigned int i6 = c6 - min_card;
	      unsigned int *tree6 = new unsigned int[i6];
	      tree5[i6] = tree6;
	      for (c7 = min_card; c7 < c6; ++c7) {
		cards[6] = c7;
		unsigned int i7 = c7 - min_card;
		tree6[i7] = he->Evaluate(cards, 7);
	      }
	    }
	  }
	}
      }
    }
  }
  char buf[500];
  sprintf(buf, "%s/hand_tree.holdem.2.0.7", g_data_root);
  Writer writer(buf);
  unsigned int max_code = kMaxCard - min_card;
  for (unsigned int i1 = 6; i1 <= max_code; ++i1) {
    unsigned int ******tree1 = tree[i1];
    for (unsigned int i2 = 5; i2 < i1; ++i2) {
      unsigned int *****tree2 = tree1[i2];
      for (unsigned int i3 = 4; i3 < i2; ++i3) {
	unsigned int ****tree3 = tree2[i3];
	for (unsigned int i4 = 3; i4 < i3; ++i4) {
	  unsigned int ***tree4 = tree3[i4];
	  for (unsigned int i5 = 2; i5 < i4; ++i5) {
	    unsigned int **tree5 = tree4[i5];
	    for (unsigned int i6 = 1; i6 < i5; ++i6) {
	      unsigned int *tree6 = tree5[i6];
	      for (unsigned int i7 = 0; i7 < i6; ++i7) {
		writer.WriteUnsignedInt(tree6[i7]);
	      }
	    }
	  }
	}
      }
    }
  }
}

static void Usage(const char *prog_name) {
  fprintf(stderr, "USAGE: %s\n", prog_name);
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc != 1) Usage(argv[0]);
  HandEvaluator he;
  DealSevenCards(&he);
}
