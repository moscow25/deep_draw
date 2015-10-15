#include <stdio.h>

#include <algorithm>
#include <vector>

#include "cards.h"
#include "constants.h"
#include "hand_tree.h"
#include "io.h"

using namespace std;

HandTree::HandTree(void) {
  printf("Reading HandTree from disk...");
  tree7_ = NULL;
  ReadSeven();
}

void HandTree::ReadSeven(void) {
  Card min_card = MakeCard(2, 0);
  char buf[500];
  sprintf(buf, "%s/hand_tree.holdem.2.0.7", g_data_root);
  Reader reader(buf);
  unsigned int num_cards = (kMaxCard - min_card) + 1;
  tree7_ = new unsigned int ******[num_cards];
  for (unsigned int i = 0; i < num_cards; ++i) tree7_[i] = NULL;
  for (unsigned int i1 = 6; i1 < num_cards; ++i1) {
    unsigned int ******tree1 = new unsigned int *****[i1];
    tree7_[i1] = tree1;
    for (unsigned int i2 = 5; i2 < i1; ++i2) {
      unsigned int *****tree2 = new unsigned int ****[i2];
      tree1[i2] = tree2;
      for (unsigned int i3 = 4; i3 < i2; ++i3) {
	unsigned int ****tree3 = new unsigned int ***[i3];
	tree2[i3] = tree3;
	for (unsigned int i4 = 3; i4 < i3; ++i4) {
	  unsigned int ***tree4 = new unsigned int **[i4];
	  tree3[i4] = tree4;
	  for (unsigned int i5 = 2; i5 < i4; ++i5) {
	    unsigned int **tree5 = new unsigned int *[i5];
	    tree4[i5] = tree5;
	    for (unsigned int i6 = 1; i6 < i5; ++i6) {
	      unsigned int *tree6 = new unsigned int[i6];
	      tree5[i6] = tree6;
	      for (unsigned int i7 = 0; i7 < i6; ++i7) {
		tree6[i7] = reader.ReadUnsignedIntOrDie();
	      }
	    }
	  }
	}
      }
    }
  }
}

HandTree::~HandTree(void) {
  Card min_card = MakeCard(2, 0);
  unsigned int max_code = kMaxCard - min_card;
  for (unsigned int i1 = 6; i1 <= max_code; ++i1) {
    unsigned int ******tree1 = tree7_[i1];
    for (unsigned int i2 = 5; i2 < i1; ++i2) {
      unsigned int *****tree2 = tree1[i2];
      for (unsigned int i3 = 4; i3 < i2; ++i3) {
	unsigned int ****tree3 = tree2[i3];
	for (unsigned int i4 = 3; i4 < i3; ++i4) {
	  unsigned int ***tree4 = tree3[i4];
	  for (unsigned int i5 = 2; i5 < i4; ++i5) {
	    unsigned int **tree5 = tree4[i5];
	    for (unsigned int i6 = 1; i6 < i5; ++i6) {
	      delete [] tree5[i6];
	    }
	    delete [] tree5;
	  }
	  delete [] tree4;
	}
	delete [] tree3;
      }
      delete [] tree2;
    }
    delete [] tree1;
  }
  delete [] tree7_;
}

unsigned int HandTree::Val(Card *cards) {
  Card min_card = MakeCard(2, 0);
  vector<Card> v(7);
  v[0] = cards[0] - min_card;
  v[1] = cards[1] - min_card;
  v[2] = cards[2] - min_card;
  v[3] = cards[3] - min_card;
  v[4] = cards[4] - min_card;
  v[5] = cards[5] - min_card;
  v[6] = cards[6] - min_card;
  sort(v.begin(), v.end());
  return tree7_[v[6]][v[5]][v[4]][v[3]][v[2]][v[1]][v[0]];
}

// board and hole_cards are filled with integer values which are equal to
// card value minus the minimum card value possible for the game.  board and
// hole_cards should be sorted from high to low.
// NOTE: requires board to have 5 cards, and hole_cards to have 2 cards. Full hands only!
unsigned int HandTree::Val(unsigned int *board, unsigned int *hole_cards) {
  unsigned int a[7];
  unsigned int i = 0, j = 0, k = 0;
  int b = board[i];
  int h = hole_cards[j];
  while (i < 5 || j < 2) {
    if (b > h) {
      a[k++] = b;
      ++i;
      if (i < 5) b = board[i];
      else       b = -1;
    } else {
      a[k++] = h;
      ++j;
      if (j < 2) h = hole_cards[j];
      else       h = -1;
    }
  }
  return tree7_[a[0]][a[1]][a[2]][a[3]][a[4]][a[5]][a[6]];
}

