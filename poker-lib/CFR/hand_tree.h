#ifndef _HAND_TREE_H_
#define _HAND_TREE_H_

#include "cards.h"

class HandTree {
public:
  HandTree(void);
  ~HandTree(void);
  // Does *not* assume cards are sorted
  unsigned int Val(Card *cards);
  // board and hole_cards are filled with integer values which are equal to
  // card value minus kMinCard.  board and hole_cards should be sorted from
  // high to low.
  unsigned int Val(unsigned int *board, unsigned int *hole_cards);
private:
  void ReadSeven(void);

  unsigned int *******tree7_;
};

#endif
