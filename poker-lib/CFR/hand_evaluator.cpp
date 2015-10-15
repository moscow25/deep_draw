#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>

#include "cards.h"
#include "hand_evaluator.h"

using namespace std;

HandEvaluator::HandEvaluator(void) {
  ranks_ = new int[7];
  suits_ = new int[7];
  rank_counts_ = new int[15];
  suit_counts_ = new int[4];
}

HandEvaluator::~HandEvaluator(void) {
  delete [] suits_;
  delete [] ranks_;
  delete [] rank_counts_;
  delete [] suit_counts_;
}

// Haven't tested passing fewer than 5 cards into Evaluate() but it appears
// it should work.  Mmm, need to make kickers work.
int HandEvaluator::Evaluate(Card *cards, int num_cards) {
  for (int r = 2; r <= 14; ++r) rank_counts_[r] = 0;
  for (int s = 0; s < 4; ++s)   suit_counts_[s] = 0;
  for (int i = 0; i < num_cards; ++i) {
    Card c = cards[i];
    int r = Rank(c);
    ranks_[i] = r;
    ++rank_counts_[r];
    int s = Suit(c);
    suits_[i] = s;
    ++suit_counts_[s];
  }
  int flush_suit = -1;
  for (int s = 0; s < 4; ++s) {
    if (suit_counts_[s] >= 5) {
      flush_suit = s;
      break;
    }
  }
  // Need to handle straights with ace as low
  int r = 14;
  int straight_rank = -1;
  while (true) {
    // See if there are 5 ranks (r, r-1, r-2, etc.) such that there is at
    // least one card in each rank.  In other words, there is an r-high
    // straight.
    int r1 = r;
    int end = r - 4;
    while (r1 >= end &&
	   ((r1 > 1 && rank_counts_[r1] > 0) ||
	    (r1 == 1 && rank_counts_[14] > 0))) {
      --r1;
    }
    if (r1 == end - 1) {
      // We found a straight
      if (flush_suit >= 0) {
	// There is a flush on the board
	if (straight_rank == -1) straight_rank = r;
	// Need to check for straight flush.  Count how many cards between
	// end and r are in the flush suit.
	int num = 0;
	for (int i = 0; i < num_cards; ++i) {
	  if (suits_[i] == flush_suit &&
	      ((ranks_[i] >= end && ranks_[i] <= r) ||
	       (end == 1 && ranks_[i] == 14))) {
	    // This assumes we have no duplicate cards in input
	    ++num;
	  }
	}
	if (num == 5) {
	  return kStraightFlush + (r - 2);
	}
	// Can't break yet - there could be a straight flush at a lower rank
	// Can only decrement r by 1.  (Suppose cards are:
	// 4c5c6c7c8c9s.)
	--r;
	if (r < 5) break;
      } else {
	straight_rank = r;
	break;
      }
    } else {
      // If we get here, there was no straight ending at r.  We know there
      // are no cards with rank r1.  Therefore r can jump to r1 - 1.
      r = r1 - 1;
      if (r < 5) break;
    }
  }
  int three_rank = -1;
  int pair_rank = -1;
  int pair2_rank = -1;
  for (int r = 14; r >= 2; --r) {
    int ct = rank_counts_[r];
    if (ct == 4) {
      int hr = -1;
      for (int i = 0; i < num_cards; ++i) {
	int r1 = ranks_[i];
	if (r1 != r && r1 > hr) hr = r1;
      }
      return kQuads + (r - 2) * 13 + hr - 2;
    } else if (ct == 3) {
      if (three_rank == -1) {
	three_rank = r;
      } else if (pair_rank == -1) {
	pair_rank = r;
      }
    } else if (ct == 2) {
      if (pair_rank == -1) {
	pair_rank = r;
      } else if (pair2_rank == -1) {
	pair2_rank = r;
      }
    }
  }
  if (three_rank >= 0 && pair_rank >= 0) {
    return kFullHouse + (three_rank - 2) * 13 + pair_rank - 2;
  }
  if (flush_suit >= 0) {
    int hr1 = -1, hr2 = -1, hr3 = -1, hr4 = -1, hr5 = -1;
    for (int i = 0; i < num_cards; ++i) {
      if (suits_[i] == flush_suit) {
	int r = ranks_[i];
	if (r > hr1) {
	  hr5 = hr4; hr4 = hr3; hr3 = hr2; hr2 = hr1; hr1 = r;
	} else if (r > hr2) {
	  hr5 = hr4; hr4 = hr3; hr3 = hr2; hr2 = r;
	} else if (r > hr3) {
	  hr5 = hr4; hr4 = hr3; hr3 = r;
	} else if (r > hr4) {
	  hr5 = hr4; hr4 = r;
	} else if (r > hr5) {
	  hr5 = r;
	}
      }
    }
    return kFlush + (hr1 - 2) * 28561 + (hr2 - 2) * 2197 + (hr3 - 2) * 169 +
      (hr4 - 2) * 13 + (hr5 - 2);
  }
  if (straight_rank >= 0) {
    return kStraight + straight_rank - 2;
  }
  if (three_rank >= 0) {
    int hr1 = -1, hr2 = -1;
    for (int i = 0; i < num_cards; ++i) {
      int r = ranks_[i];
      if (r != three_rank) {
	if (r > hr1) {
	  hr2 = hr1; hr1 = r;
	} else if (r > hr2) {
	  hr2 = r;
	}
      }
    }
    if (num_cards == 3) {
      // No kicker
      return kThreeOfAKind + (three_rank - 2) * 169;
    } else if (num_cards == 4) {
      // Only one kicker
      return kThreeOfAKind + (three_rank - 2) * 169 + (hr1 - 2) * 13;
    } else {
      // Two kickers
      return kThreeOfAKind + (three_rank - 2) * 169 + (hr1 - 2) * 13 +
	(hr2 - 2);
    }
  }
  if (pair2_rank >= 0) {
    int hr1 = -1;
    for (int i = 0; i < num_cards; ++i) {
      int r = ranks_[i];
      if (r != pair_rank && r != pair2_rank && r > hr1) hr1 = r;
    }
    if (num_cards < 5) {
      // No kicker
      return kTwoPair + (pair_rank - 2) * 169 + (pair2_rank - 2) * 13;
    } else {
      // Encode two pair ranks plus kicker
      return kTwoPair + (pair_rank - 2) * 169 + (pair2_rank - 2) * 13 +
	(hr1 - 2);
    }
  }
  if (pair_rank >= 0) {
    int hr1 = -1, hr2 = -1, hr3 = -1;
    for (int i = 0; i < num_cards; ++i) {
      int r = ranks_[i];
      if (r != pair_rank) {
	if (r > hr1) {
	  hr3 = hr2; hr2 = hr1; hr1 = r;
	} else if (r > hr2) {
	  hr3 = hr2; hr2 = r;
	} else if (r > hr3) {
	  hr3 = r;
	}
      }
    }
    if (num_cards == 3) {
      // One kicker
      return kPair + (pair_rank - 2) * 2197 + (hr1 - 2) * 169;
    } else if (num_cards == 4) {
      // Two kickers
      return kPair + (pair_rank - 2) * 2197 + (hr1 - 2) * 169 + (hr2 - 2) * 13;
    } else {
      // Three kickers
      return kPair + (pair_rank - 2) * 2197 + (hr1 - 2) * 169 + (hr2 - 2) * 13 +
	(hr3 - 2);
    }
  }

  int hr1 = -1, hr2 = -1, hr3 = -1, hr4 = -1, hr5 = -1;
  for (int i = 0; i < num_cards; ++i) {
    int r = ranks_[i];
    if (r > hr1) {
      hr5 = hr4; hr4 = hr3; hr3 = hr2; hr2 = hr1; hr1 = r;
    } else if (r > hr2) {
      hr5 = hr4; hr4 = hr3; hr3 = hr2; hr2 = r;
    } else if (r > hr3) {
      hr5 = hr4; hr4 = hr3; hr3 = r;
    } else if (r > hr4) {
      hr5 = hr4; hr4 = r;
    } else if (r > hr5) {
      hr5 = r;
    }
  }
  if (num_cards == 3) {
    // Encode top three ranks
    return kNoPair + (hr1 - 2) * 28561 + (hr2 - 2) * 2197 + (hr3 - 2) * 169;
  } else if (num_cards == 4) {
    // Encode top four ranks
    return kNoPair + (hr1 - 2) * 28561 + (hr2 - 2) * 2197 + (hr3 - 2) * 169 +
      (hr4 - 2) * 13;
  } else {
    // Encode top five ranks
    return kNoPair + (hr1 - 2) * 28561 + (hr2 - 2) * 2197 + (hr3 - 2) * 169 +
      (hr4 - 2) * 13 + (hr5 - 2);
  }
}
