#ifndef _HAND_EVALUATOR_H_
#define _HAND_EVALUATOR_H_

#include "cards.h"

class HandEvaluator {
 public:
  HandEvaluator(void);
  ~HandEvaluator(void);
  int Evaluate(Card *cards, int num_cards);

  static const int kMaxHandVal = 775905;
  static const int kStraightFlush = 775892;
  static const int kQuads = 775723;
  static const int kFullHouse = 775554;
  static const int kFlush = 404261;
  static const int kStraight = 404248;
  static const int kThreeOfAKind = 402051;
  static const int kTwoPair = 399854;
  static const int kPair = 371293;
  static const int kNoPair = 0;

 private:
  int *ranks_;
  int *suits_;
  int *rank_counts_;
  int *suit_counts_;
};

#endif
