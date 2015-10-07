#include <stdio.h>
#include <stdlib.h>

#include "canonical.h"
#include "cards.h"
#include "constants.h"

// There are 13 possible ranks (2...A).  For a given street we encode the
// suit with 13 bits indicating whether that rank is present in the given
// suit.
static unsigned int EncodeRanksOfRawSuitForStreet(unsigned int raw_suit,
						  unsigned int min_street,
						  Card *raw_cards,
						  unsigned int street) {
  unsigned int index = 0;
  for (unsigned int s = min_street; s < street; ++s) {
    index += NumCardsForStreet(s);
  }
  unsigned int num_cards_for_street = NumCardsForStreet(street);
  unsigned int street_code = 0;
  for (unsigned int i = index; i < index + num_cards_for_street; ++i) {
    Card raw_card = raw_cards[i];
    unsigned int this_raw_suit = Suit(raw_card);
    if (this_raw_suit == raw_suit) {
      unsigned int rank = Rank(raw_card);
      street_code |= (1 << (rank - 2));
    }
  }
  return street_code;
}

// Each street is encoded separately.  Then the codes for each street are
// concatenated into an overall code.  The most significant bits are dedicated
// to the flop, the next significant to the turn, etc., with the least
// significant dedicated to the hole cards.
static unsigned long long int EncodeRanksOfRawSuit(unsigned int raw_suit,
						   Card *raw_cards,
						   unsigned int min_street,
						   unsigned int max_street) {
  unsigned long long int code = 0;
  for (unsigned int s = min_street; s <= max_street; ++s) {
    unsigned long long int street_code =
      EncodeRanksOfRawSuitForStreet(raw_suit, min_street, raw_cards, s);
    if (s == 0) {
      code |= street_code;
    } else {
      code |= street_code << (16 * (max_street + 1 - s));
    }
  }
  return code;
}

static void Sort3(Card *canon_street_cards) {
  // Sort the three cards
  if (canon_street_cards[0] > canon_street_cards[1]) {
    if (canon_street_cards[0] > canon_street_cards[2]) {
      if (canon_street_cards[1] > canon_street_cards[2]) {
	// 0 > 1 > 2
	// Do nothing
      } else {
	// 0 > 2 > 1
	Card temp = canon_street_cards[1];
	canon_street_cards[1] = canon_street_cards[2];
	canon_street_cards[2] = temp;
      }
    } else {
      // 2 > 0 > 1
      Card old0 = canon_street_cards[0];
      Card old1 = canon_street_cards[1];
      canon_street_cards[0] = canon_street_cards[2];
      canon_street_cards[1] = old0;
      canon_street_cards[2] = old1;
    }
  } else {
    // card1 > card0
    if (canon_street_cards[0] > canon_street_cards[2]) {
      // 1 > 0 > 2
      Card old0 = canon_street_cards[0];
      canon_street_cards[0] = canon_street_cards[1];
      canon_street_cards[1] = old0;
    } else {
      // card2 > card0
      Card old2 = canon_street_cards[2];
      canon_street_cards[2] = canon_street_cards[0];
      if (canon_street_cards[1] > old2) {
	// 1 > 2 > 0
	canon_street_cards[0] = canon_street_cards[1];
	canon_street_cards[1] = old2;
      } else {
	// 2 > 1 > 0
	canon_street_cards[0] = old2;
      }
    }
  }
}

void CanonicalizeCards(Card *raw_cards, unsigned int min_street,
		       unsigned int max_street, Card *canon_cards,
		       unsigned int *suit_mapping) {
  unsigned long long int suit_codes[4];
  for (unsigned int s = 0; s < 4; ++s) {
    suit_codes[s] = EncodeRanksOfRawSuit(s, raw_cards, min_street, max_street);
  }
  unsigned int sorted_suits[4];
  bool used[4];
  for (unsigned int s = 0; s < 4; ++s) used[s] = false;
  for (unsigned int pos = 0; pos < 4; ++pos) {
    unsigned int best_s = kMaxUInt;
    unsigned long long int best_rank_code = 0;
    for (unsigned int s = 0; s < 4; ++s) {
      if (used[s]) continue;
      unsigned long long int rank_code = suit_codes[s];
      if (best_s == kMaxUInt || rank_code > best_rank_code) {
	best_rank_code = rank_code;
	best_s = s;
      }
    }
    sorted_suits[pos] = best_s;
    used[best_s] = true;
  }

  for (unsigned int i = 0; i < 4; ++i) {
    unsigned int raw_suit = sorted_suits[i];
    suit_mapping[raw_suit] = 3 - i;
  }

  // Canonicalize the cards.  Also sort each street's cards from high to low.
  unsigned int index = 0;
  for (unsigned int street = min_street; street <= max_street; ++street) {
    unsigned int num_cards_for_street = NumCardsForStreet(street);
    Card *canon_street_cards = new Card[num_cards_for_street];
    for (unsigned int i = 0; i < num_cards_for_street; ++i) {
      Card raw_card = raw_cards[index + i];
      canon_street_cards[i] = MakeCard(Rank(raw_card),
				       suit_mapping[Suit(raw_card)]);
    }
    if (num_cards_for_street == 1) {
      // Nothing to do
    } else if (num_cards_for_street == 2) {
      if (canon_street_cards[1] > canon_street_cards[0]) {
	Card temp = canon_street_cards[0];
	canon_street_cards[0] = canon_street_cards[1];
	canon_street_cards[1] = temp;
      }
    } else if (num_cards_for_street == 3) {
      Sort3(canon_street_cards);
    } else {
      fprintf(stderr, "%i cards on street %i not supported\n", street,
	      num_cards_for_street);
      exit(-1);
    }
    for (unsigned int i = 0; i < num_cards_for_street; ++i) {
      canon_cards[index + i] = canon_street_cards[i];
    }
    delete [] canon_street_cards;
    index += num_cards_for_street;
  }
}

void CanonicalizeCards(Card *raw_cards, unsigned int min_street,
		       unsigned int max_street, Card *canon_cards) {
  unsigned int suit_mapping[4];
  CanonicalizeCards(raw_cards, min_street, max_street, canon_cards,
		    suit_mapping);
}

unsigned int EncodeTwoCards(Card *cards) {
  static unsigned int mult0 = kEndCards;
  return cards[0] * mult0 + cards[1];
}

unsigned int EncodeThreeCards(Card *cards) {
  static unsigned int mult1 = kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards;
  return cards[0] * mult0 + cards[1] * mult1 + cards[2];
}

unsigned int EncodeFourCards(Card *cards) {
  static unsigned int mult2 = kEndCards;
  static unsigned int mult1 = kEndCards * kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards * kEndCards;
  return cards[0] * mult0 + cards[1] * mult1 + cards[2] * mult2 + cards[3];
}

unsigned int EncodeFiveCards(Card *cards) {
  static unsigned int mult3 = kEndCards;
  static unsigned int mult2 = kEndCards * kEndCards;
  static unsigned int mult1 = kEndCards * kEndCards * kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards * kEndCards * kEndCards;
  return cards[0] * mult0 + cards[1] * mult1 + cards[2] * mult2 +
    cards[3] * mult3 + cards[4];
}

unsigned int CanonicalCode(Card *canon_cards, unsigned int num_cards) {
  if (num_cards == 5) {
    return EncodeFiveCards(canon_cards);
  } else if (num_cards == 4) {
    return EncodeFourCards(canon_cards);
  } else if (num_cards == 3) {
    return EncodeThreeCards(canon_cards);
  } else if (num_cards == 2) {
    return EncodeTwoCards(canon_cards);
  } else if (num_cards == 1) {
    return canon_cards[0];
  } else {
    fprintf(stderr, "CanonCode: %i cards not supported\n", num_cards);
    exit(-1);
#if 0
    unsigned long long int multipliers[7];
    unsigned long long int mult = 1;
    for (unsigned int i = 0; i < num_cards; ++i) {
      multipliers[num_cards - i - 1] = mult;
      mult *= kEndCards;
    }
    unsigned long long int code = 0;
    for (unsigned int i = 0; i < num_cards; ++i) {
      code += canon_cards[i] * multipliers[i];
    }
    return code;
#endif
  }
}

void DecodeTwoCards(unsigned int canon_code, Card *cards) {
  static unsigned int mult0 = kEndCards;
  cards[0] = canon_code / mult0;
  cards[1] = canon_code % mult0;
}

void DecodeThreeCards(unsigned int canon_code, Card *cards) {
  static unsigned int mult1 = kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards;
  cards[0] = canon_code / mult0;
  unsigned int rem = canon_code % mult0;
  cards[1] = rem / mult1;
  cards[2] = rem % mult1;
}

void DecodeFourCards(unsigned int canon_code, Card *cards) {
  static unsigned int mult2 = kEndCards;
  static unsigned int mult1 = kEndCards * kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards * kEndCards;
  cards[0] = canon_code / mult0;
  unsigned int rem1 = canon_code % mult0;
  cards[1] = rem1 / mult1;
  unsigned int rem2 = rem1 % mult1;
  cards[2] = rem2 / mult2;
  cards[3] = rem2 % mult2;
}

void DecodeFiveCards(unsigned int canon_code, Card *cards) {
  static unsigned int mult3 = kEndCards;
  static unsigned int mult2 = kEndCards * kEndCards;
  static unsigned int mult1 = kEndCards * kEndCards * kEndCards;
  static unsigned int mult0 = kEndCards * kEndCards * kEndCards * kEndCards;
  cards[0] = canon_code / mult0;
  unsigned int rem1 = canon_code % mult0;
  cards[1] = rem1 / mult1;
  unsigned int rem2 = rem1 % mult1;
  cards[2] = rem2 / mult2;
  unsigned int rem3 = rem2 % mult2;
  cards[3] = rem3 / mult3;
  cards[4] = rem3 % mult3;
}
