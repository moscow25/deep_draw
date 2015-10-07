#ifndef _CANONICAL_H_
#define _CANONICAL_H_

#include "cards.h"

void CanonicalizeCards(Card *raw_cards, unsigned int min_street,
		       unsigned int max_street, Card *canon_cards,
		       unsigned int *suit_mapping);
void CanonicalizeCards(Card *raw_cards, unsigned int min_street,
		       unsigned int max_street, Card *canon_cards);

unsigned int EncodeTwoCards(Card *cards);
unsigned int EncodeThreeCards(Card *cards);
unsigned int EncodeFourCards(Card *cards);
unsigned int EncodeFiveCards(Card *cards);
unsigned int CanonicalCode(Card *canon_cards, unsigned int num_cards);
void DecodeTwoCards(unsigned int canon_code, Card *cards);
void DecodeThreeCards(unsigned int canon_code, Card *cards);
void DecodeFourCards(unsigned int canon_code, Card *cards);
void DecodeFiveCards(unsigned int canon_code, Card *cards);

#endif
