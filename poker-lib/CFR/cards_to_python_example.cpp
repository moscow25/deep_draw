//  Copyright Nikolai Yakovenko, PokerPoker LLC 2015
//
// Basic examples that to link poker canonical forms and other C++/CFR code to Python. 

#include <stdio.h>
#include <stdlib.h>

// Poker classes we need for this example
#include "canonical.h"
#include "cards.h"
#include "constants.h"

// Include other boost/python libraries as need.
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



// Can we get this basic loop to run, when called from Python?
// Can we pass a variable?
// Can we get a variable back?
void loop() {
  Card raw_cards[3], canon_cards[3];
  for (Card hi = kMinCard + 2; hi <= kMaxCard; ++hi) {
    raw_cards[0] = hi;
    for (Card mid = kMinCard + 1; mid < hi; ++mid) {
      raw_cards[1] = mid;
      for (Card lo = kMinCard; lo < mid; ++lo) {
	raw_cards[2] = lo;
	CanonicalizeCards(raw_cards, 1, 1, canon_cards);
	OutputThreeCards(canon_cards);
	printf("\n");
      }
    }
  }
}

// OK, now see if we can get canonical form for arbitrary group of cards from string?
// Mimmicks the operations of CanonicalizeCards(raw_cards, low, high, canon_cards);
std::string canonical_board(const std::string& string_cards, int low, int high) {
  int length = string_cards.size()/2; // number of cards, not number of characters...
  Card raw_cards[length], canon_cards[length];

  // Convert string into Card array. Library functions require char* from string first
  char char_cards[length*2];
  for (int i=0;i<length*2;i++) {
    char c = string_cards[i];
    char_cards[i] = c;
  }
  for (int i=0;i<length;i++) {
    Card c = ParseCard(char_cards + i*2);
    raw_cards[i] = c;
  }

  // OutputNCards(raw_cards, length); printf("\n");
  CanonicalizeCards(raw_cards, low, high, canon_cards);
  // OutputNCards(canon_cards, length); printf("\n");

  // Now convert back to string...
  string string_canonical = "";
  for (int i=0;i<length;i++) {
    std::string card;
    CardName(canon_cards[i], &card);
    string_canonical += card;
  }
  return string_canonical;
}


BOOST_PYTHON_MODULE(cards_to_python_ext)
{
  using namespace boost::python;
  def("loop", loop);
  def("canonical_board", canonical_board);
}
