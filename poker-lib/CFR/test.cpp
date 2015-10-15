#include "canonical.h"
#include "cards.h"

int main(int argc, char *argv[]) {
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
