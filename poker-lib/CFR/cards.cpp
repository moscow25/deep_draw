#include <stdio.h>
#include <stdlib.h>

#include "cards.h"

void OutputRank(unsigned int rank) {
  if (rank < 10) {
    printf("%i", rank);
  } else if (rank == 10) {
    printf("T");
  } else if (rank == 11) {
    printf("J");
  } else if (rank == 12) {
    printf("Q");
  } else if (rank == 13) {
    printf("K");
  } else if (rank == 14) {
    printf("A");
  } else {
    fprintf(stderr, "Illegal rank %i\n", rank);
    exit(-1);
  }
}

void OutputCard(Card card) {
  unsigned int rank = Rank(card);
  unsigned int suit = Suit(card);

  OutputRank(rank);
  switch (suit) {
  case 0:
    printf("c"); break;
  case 1:
    printf("d"); break;
  case 2:
    printf("h"); break;
  case 3:
    printf("s"); break;
  default:
    fprintf(stderr, "Illegal suit\n");
    exit(-1);
  }
}

void CardName(Card c, string *name) {
  *name = "";
  unsigned int rank = Rank(c);
  unsigned int suit = Suit(c);

  if (rank < 10) {
    *name += rank + 48;
  } else if (rank == 10) {
    *name += "T";
  } else if (rank == 11) {
    *name += "J";
  } else if (rank == 12) {
    *name += "Q";
  } else if (rank == 13) {
    *name += "K";
  } else if (rank == 14) {
    *name += "A";
  }
  switch (suit) {
  case 0:
    *name += "c"; break;
  case 1:
    *name += "d"; break;
  case 2:
    *name += "h"; break;
  case 3:
    *name += "s"; break;
  default:
    fprintf(stderr, "Illegal suit\n");
    exit(-1);
  }
}

void OutputTwoCards(Card c1, Card c2) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
}

void OutputTwoCards(Card *cards) {
  OutputTwoCards(cards[0], cards[1]);
}

void OutputThreeCards(Card c1, Card c2, Card c3) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
  printf(" ");
  OutputCard(c3);
}

void OutputThreeCards(Card *cards) {
  OutputThreeCards(cards[0], cards[1], cards[2]);
}

void OutputFourCards(Card c1, Card c2, Card c3, Card c4) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
  printf(" ");
  OutputCard(c3);
  printf(" ");
  OutputCard(c4);
}

void OutputFourCards(Card *cards) {
  OutputFourCards(cards[0], cards[1], cards[2], cards[3]);
}

void OutputFiveCards(Card c1, Card c2, Card c3, Card c4, Card c5) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
  printf(" ");
  OutputCard(c3);
  printf(" ");
  OutputCard(c4);
  printf(" ");
  OutputCard(c5);
}

void OutputFiveCards(Card *cards) {
  OutputFiveCards(cards[0], cards[1], cards[2], cards[3], cards[4]);
}

void OutputSixCards(Card c1, Card c2, Card c3, Card c4, Card c5, Card c6) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
  printf(" ");
  OutputCard(c3);
  printf(" ");
  OutputCard(c4);
  printf(" ");
  OutputCard(c5);
  printf(" ");
  OutputCard(c6);
}

void OutputSixCards(Card *cards) {
  OutputSixCards(cards[0], cards[1], cards[2], cards[3], cards[4], cards[5]);
}

void OutputSevenCards(Card c1, Card c2, Card c3, Card c4, Card c5,
		      Card c6, Card c7) {
  OutputCard(c1);
  printf(" ");
  OutputCard(c2);
  printf(" ");
  OutputCard(c3);
  printf(" ");
  OutputCard(c4);
  printf(" ");
  OutputCard(c5);
  printf(" ");
  OutputCard(c6);
  printf(" ");
  OutputCard(c7);
}

void OutputSevenCards(Card *cards) {
  OutputSevenCards(cards[0], cards[1], cards[2], cards[3], cards[4], cards[5],
		   cards[6]);
}

void OutputNCards(Card *cards, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    if (i > 0) printf(" ");
    OutputCard(cards[i]);
  }
}

Card ParseCard(const char *str) {
  char c = str[0];
  unsigned int rank;
  if (c >= '0' && c <= '9') {
    rank = (c - '0');
  } else if (c == 'A') {
    rank = 14;
  } else if (c == 'K') {
    rank = 13;
  } else if (c == 'Q') {
    rank = 12;
  } else if (c == 'J') {
    rank = 11;
  } else if (c == 'T') {
    rank = 10;
  } else {
    fprintf(stderr, "Couldn't parse card rank\n");
    fprintf(stderr, "Str %s\n", str);
    exit(-1);
  }
  c = str[1];
  if (c == 'c') {
    return MakeCard(rank, 0);
  } else if (c == 'd') {
    return MakeCard(rank, 1);
  } else if (c == 'h') {
    return MakeCard(rank, 2);
  } else if (c == 's') {
    return MakeCard(rank, 3);
  } else {
    fprintf(stderr, "Couldn't parse card suit\n");
    fprintf(stderr, "Str %s\n", str);
    exit(-1);
  }
}

void ParseTwoCards(const char *str, bool space_separated, Card *cards) {
  cards[0] = ParseCard(str);
  if (space_separated) {
    cards[1] = ParseCard(str + 3);
  } else {
    cards[1] = ParseCard(str + 2);
  }
}

void ParseThreeCards(const char *str, bool space_separated, Card *cards) {
  cards[0] = ParseCard(str);
  if (space_separated) {
    cards[1] = ParseCard(str + 3);
    cards[2] = ParseCard(str + 6);
  } else {
    cards[1] = ParseCard(str + 2);
    cards[2] = ParseCard(str + 4);
  }
}

void ParseFiveCards(const char *str, bool space_separated, Card *cards) {
  cards[0] = ParseCard(str);
  if (space_separated) {
    cards[1] = ParseCard(str + 3);
    cards[2] = ParseCard(str + 6);
    cards[3] = ParseCard(str + 8);
    cards[4] = ParseCard(str + 12);
  } else {
    cards[1] = ParseCard(str + 2);
    cards[2] = ParseCard(str + 4);
    cards[3] = ParseCard(str + 6);
    cards[4] = ParseCard(str + 8);
  }
}

bool OnBoard(Card c, Card *board, unsigned int num_board) {
  for (unsigned int i = 0; i < num_board; ++i) if (c == board[i]) return true;
  return false;
}

unsigned int MaxSuit(Card *board, unsigned int num_board) {
  unsigned int max_suit = Suit(board[0]);
  for (unsigned int i = 1; i < num_board; ++i) {
    unsigned int s = Suit(board[i]);
    if (s > max_suit) max_suit = s;
  }
  return max_suit;
}

