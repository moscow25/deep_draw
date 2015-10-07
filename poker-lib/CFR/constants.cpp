#include <stdio.h>
#include <stdlib.h>

#include "constants.h"

const char *g_data_root = "/Users/kolya/Desktop/ML/learning/open-pure-cfr-buckets/simulation/data/"; // "/data3/nikolai";

unsigned int NumCardsForStreet(unsigned int s) {
  if (s == 0)      return 2;
  else if (s == 1) return 3;
  else if (s == 2) return 1;
  else if (s == 3) return 1;
  fprintf(stderr, "Unexpected street: %i\n", s);
  exit(-1);
}

unsigned int NumCardsInDeck(void) {
  return 52;
}

unsigned int NumHoleCardPairs(unsigned int s) {
  if (s == 3) {
    return 1081;
  } else if (s == 2) {
    return 1128;
  } else if (s == 1) {
    return 1176;
  } else { // s == 0
    return 1326;
  }
}

