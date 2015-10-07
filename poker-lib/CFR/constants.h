#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

static const unsigned int kMaxUnsignedShort = 65535U;
static const int kMaxInt = 2147483647;
static const int kMinInt = -2147483648;
static const unsigned int kMaxUInt = 4294967295U;
extern const char *g_data_root;

unsigned int NumCardsForStreet(unsigned int s);
unsigned int NumCardsInDeck(void);
unsigned int NumHoleCardPairs(unsigned int s);

#endif
