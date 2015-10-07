#ifndef _CARDS_H_
#define _CARDS_H_

#include <string>

using namespace std;

typedef int Card;

// 14*4+3
const int kMaxCard = 59;
const int kEndCards = 60;
// 2*4+0
const int kMinCard = 8;

#define MakeCard(rank, suit) ((rank) * 4 + suit)
#define Rank(card)           ((unsigned int)(card / 4))
#define Suit(card)           ((unsigned int)(card % 4))

void OutputRank(unsigned int rank);
void OutputCard(Card card);
void CardName(Card c, string *name);
void OutputTwoCards(Card c1, Card c2);
void OutputTwoCards(Card *cards);
void OutputThreeCards(Card c1, Card c2, Card c3);
void OutputThreeCards(Card *cards);
void OutputFourCards(Card c1, Card c2, Card c3, Card c4);
void OutputFourCards(Card *cards);
void OutputFiveCards(Card c1, Card c2, Card c3, Card c4, Card c5);
void OutputFiveCards(Card *cards);
void OutputSixCards(Card c1, Card c2, Card c3, Card c4, Card c5, Card c6);
void OutputSixCards(Card *cards);
void OutputSevenCards(Card c1, Card c2, Card c3, Card c4, Card c5,
		      Card c6, Card c7);
void OutputSevenCards(Card *cards);
void OutputNCards(Card *cards, unsigned int n);
Card ParseCard(const char *str);
void ParseTwoCards(const char *str, bool space_separated, Card *cards);
void ParseThreeCards(const char *str, bool space_separated, Card *cards);
void ParseFiveCards(const char *str, bool space_separated, Card *cards);
bool OnBoard(Card c, Card *board, unsigned int num_board);
unsigned int MaxSuit(Card *board, unsigned int num_board);

#endif
