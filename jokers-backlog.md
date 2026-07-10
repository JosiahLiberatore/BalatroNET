# jokers backlog — one line each, no design work until core loop scores a hand

- Jolly/Zany/Mad/Crazy/Droll family: trigger on hand CONTAINS type — read the raw flags vector from scoreHand, not the argmax'd handtype
- To Do List (and exact-hand checks): compare handtype == k — argmax already collapsed correctly
- containment semantics (verified): 4oak contains pair + trips but NOT two pair (needs two distinct ranks); full house DOES contain two pair — scoreHand's >= flags already match Balatro exactly, no fix-up needed
- expose flags as extra return from scoreHand when jokers land — already computed, free
