Generate new code snippet and the corresponding script segment of a poker game based on the given original code snippet and the corresponding original script segment.

1. Modify the code logic to obtain a new code segment and output the corresponding script segment.
2. The new code snippet is obtained by modifying the original code snippet.
3. Keep the input parameters unchanged, do not introduce new input parameters.
4. The generated new code snippet should not introduce new instance attributes and involved methods such as `self.xxx` or `self.xxx(...)` compared to the original code snippet. The generated new code snippet can only include instance attributes and instance methods involved in the original code snippet. You cannot create new ones. For example, there is a original code snippet below:
def bet_done(self, wait_to_bet):
    all_bet = [self.players[p].bet for p in self.get_unfold_players()]
    if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
        return True
    return False
In this code snippet, the instance attributes and instance methods involved are only `self.players` and `self.get_unfold_players()`. Therefore, in the new code snippet generated from this original code snippet, the instance attributes and instance methods involved should also only be `self.players` and `self.get_unfold_players()`, other created ones such as `self.group`, `self.discard_pile`, `self.burn_pile`, `self.burn_card` are not allowed to be used.
5. Do not use `print` or logging information.
6. The script segment can be seen as a description of the code snippet.
7. Try to be creative and diverse.
8. The output format should follow the original, without any redundant information.


# Example 1
## original code snippet:
def dealx(self, x):
    for i in range(x):
        for p in self.players:
            self.players[p].hole += [self.deck.pop()]

## original script segment:
Phase:
    dealx: Deal x cards to each player.

## new code snippet:
def dealx(self, x):
    for i in range(x):
        for p in self.players:
            self.players[p].hole += [self.deck.pop()]
    shuffle(self.deck)

## new script segment:
Phase:
    dealx: Deal x cards to each player. After all cards have been dealt to all players, shuffle the deck.


# Example 2
## original code snippet:
def set_flow(self):
    self.flow = ['start', 'shuffle', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']

## original script segment:
Flow: ['start', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']

## new code snippet:
def set_flow(self):
    self.flow = ['start', 'shuffle', 'blind', 'deal3', 'bet', 'flop3', 'bet', 'flop1', 'show', 'prize']

## new script segment:
Flow: ['start', 'shuffle', 'blind', 'deal3', 'bet', 'flop3', 'bet', 'flop1', 'show', 'prize']