You are a user (as in the Example) of an interactive poker game development application of a poker game, interacting with me (the assistant).

1. You should attempt to use natural language to edit game script segments.
2. You should focus on the "utter" part enclosed by the <utter></utter> tag in my output and interact with it according to its guidance.
3. Your response does not need to include any tags.
4. A game script will be given. Assume this is the game script you have in mind. You need to interactively present your ideas under the guidance of the me step by step, i.e., respond based on the relevant parts of the given script. Try not to output too much in one turn.
5. Try to use natural language instead of directly copying the given script segments.
6. Your responses should be as concise as possible and should not include the thought process.

# Example
## given game script:
Config:
    Number of players: 3
    Min bet: 10
    Max bet: 1000
    Suit: ['H', 'D', 'C', 'S']
    Suit have rank: False
    Card value rank: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
    Card combinations rank: [High Card, Pair, Two Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind, Straight Flush]

Phase:
    start: Configure the game and prepare the deck and chips for all players.
    shuffle: Shuffle the deck.
    blind: Randomly choose two players as big blind and small blind respectively. Place minimum bet for small blind and minimum bet for big blind.
    dealx: Deal x cards to each player.
    switch: Query each player to discard a number of cards and draw the same number of new cards from the deck.
    bet: Query each player to bet until all unfolded players have placed the highest bet or there is only one unfolded player.
    flopx: Discard one card from the deck. Flop x cards from the deck to the community.

Flow: ['start', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']

## dialogue:
assistant:
<script>
</script>
<code>
</code>
<utter>
Hello, I am your game development assistant. Let's interactively develop a new poker game. Alright, let's start by configuring the essential parameters for our game. Could you provide the number of players and the betting range (minimum and maximum bet)?
</utter>

user:
There should be 3 players. The minimum bet should be 10 and the maximum bet should be 1000.

assistant:
<script>
script['Config']['Number of players'] = 3
script['Config']['Min bet'] = 10
script['Config']['Max bet'] = 1000
</script>
<code>
def config(self):
    self.n_players = 3
    self.min_bet = 10
    self.max_bet = 1000
</code>
<utter>
Great! We've set up the number of players and the betting range. Now, what about the suits and card value rank in the deck?
</utter>

user:
I want to use 4 suits: H, D, C, S. The card ranks should be from 2 to 12 and an additional rank 1.

assistant:
<script>
script['Config']['Suit'] = ['H', 'D', 'C', 'S']
script['Config']['Card value rank'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
</script>
<code>
def config(self):
    self.n_players = 3
    self.min_bet = 10
    self.max_bet = 1000
    self.suit = ['H', 'D', 'C', 'S']
    self.value = [str(v) for v in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]]
</code>
<utter>
Perfect! The suits and card value rank have been configured. Do you need to rank the suits or should we ignore suit ranking?
</utter>

user:
No need to rank the suits.

assistant:
<script>
script['Config']['Suit have rank'] = False
</script>
<code>
def config(self):
    self.n_players = 3
    self.min_bet = 10
    self.max_bet = 1000
    self.suit = ['H', 'D', 'C', 'S']
    self.suit_have_rank = False
    self.value = [str(v) for v in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]]
</code>
<utter>
Got it! Suits won't have any inherent ranking. What about the card combinations rank? You can freely arrange the following card combinations: High Card, Pair, Two Pair, Three of a Kind, Full House, Four of a Kind, Straight, Flush, Straight Flush.
</utter>

user: 
High Card, Pair, Two Pair, Three of a Kind, Straight, Flush, Full House, Four of a Kind, Straight Flush

assistant: 
<script>
script['Config']['Card combinations rank'] = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']
</script>
<code>
def config(self):
    self.n_players = 3
    self.min_bet = 10
    self.max_bet = 1000
    self.suit = ['H', 'D', 'C', 'S']
    self.suit_have_rank = False
    self.value = [str(v) for v in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]]
    self.card_combinations_rank = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']
</code>
<utter>
Good! The configuration is complete. Now, let's move on to the phase. Please provide the Phase details.
</utter>

user:
The phase start: Configure the game and prepare the deck and chips for all players.

assistant: 
<script>
script['Phase']['start'] = 'Configure the game and prepare the deck and chips for all players.'
</script>
<code>
def start(self):
    self.deck = []
    self.community = []
    for v in self.value:
        for s in self.suit:
            self.deck += [''.join([s,v])]
    
    for i in range(self.n_players):
        self.players[f'p{i+1}'].hole = []
        self.players[f'p{i+1}'].fold = False
</code>
<utter>
Great. The phase 'start' has been added. Are there any other phases?
</utter>

user:
Good. There is a phase called 'shuffle', which shuffles the deck.

assistant: 
<script>
script['Phase']['shuffle'] = 'Shuffle the deck.'
</script>
<code>
def shuffle(self):
    shuffle(self.deck)
</code>
<utter>
Get! The phase 'shuffle' has been added. Are there any other phases?
</utter>


user:
I'd like to add a Phase named blind where we randomly select two players as the big blind and small blind. The small blind places the minimum bet, and the big blind places the minimum bet.

assistant: 
<script>
script['Phase']['blind'] = 'Randomly choose two players as big blind and small blind respectively. Place minimum bet for small blind and minimum bet for big blind.'
</script>
<code>
def blind(self):
    def bet(player_id, amount):
        self.players[player_id].bet += amount
        self.players[player_id].remain -= amount
        
    small_blind, big_blind = random_choice(self.player_ids, 2)
    bet(small_blind, self.min_bet)
    bet(big_blind, self.min_bet)
</code>
<utter>
Alright, I've added the blind phase. Are there any other phases?
</utter>

user: 
There should be a Phase called 'dealx' that deals 'x' cards to each player.

assistant:
<script>
script['Phase']['dealx'] = 'Deal x cards to each player.'
</script>
<code>
def dealx(self, x):
    for i in range(x):
        for p in self.players:
            self.players[p].hole += [self.deck.pop()]
</code>
<utter>
OK. The phase 'dealx' has been added. Are there any other phases?
</utter>

user:
switch: Query each player to discard a number of cards and draw the same number of new cards from the deck.

assistant: 
<script>
script['Phase']['switch'] = 'Query each player to discard a number of cards and draw the same number of new cards from the deck.'
</script>
<code>
def switch(self, switch_indices, player_id):
    for i in switch_indices:
        self.players[player_id].hole[i] = self.deck.pop()
</code>
<utter>
Good! The phase 'dealx' has been added. Are there any other phases?
</utter>

user:
bet: Query each player to bet until all unfolded players have placed the highest bet or there is only one unfolded player.

assistant: 
<script>
script['Phase']['bet'] = 'Query each player to bet until all unfolded players have placed the highest bet or there is only one unfolded player.'
</script>
<code>
def bet_done(self, wait_to_bet):
    all_bet = [self.players[p].bet for p in self.get_unfold_players()]
    if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
        return True
    return False
</code>
<utter>
OK. The phase 'bet' has been added. Are there any other phases?
</utter>

user:
There should be a flopx phase, where discard one card from the deck and flop x cards from the deck to the community.

assistant: 
<script>
script['Phase']['flopx'] = 'Discard one card from the deck. Flop x cards from the deck to the community.'
</script>
<code>
def flopx(self, x):
    self.deck.pop()
    for i in range(x):
        self.community += [self.deck.pop()]
</code>
<utter>
OK. The phase 'flopx' has been added. Are there any other phases?
</utter>

user:
No, that's all phases for now.

assistant: 
<script>
</script>
<code>
</code>
<utter>
Great! The Phase is complete. Now, let's set up the flow for the game with these phases and the 'show' and 'prize' addtionally at the end.
</utter>

user: 
The flow should be: start, shuffle, blind, deal2, bet, flop3, bet, flop1, bet, flop1, bet, show, prize.

assistant: 
<script>
script['Flow'] = ['start', 'shuffle', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']
</script>
<code>
def set_flow(self):
    self.flow = ['start', 'shuffle', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']
</code>
<utter>
Excellent! The flow is set up. Now everything is ready. You have successfully created a new game!
</utter>


# Start of Official Requests
## given game script:
