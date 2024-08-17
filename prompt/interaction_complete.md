Generate a dialogue between a user and an assistant based on the following rules and given script segment and code snippet.

1. The user edits game script segments using natural language during interactions with the assistant.
2. The assistant interacts with the user to achieve interactive game development. The assistant guides the user in editing game script segments, generates corresponding code snippets, and interacts with the user through dialogue.
3. Each turn of the assistant's output should include three processes: "script", "code", and "utter", corresponding to three blocks:  <script></script>, <code></code>, <utter></utter>. Formally, these three blocks must exist, even if the content is empty.
4. The 'script' process: The assistant generates the game script segment based on the user's input of the last turn. Return modifications to the script as changes, rather than returning the entire script. The script is a Python dict, so you can use simple Python code to represent modifications to it, such as: script['xxx'] = 'xxx'. The 'script' process should be enclosed using '<script>' tag.
5. The 'code' process: The assistant generates the corresponding Python code snippet based on the game script segment from the 'script' process. The complete code is a CustomGame class that inherits from GameBase class, but only the methods related to the given script segment need to be generated. The 'code' process should be enclosed using '<code>' tag.
6. The 'utter' process: The assistant interacts with the user, including responding to the user's input of the last turn, summarizing the results of the current turn, and guiding the user to continue with the current turn of interaction. The 'utter' process should be enclosed using '<utter>' tag.
7. The script segment and code snippet have already been provided. You need to randomly distribute them across multiple turns and generate an interactive dialogue between the assistant and the user. This means the assistant guides the user step by step to complete this game script segment. In a single turn of dialogue, the user's input should not contain too much information. If a large input is required, it should be divided into multiple turns.
8. In the assistant's 'script' and 'code' process, use the given script segment and code snippet; do not write your own.
9. The dialogue must cover and only cover all the given script segment, and no other content should appear.
10. The assistant does not know about the existence of the script segment in the dialogue and needs to obtain it from the user's input.
11. The given script segment and code snippet are essentially an outline of the plot development. The assistant's 'script' and 'code' process must be entirely derived from or inferred from the user's input. The user's input should be more natural language-based and not a direct copy of the given script segement.
12. In the first turn, the 'script' and 'code' process of the assistant should be empty because the user has not yet input a game script segment. In the first turn, the assistant should greet the user and start guiding them. In the end, after the user has completed the entire script under the assistant's guidance, the assistant should convey to the user that the game development is complete.
13. The assistant should guide the user step by step along a specific line to complete each part of the game script:
Config->Phase->Flow
14. Output format requirements: Use the following format. There should be no empty lines within the user or assistant utterances.
Example format:
assistant: 
<script>
...
</script>
<code>
...
</code>
<utter>
...
</utter>

user: 
...


# Example:
## script segment:
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

## code snippet:    
def config(self):
    self.n_players = 3
    self.min_bet = 10
    self.max_bet = 1000
    self.suit = ['H', 'D', 'C', 'S']
    self.suit_have_rank = False
    self.value = [str(v) for v in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]]
    self.card_combinations_rank = ['High Card', 'Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']

def start(self):
    self.deck = []
    self.community = []
    for v in self.value:
        for s in self.suit:
            self.deck += [''.join([s,v])]
    
    for i in range(self.n_players):
        self.players[f'p{i+1}'].hole = []
        self.players[f'p{i+1}'].fold = False

def shuffle(self):
    shuffle(self.deck)

def blind(self):
    def bet(player_id, amount):
        self.players[player_id].bet += amount
        self.players[player_id].remain -= amount
        
    small_blind, big_blind = random_choice(self.player_ids, 2)
    bet(small_blind, self.min_bet)
    bet(big_blind, self.min_bet)

def dealx(self, x):
    for i in range(x):
        for p in self.players:
            self.players[p].hole += [self.deck.pop()]
        
def switch(self, switch_indices, player_id):
    for i in switch_indices:
        self.players[player_id].hole[i] = self.deck.pop()

def bet_done(self, wait_to_bet):
    all_bet = [self.players[p].bet for p in self.get_unfold_players()]
    if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
        return True
    return False

def flopx(self, x):
    self.deck.pop()
    for i in range(x):
        self.community += [self.deck.pop()]
    
def set_flow(self):
    self.flow = ['start', 'shuffle', 'blind', 'deal2', 'bet', 'flop3', 'bet', 'flop1', 'bet', 'flop1', 'bet', 'show', 'prize']
    
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


