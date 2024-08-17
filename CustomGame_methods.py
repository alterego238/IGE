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