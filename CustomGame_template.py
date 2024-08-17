def config(self):
    self.n_players = config['n_players']
    self.min_bet = config['min_bet']
    self.max_bet = config['max_bet']
    self.suit = config['suit']
    self.suit_have_rank = config['suit_have_rank']
    self.value = [str(v) for v in config['value']]
    self.card_combinations_rank = config['card_combinations_rank']

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

config['blind_code']

config['dealx_code']
        
def switch(self, switch_indices, player_id):
    for i in switch_indices:
        self.players[player_id].hole[i] = self.deck.pop()

def bet_done(self, wait_to_bet):
    all_bet = [self.players[p].bet for p in self.get_unfold_players()]
    if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
        return True
    return False

config['flopx_code']
    
def set_flow(self):
    self.flow = config['flow']