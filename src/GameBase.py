import random
from itertools import combinations, product
from collections import Counter
#from core import *  
import logging
import json
from abc import ABC, abstractmethod
import inspect

def shuffle(x):
    random.shuffle(x)

def random_choice(x,k=1):
    return random.choice(x) if k==1 else random.sample(x,k=k)

class HandWrapper:
    def __init__(self,hand):
        self.hand=list(hand)
        self.suit_dict={}
        for c in hand:
            suit=c[0]
            self.suit_dict[suit]=self.suit_dict[suit] + [c] if suit in self.suit_dict else [c]
        self.value_dict={}
        for c in hand:
            value=c[1:]
            self.value_dict[value]=self.value_dict[value] + [c] if value in self.value_dict else [c]
    
    def __repr__(self):
        return str(self.hand)
    
    def common(self,x):
        ret=[]
        for _,v in self.value_dict.items():
            if len(v)==x:
                ret+=[v]
        return ret
    
    def most_common(self):
        x = max([len(v) for v in self.value_dict.values()])
        return self.common(x)
    
    def max(self,get_card_rank=None):
        if not get_card_rank:
            def get_card_rank(card):
                return int(card[1:])
        return max(self.hand,key=lambda x:get_card_rank(x))
    
    def sort(self,get_card_rank=None,reverse=True):
        if not get_card_rank:
            def get_card_rank(card):
                return int(card[1:])
        self.hand.sort(key=lambda x:get_card_rank(x),reverse=reverse)
    
    def sort_most_common(self,get_card_rank=None,reverse=True):
        if not get_card_rank:
            def get_card_rank(card):
                return int(card[1:])
        max_count=len(self.most_common()[0])
        common_chunks=[]
        for count in range(max_count,0,-1):
            common=self.common(count)
            common.sort(key=lambda x:get_card_rank(max(x,key=lambda x:get_card_rank(x))),reverse=reverse)
            common_chunks+=common
        self.hand=[c for chunk in common_chunks for c in chunk]

    def cmp_most_common(self,other,get_card_rank=None,reverse=True):
        if not get_card_rank:
            def get_card_rank(card):
                return int(card[1:])
        self.sort_most_common(get_card_rank=get_card_rank,reverse=reverse)
        other.sort_most_common(get_card_rank=get_card_rank,reverse=reverse)
        for c1,c2 in zip(self.hand,other.hand):
            if get_card_rank(c1)>get_card_rank(c2):
                return 1
            if get_card_rank(c1)<get_card_rank(c2):
                return -1
        return 0
    
    def cmp(self,other,get_card_rank=None,reverse=True):
        if not get_card_rank:
            def get_card_rank(card):
                return int(card[1:])
        self.sort(get_card_rank=get_card_rank,reverse=reverse)
        other.sort(get_card_rank=get_card_rank,reverse=reverse)
        for c1,c2 in zip(self.hand,other.hand):
            if get_card_rank(c1)>get_card_rank(c2):
                return 1
            if get_card_rank(c1)<get_card_rank(c2):
                return -1
        return 0

class Player:
    def __init__(self, bet, remain, hole, fold) -> None:
        #self.chips = chips
        self.bet = bet
        self.remain = remain
        self.hole = hole
        self.fold = fold

class GameBase:
    def __init__(self):
        self.config()
        self.set_flow()
        
        self.players = {}
        
        for i in range(self.n_players):
            self.players[f'p{i+1}'] = Player(0, self.max_bet, [], False)
            
        self.player_ids = list(self.players.keys())
        
        self.set_logger()
        
    def config(self):
        pass
    
    def set_flow(self):
        pass
        
    def set_logger(self):
        open('game_log.log', 'w').close()
        
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG) 
        fh = logging.FileHandler('game_log.log', mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger
        
    def log(self, message):
        state = {
            'flow': [self.flow[i] for i in range(self.j + 1)],
            'deck': self.deck,
            'community': self.community,
            'players': [f'{p}: {item.bet}/{item.remain}/{item.hole}/{item.fold}' for p, item in self.players.items()]
        }
        
        str_state = 'state:\n' + '\n'.join([f'{key}: {", ".join(value)}' for key, value in state.items()])
        str_message = 'message:\n' + ', '.join([f'{key}: {value}' for key, value in message.items()])
        
        self.logger.info('')
        self.logger.info(str_state)
        self.logger.info(str_message)
        '''print('')
        print(str_state)
        print(str_message)'''
        
    
    def is_flush(self, cards):
        suits=[c[0] for c in cards]
        suit_counts=Counter(suits)
        return any([v>=5 for v in suit_counts.values()])

    def is_straight(self, cards,get_value_rank):
        values=[get_value_rank(c[1:]) for c in cards]
        return len(set(values))==5 and max(values)-min(values)==4

    def is_straightflush(self, cards,get_value_rank):
        return self.is_flush(cards) and self.is_straight(cards,get_value_rank)

    def is_fourofakind(self, cards):
        values=[c[1:] for c in cards]
        value_counts=Counter(values)
        return 4 in value_counts.values()

    def is_fullhouse(self, cards):
        values=[c[1:] for c in cards]
        value_counts=Counter(values)
        return set(value_counts.values())=={2,3}

    def is_threeofakind(self, cards):
        values=[c[1:] for c in cards]
        value_counts=Counter(values)
        return 3 in value_counts.values()

    def is_twopair(self, cards):
        values=[c[1:] for c in cards]
        value_counts=Counter(values)
        return list(value_counts.values()).count(2)==2

    def is_pair(self, cards):
        values=[c[1:] for c in cards]
        value_counts=Counter(values)
        return 2 in value_counts.values()


    def get_suit_rank(self, x):
        return {s:i+1 for i, s in enumerate(self.suit)}[x] if self.suit_have_rank else 1

    
    def get_value_rank(self, value):
        return {v:i+1 for i, v in enumerate(self.value)}[value]

    def get_card_rank(self, card):
            suit,value=card[0],card[1:]
            return self.get_value_rank(value) + self.get_suit_rank(suit)/10

    def get_combination_rank(self, x):
        return {c:i+1 for i, c in enumerate(self.card_combinations_rank)}[x]

    def cmp_hand(self, hand1, hand2):
        if hand1[1]!=hand2[1]:
            return 1 if self.get_combination_rank(hand1[1])>self.get_combination_rank(hand2[1]) else -1
        else:
            return HandWrapper(hand1[0]).cmp_most_common(HandWrapper(hand2[0]),self.get_card_rank)

    def evaluate(self, hole_cards, community_cards):
        all_possible_hands=combinations(hole_cards+community_cards,5)

        best=(None,None)
        for hand in all_possible_hands:
            if self.is_straightflush(hand,self.get_value_rank):
                combination='Straight Flush'
            elif self.is_fourofakind(hand):
                combination='Four of a Kind'
            elif self.is_fullhouse(hand):
                combination='Full House'
            elif self.is_flush(hand):
                combination='Flush'
            elif self.is_straight(hand,self.get_value_rank):
                combination='Straight'
            elif self.is_threeofakind(hand):
                combination='Three of a Kind'
            elif self.is_twopair(hand):
                combination='Two Pair'
            elif self.is_pair(hand):
                combination='Pair'
            else:
                combination='High Card'
            if not best[0]:
                best=(hand,combination)
                continue
            if self.cmp_hand((hand,combination),best)==1:
                best=(hand,combination)

        return best

    def get_winners(self, show):
        winners=[]
        best=(None,None)
        for p in show:
            if not best[0] or self.cmp_hand(show[p],best)==1:
                best=show[p]
                winners=[p]
            elif self.cmp_hand(show[p],best)==0:
                winners+=[p]
        return winners
    
    def get_parameter_count(self, method_name):
        method = getattr(self, method_name)
        signature = inspect.signature(method)
        parameters = signature.parameters
        return len(parameters)

    def get_unfold_players(self):
        return [p for p in self.players if not self.players[p].fold]
    
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
    
    def flopx(self, x):
        self.deck.pop()
        for i in range(x):
            self.community += [self.deck.pop()]
            
    def switch(self, switch_indices, player_id):
        for i in switch_indices:
            self.players[player_id].hole[i] = self.deck.pop()
    
    def bet_done(self, wait_to_bet):
        all_bet = [self.players[p].bet for p in self.get_unfold_players()]
        if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
            return True
        return False
    
    def show(self):
        show = {}
            
        for p in self.player_ids:
            if self.players[p].fold:
                continue
            show[p] = self.evaluate(self.players[p].hole, self.community) if self.get_parameter_count('evaluate') == 2 else self.evaluate(self.players[p].hole)
            
        self.winners = self.get_winners(show)
            
        self.log({'src':'engine','trg':'all','content':f'Showdown: {show}; winners: {", ".join(self.winners)}'})
    
    def prize(self):
        total_bonus = 0
        for p in self.player_ids:
            total_bonus += self.players[p].bet
        n_winners = len(self.winners)
        bonus = total_bonus // n_winners
        for w in self.winners:
            self.players[w].remain += bonus

        for p in self.player_ids:
            self.players[p].bet = 0
            self.players[p].fold = False

        self.log({'src':'engine','trg':'all','content':'End of the round.'})
    
    def step(self, j, auto):
        self.j = j
        flow = self.flow[j]
        
        if flow == "start":
            self.start()
            self.log({'src':'engine','trg':'all','content':'Start of a new round.'})
        elif flow == "shuffle":
            self.shuffle()
            self.log({'src':'engine','trg':'all','content':'Shuffled.'})
        elif flow == "blind":
            self.blind()
            self.log({'src':'engine','trg':'all','content':f'Placing of blind bet.'})
        elif flow.startswith("deal"):
            x = eval(flow.split("deal")[-1])
            self.dealx(x)
            self.log({'src':'engine','trg':'all','content':f'Dealing {x} cards to each player.'})
        elif flow.startswith("flop"):
            x = eval(flow.split("flop")[-1])
            self.flopx(x)
            self.log({'src':'engine','trg':'all','content':f'Flopping {x} cards to from the deck.'})
        elif flow == "switch":
            if auto:
                for p in self.players:
                    self.log({'src':'engine','trg':f'{p}','content':'It is your turn to switch.'})
                    x = random.choice(range(len(self.players[p].hole)))
                    x = 0 if x > len(self.deck) else x
                    
                    if x > 0:
                        pop_indices = random.sample(range(len(self.players[p].hole)), x)
                        self.log({"src": p, "trg": "engine", "content": "Switch {}.".format(" ".join([str(i) for i in pop_indices]))})
                        self.switch(pop_indices, p)
                    else:
                        self.log({"src": p, "trg": "engine", "content": "Skip."})
            else:
                for p in self.players:
                    self.log({'src':'engine','trg':f'{p}','content':'It is your turn to switch.'})
                    input_pop_indices = input(f"{p} input(enter a list separated by commas ','): ").strip()
                    if input_pop_indices:
                        pop_indices = [eval(item) for item in input_pop_indices.split(',')]
                        self.log({"src": p, "trg": "engine", "content": "Switch {}.".format(" ".join([str(i) for i in pop_indices]))})
                        self.switch(pop_indices, p)
                    else:
                        self.log({"src": p, "trg": "engine", "content": "Skip."})
        elif flow == "bet":
            wait_to_bet = self.get_unfold_players()
            if len(wait_to_bet) == 1:
                self.log({'src':'engine','trg':'all','content':'Bet is done.'})
                return
            
            message = {'src':'engine','trg':f'{wait_to_bet[0]}','content':'It is your turn to bet.'}
            self.log(message)
            
            bar = self.min_bet
            i = 0
            while 1:
                player_id = self.player_ids[i]
                
                if self.players[player_id].fold:
                    i = (i + 1) % self.n_players
                    continue
                
                if auto:
                    rnd = 0.2# if wait_to_bet else 0
                    if random.random() < rnd: # Raise to x
                        amount = bar + random.randint(1, 10) * self.min_bet - self.players[player_id].bet
                        
                        self.players[player_id].bet += amount
                        self.players[player_id].remain -= amount
                        
                        bar = self.players[player_id].bet
                        
                        act = "Raise to {}".format(bar)
                    elif random.random() < 0.1: # Fold
                        act = "Fold"
                
                        self.players[player_id].fold = True        
                    else: # Call/Check
                        if bar > self.players[player_id].bet:
                            amount = bar - self.players[player_id].bet
                            
                            self.players[player_id].bet += amount
                            self.players[player_id].remain -= amount
                            
                            act = "Call"
                        else:
                            act = "Check"
                else:
                    act = input(f"{player_id} input(enter 'c' for call/check, 'f' for fold, 'r x' for raise to x): ").strip()
                    if act[0] == 'r': # Raise to x
                        x = eval(act.split(' ')[-1])
                        amount = x - self.players[player_id].bet
                        
                        self.players[player_id].bet += amount
                        self.players[player_id].remain -= amount
                        
                        bar = self.players[player_id].bet
                        
                        act = "Raise to {}".format(bar)
                    elif act[0] == 'f': # Fold
                        act = "Fold"
                
                        self.players[player_id].fold = True        
                    elif act[0] == 'c': # Call/Check
                        if bar > self.players[player_id].bet:
                            amount = bar - self.players[player_id].bet
                            
                            self.players[player_id].bet += amount
                            self.players[player_id].remain -= amount
                            
                            act = "Call"
                        else:
                            act = "Check"
                    else:
                        raise Exception("Invalid act: " + act)

                self.log({"src": player_id, "trg": "engine", "content": "{}.".format(act)})

                i = (i + 1) % self.n_players
                
                player_id = self.player_ids[i]
                while self.players[player_id].fold:
                    i = (i + 1) % self.n_players
                    player_id = self.player_ids[i]
                
                if wait_to_bet:
                    wait_to_bet.pop(0)
                if len(self.get_unfold_players()) == 1 or self.bet_done(wait_to_bet):
                    self.log({'src':'engine','trg':'all','content':'Bet is done.'})
                else:
                    self.log({'src':'engine','trg':f'{player_id}','content':'It is your turn to bet.'})
                
                if len(self.get_unfold_players()) == 1 or self.bet_done(wait_to_bet):
                    break
        elif flow == "show":
            self.show()
            
        elif flow == "prize":
            self.prize()
        else:
            pass
            #raise Exception("Invalid flow: " + flow)
    
    def play(self, auto):
        for j in range(len(self.flow)):
            self.step(j, auto)
        self.state = {
            'flow': [self.flow[i] for i in range(self.j + 1)],
            'deck': self.deck,
            'community': self.community,
            'players': [f'{p}: {item.bet}/{item.remain}/{item.hole}/{item.fold}' for p, item in self.players.items()]
        }
            


if __name__ == '__main__':
    config = {
        "n_players": random.randint(3, 5),
        "min_bet": random.choice([2, 4, 6, 8, 10]),
        "max_bet": random.choice([500, 800, 1000, 2000, 5000, 10000]),
        "suit": random.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), k=4),
        "value": random.choice([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1], [1, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], [7, 3, 2, 20, 18, 11, 6, 8, 4, 17, 12, 9, 5]]),
        "flow": random.choice([
            ["start", "shuffle", "blind", "deal2", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
            ["start", "shuffle", "deal2", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
            ["start", "shuffle", "blind", "deal4", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
            ["start", "shuffle", "blind", "deal3", "bet", "flop3", "bet", "flop1", "show", "prize"],
            ["start", "shuffle", "deal3", "bet", "flop3", "bet", "flop1", "bet", "flop1", "bet", "show", "prize"],
            ["start", "shuffle", "blind", "deal2", "bet", "flop3", "deal1", "bet", "flop1", "deal1", "bet", "show", "prize"]
        ])
    }
    
    class CustomGame(GameBase):        
        def config(self):
            self.n_players = config['n_players']
            self.min_bet = config['min_bet']
            self.max_bet = config['max_bet']
            self.suit = config['suit']
            self.suit_have_rank = False
            self.value = [str(v) for v in config['value']]
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
        
        def flopx(self, x):
            self.deck.pop()
            for i in range(x):
                self.community += [self.deck.pop()]
                
        def switch(self, switch_indices, player_id):
            for i in switch_indices:
                self.players[player_id].hole[i] = self.deck.pop()
        
        def bet_done(self, wait_to_bet):
            all_bet = [self.players[p].bet for p in self.get_unfold_players()]
            if not wait_to_bet and all([b==all_bet[0] for b in all_bet]):
                return True
            return False
            
        def set_flow(self):
            self.flow = config['flow']

    game = CustomGame()
    game.play(auto=True)
    #game.play(auto=True)