"""
mit.py  —  SneakBot v3 (MIT-Architecture Edition)
=======================================================
Implements MIT 2024 architectural concepts adapted for the IIT Pokerbots 2026
Python engine (pkbot framework + eval7).

Key improvements over normal_bid_use.py:
  1. HandRange class — Bayesian range tracking over all 1326 two-card combos,
     updated after every observed opponent action (fold/call/raise).
     Opponent range is used to compute range-weighted equity estimates,
     replacing or augmenting pure Monte-Carlo.

  2. MIT geometric-series bid formula — replaces the VOI heuristic:
       equity_with_bid = ((1 / (1 - equity_diff)) - 1) * pot
       default_bid     = ceil(equity_with_bid * BID_MULTIPLIER=1.8) + 1
     Derived from the infinite geometric series for information value.

  3. OppBidTracker — 4-value opponent bid model (MIT auction.cpp port):
       • abs_min / abs_max  — absolute chip range of past bids
       • pct_min / pct_max  — bid/pot % range
       • is_excessive_bidder — bids near all-in repeatedly
     After ≥ 5 observations the bid adapts to exploit the opponent's pattern.

  4. Range-weighted equity — after opponent actions, equity is blended:
       eq_blend = (1-RANGE_WEIGHT) * mc_equity + RANGE_WEIGHT * range_equity
     where range_equity = eval7 scores weighted by opponent range probability.

  5. All strategies from normal_bid_use.py preserved verbatim — preflop
     equity table, _analyze_opp_card(), draw-denial sizing, 6 threat modes,
     and the 3-way postflop dispatch (_with_info / _vs_info / _neutral).
     MIT improvements are overlaid, not replacing proven logic.

References:
  Vickrey (1961) — Second-Price Auction   Howard (1966) — VOI
  Chen & Ankenman (2006) — Mathematics of Poker
  Bowling et al. (2015) — Solving HU Limit Hold'em   (Science)
  Brown & Sandholm (2017) — Libratus   (Science)
  MIT Pokerbots 2024 C++ source — auction.cpp / range.cpp / cfr.cpp
"""

import random
import math
import itertools
import time
from collections import defaultdict

import eval7

from pkbot.base   import BaseBot, GameInfo
from pkbot.states import PokerState
from pkbot.actions import (ActionFold, ActionCall, ActionCheck,
                            ActionRaise, ActionBid)
from pkbot.runner import run_bot, parse_args

# ─────────────────────────────────────────────────────────────────────────────
# Game constants  (must match engine config)
# ─────────────────────────────────────────────────────────────────────────────
STARTING_STACK = 5_000
BIG_BLIND      = 20
SMALL_BLIND    = 10
NUM_ROUNDS     = 1_000
MC_ITERS       = 600          # Monte-Carlo iterations for postflop equity
MC_ITERS_FAST  = 80           # Used when time_bank < 4 s

# MIT constants
BID_MULTIPLIER  = 1.8         # MIT auction.cpp: default_bid = equity_with_bid×1.8+1
BID_OBS_MIN     = 5           # observations needed before bid exploitation kicks in
WEIGHT_UNIFORM  = 0.05        # range regularization (5% uniform floor, MIT range.cpp)
RANGE_BLEND_W   = 0.40        # weight given to range-equity vs MC-equity blend
RANGE_UPD_EVERY = 1           # update range on every opponent action

# ─────────────────────────────────────────────────────────────────────────────
# Card universe — ordered consistently with eval7
# ─────────────────────────────────────────────────────────────────────────────
_RANKS  = '23456789TJQKA'
_SUITS  = 'cdhs'
_ALL_EVAL7_CARDS = [eval7.Card(r + s) for r in _RANKS for s in _SUITS]  # 52 cards, index 0-51
_CARD_TO_IDX     = {str(c): i for i, c in enumerate(_ALL_EVAL7_CARDS)}

# All 1326 two-card combos as (i, j) with i < j (indices into _ALL_EVAL7_CARDS)
_ALL_COMBOS = list(itertools.combinations(range(52), 2))

# Pre-build card index for fast blocking checks
_RANK_VALUE = {r: i for i, r in enumerate('23456789TJQKA')}


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helpers  (unchanged from normal_bid_use.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_preflop_equity_table() -> dict:
    """
    Exhaustive reference equity table for HU no-limit hold'em (1000 rounds).
    Keys: canonical form ('AA', 'AKs', 'AKo').  Values: win-rate vs random.
    Sources: PokerStove, PokerTracker batch simulation, solver-validated.
    """
    table = {
        # Pocket pairs (equity vs random HU opponent)
        'AA':0.852,'KK':0.824,'QQ':0.800,'JJ':0.775,'TT':0.751,
        '99':0.721,'88':0.691,'77':0.662,'66':0.633,'55':0.603,
        '44':0.570,'33':0.540,'22':0.512,
        # Suited aces
        'AKs':0.670,'AQs':0.661,'AJs':0.654,'ATs':0.647,
        'A9s':0.632,'A8s':0.623,'A7s':0.614,'A6s':0.605,
        'A5s':0.601,'A4s':0.592,'A3s':0.584,'A2s':0.577,
        # Offsuit aces
        'AKo':0.652,'AQo':0.641,'AJo':0.630,'ATo':0.620,
        'A9o':0.604,'A8o':0.595,'A7o':0.585,'A6o':0.576,
        'A5o':0.572,'A4o':0.563,'A3o':0.554,'A2o':0.547,
        # Suited kings
        'KQs':0.634,'KJs':0.626,'KTs':0.618,'K9s':0.607,
        'K8s':0.597,'K7s':0.588,'K6s':0.579,'K5s':0.570,
        'K4s':0.561,'K3s':0.553,'K2s':0.545,
        # Offsuit kings
        'KQo':0.619,'KJo':0.609,'KTo':0.600,'K9o':0.588,
        'K8o':0.578,'K7o':0.569,'K6o':0.559,'K5o':0.551,
        'K4o':0.542,'K3o':0.534,'K2o':0.526,
        # Suited queens
        'QJs':0.619,'QTs':0.611,'Q9s':0.600,'Q8s':0.590,
        'Q7s':0.580,'Q6s':0.571,'Q5s':0.562,'Q4s':0.553,
        'Q3s':0.544,'Q2s':0.536,
        # Offsuit queens
        'QJo':0.602,'QTo':0.593,'Q9o':0.581,'Q8o':0.571,
        'Q7o':0.561,'Q6o':0.552,'Q5o':0.543,'Q4o':0.534,
        'Q3o':0.525,'Q2o':0.517,
        # Suited jacks
        'JTs':0.603,'J9s':0.592,'J8s':0.581,'J7s':0.570,
        'J6s':0.560,'J5s':0.551,'J4s':0.542,'J3s':0.533,'J2s':0.524,
        # Offsuit jacks
        'JTo':0.585,'J9o':0.573,'J8o':0.562,'J7o':0.551,
        'J6o':0.541,'J5o':0.531,'J4o':0.522,'J3o':0.513,'J2o':0.505,
        # Suited tens
        'T9s':0.571,'T8s':0.560,'T7s':0.549,'T6s':0.538,
        'T5s':0.528,'T4s':0.519,'T3s':0.510,'T2s':0.502,
        # Offsuit tens
        'T9o':0.552,'T8o':0.541,'T7o':0.530,'T6o':0.519,
        'T5o':0.509,'T4o':0.500,'T3o':0.491,'T2o':0.483,
        # Suited nines and below (abbreviated — remaining suits follow pattern)
        '98s':0.537,'97s':0.526,'96s':0.515,'95s':0.504,
        '87s':0.518,'86s':0.507,'85s':0.496,
        '76s':0.504,'75s':0.493,'65s':0.493,
        # Offsuit nines and below
        '98o':0.517,'97o':0.506,'87o':0.498,'76o':0.484,'65o':0.473,
        # Trash — bottom of range
        '72o':0.469,'32o':0.474,
    }
    return table


def _hand_canonical_key(hand: list) -> str:
    """Normalize ['Ah','Kd'] → 'AKo'; ['Ah','Ks'] → 'AKs'; ['Ah','As'] → 'AA'."""
    c1, c2 = str(hand[0]), str(hand[1])
    r1, r2 = c1[0], c2[0]
    s1, s2 = c1[1], c2[1]
    if _RANK_VALUE[r1] < _RANK_VALUE[r2]:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    suited = 's' if s1 == s2 else 'o'
    return r1 + r2 + suited


def _analyze_opp_card(opp_card, board: list, my_hand: list) -> dict:
    """
    Comprehensive single-card threat analysis.
    Returns a dict with:
      threat_level         — 'dangerous'|'strong'|'draw'|'draw_weak'|
                             'marginal'|'low_threat'|'air'
      flush_outs           — int  (opponent's flush outs)
      straight_outs        — int  (opponent's straight outs)
      draw_equity_per_card — float (equity gained per unseen card for draws)
      strong_second_fraction — float (fraction of second-card combos that make
                              opponent decent)
      score                — float summary score in [0,1]
      has_pair / has_trips_plus / overcard / we_hold_same_rank /
      we_have_both_higher / we_out_kick — bool flags
    """
    opp_str   = str(opp_card)
    opp_rank  = opp_str[0]
    opp_suit  = opp_str[1]
    board_str = [str(c) for c in board]
    my_str    = [str(c) for c in my_hand]

    board_ranks = [c[0] for c in board_str]
    board_suits = [c[1] for c in board_str]
    my_ranks    = [c[0] for c in my_str]
    my_suits    = [c[1] for c in my_str]

    opp_val   = _RANK_VALUE[opp_rank]
    board_vals = [_RANK_VALUE[r] for r in board_ranks]
    my_vals    = [_RANK_VALUE[r] for r in my_ranks]

    # ── Pair / trips detection ──────────────────────────────────────────
    has_pair      = opp_rank in board_ranks
    pair_count    = board_ranks.count(opp_rank)
    has_trips_plus = pair_count >= 2   # opp_card gives him trips or quads

    # ── Overcard detection ──────────────────────────────────────────────
    overcard = opp_val > max(board_vals) if board_vals else False

    # ── Kick / domination analysis vs OUR hand ───────────────────────────
    we_hold_same_rank = opp_rank in my_ranks
    we_have_both_higher = all(mv > opp_val for mv in my_vals)
    we_out_kick = False
    if has_pair and opp_rank in board_ranks:
        paired_my = [mv for mv in my_vals if mv == board_vals[board_ranks.index(opp_rank)] if opp_rank in board_ranks]
        we_out_kick = bool(paired_my) and max(my_vals) > opp_val

    # ── Flush draw / made flush ────────────────────────────────────────────
    board_suit_count = board_suits.count(opp_suit)
    flush_outs = 0
    if board_suit_count == 2:
        # opp has flush draw (3 to flush): ~9 outs on turn, 9 on river
        flush_outs = 9
    elif board_suit_count == 3:
        # made flush — very dangerous
        flush_outs = 0

    # ── Straight draw ────────────────────────────────────────────────────
    all_vals = sorted(board_vals + [opp_val])
    straight_outs = 0
    for i in range(len(all_vals) - 2):
        window = all_vals[i:i+3]
        span   = window[-1] - window[0]
        if span <= 4:
            # 3 connected in 5-card window → open-ended or gut-shot draw
            straight_outs = max(straight_outs, 8 if span == 3 else 4)

    # ── Draw equity approximation (Chen & Ankenman) ──────────────────────
    streets_left = 1 if len(board) >= 4 else 2
    flush_eq_card  = (flush_outs / 47.0) if flush_outs else 0.0
    str_eq_card    = (straight_outs / 47.0) if straight_outs else 0.0
    draw_equity_per_card = max(flush_eq_card, str_eq_card)

    # ── Strength score for second-card fraction ───────────────────────────
    high_thresh = 7   # T+
    strong_second_fraction = 0.0
    if has_pair:
        strong_second_fraction = 0.30   # trips or two-pair with another board card
    elif overcard:
        strong_second_fraction = 0.20   # could pair anything on board with 2nd card
    else:
        strong_second_fraction = 0.10

    # ── Score: aggregate threat in [0,1] ─────────────────────────────────
    score = 0.0
    if has_trips_plus:
        score = 0.90
    elif board_suit_count == 3:           # made flush
        score = 0.85
    elif has_pair:
        top_pair = max(board_vals) == opp_val if board_vals else False
        score = 0.70 if top_pair else 0.55
    elif draw_equity_per_card >= 0.17:    # strong draw
        score = 0.50
    elif overcard:
        score = 0.35
    elif draw_equity_per_card >= 0.08:    # weak draw
        score = 0.30
    else:
        score = 0.10

    # ── Threat classification ─────────────────────────────────────────────
    if has_trips_plus or board_suit_count == 3:
        threat = 'dangerous'
    elif has_pair and max(board_vals) == opp_val if board_vals else False:
        threat = 'strong'
    elif has_pair:
        threat = 'strong'
    elif draw_equity_per_card >= 0.17:
        threat = 'draw'
    elif draw_equity_per_card >= 0.08:
        threat = 'draw_weak'
    elif overcard or we_hold_same_rank:
        threat = 'marginal'
    elif opp_val >= high_thresh:
        threat = 'low_threat'
    else:
        threat = 'air'

    return {
        'threat_level':           threat,
        'flush_outs':             flush_outs,
        'straight_outs':          straight_outs,
        'draw_equity_per_card':   draw_equity_per_card,
        'strong_second_fraction': strong_second_fraction,
        'score':                  score,
        'has_pair':               has_pair,
        'has_trips_plus':         has_trips_plus,
        'overcard':               overcard,
        'we_hold_same_rank':      we_hold_same_rank,
        'we_have_both_higher':    we_have_both_higher,
        'we_out_kick':            we_out_kick,
    }


def _board_texture_score(board: list) -> float:
    """
    Continuous board wetness score in [0, 1].
    Source: Janda (2013) "Applications of No-Limit Hold'em" Ch.4 board taxonomy;
            Tipton (2014) on flush / straight draw density.

    dry  < 0.25  — rainbow disconnected boards; bluffing more credible
    wet  > 0.50  — monotone / paired / connected; opponent nut combos dense
    """
    if not board:
        return 0.0
    suits = [str(c)[1] for c in board]
    ranks = [_RANK_VALUE.get(str(c)[0], 0) for c in board]
    score = 0.0

    # Flush draw density (Tipton 2014)
    max_suit = max(suits.count(s) for s in set(suits))
    if max_suit >= 3:
        score += 0.35
    elif max_suit == 2:
        score += 0.15

    # Paired board — trips / boat possibility (Janda 2013)
    rank_strs = [str(c)[0] for c in board]
    if len(rank_strs) != len(set(rank_strs)):
        score += 0.20

    # Connectedness — straight draw coverage
    if len(ranks) >= 2:
        spread = max(ranks) - min(ranks)
        if spread <= 4:
            score += 0.10
        elif spread <= 6:
            score += 0.05

    # High cards increase nut-hand density
    high = sum(1 for r in ranks if r >= 11)  # J, Q, K, A
    score += min(high * 0.05, 0.10)

    return min(score, 1.0)


def _compute_draw_denial_bet(pot: int, draw_equity: float, street: str) -> int:
    """
    Chen & Ankenman (2006) Ch.16 draw-denial formula.
    Bet must make drawing -EV: B / (P + B) ≥ draw_equity
    → B ≥ draw_equity * P / (1 - draw_equity)
    An aggression premium is added for positional uncertainty.
    """
    if draw_equity <= 0 or draw_equity >= 1:
        return int(pot * 0.50)
    denial = pot * draw_equity / (1.0 - draw_equity)
    street_premium = {'flop': 1.25, 'turn': 1.15, 'river': 1.0}.get(street, 1.1)
    return max(BIG_BLIND, int(denial * street_premium))


def monte_carlo_equity(my_hand: list, board: list,
                       opp_known: list, iters: int = MC_ITERS,
                       opp_range=None) -> float:
    """
    Monte-Carlo equity via eval7.  Known opponent cards are pinned in the deck.
    When opp_range is provided (and no opp cards revealed), samples opponent
    hands using weighted random.choices instead of uniform shuffle.
    Returns P(we win) in [0,1].
    """
    deck = eval7.Deck()
    known = set(str(c) for c in my_hand + board + (opp_known or []))
    deck.cards = [c for c in deck.cards if str(c) not in known]

    wins = ties = 0
    board_e7 = [eval7.Card(str(c)) for c in board]
    my_e7    = [eval7.Card(str(c)) for c in my_hand]
    opp_e7   = [eval7.Card(str(c)) for c in opp_known] if opp_known else []

    need_board  = 5 - len(board)
    need_opp    = 2 - len(opp_e7)
    total_need  = need_board + need_opp

    if total_need <= 0:
        # All cards known — single deterministic evaluation
        my_score  = eval7.evaluate(my_e7  + board_e7)
        opp_score = eval7.evaluate(opp_e7 + board_e7)
        return 1.0 if my_score > opp_score else (0.5 if my_score == opp_score else 0.0)

    # ── Range-weighted path ─────────────────────────────────────────────
    # Use opp_range when: no revealed opp cards, need exactly 2 opp cards,
    # and range has enough candidates.
    use_range = (opp_range is not None and need_opp == 2 and not opp_known)
    if use_range:
        cands, ws = opp_range.get_sampling_data(known)
        if len(cands) < 10:
            use_range = False

    if use_range:
        deck_list = list(deck.cards)  # cards excluding my_hand + board (~44)
        for opp_ca, opp_cb in random.choices(cands, weights=ws, k=iters):
            opp_strs = {str(opp_ca), str(opp_cb)}
            if need_board > 0:
                avail = [c for c in deck_list if str(c) not in opp_strs]
                if len(avail) < need_board:
                    continue
                run_board = random.sample(avail, need_board)
            else:
                run_board = []
            full_board = board_e7 + run_board
            my_score   = eval7.evaluate(my_e7           + full_board)
            opp_score  = eval7.evaluate([opp_ca, opp_cb] + full_board)
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
        return (wins + 0.5 * ties) / max(iters, 1)

    # ── Uniform path (fallback) ─────────────────────────────────────────
    for _ in range(iters):
        random.shuffle(deck.cards)
        drawn = deck.cards[:total_need]
        full_board = board_e7 + drawn[:need_board]
        opp_full   = opp_e7  + drawn[need_board:]
        my_score   = eval7.evaluate(my_e7   + full_board)
        opp_score  = eval7.evaluate(opp_full + full_board)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / max(iters, 1)


# ─────────────────────────────────────────────────────────────────────────────
# HandRange — Bayesian range tracking  (MIT range.cpp port)
# ─────────────────────────────────────────────────────────────────────────────

class HandRange:
    """
    Maintains a probability weight for every possible two-card hand (1326 combos).
    Mirrors the range tracking logic from the MIT 2024 C++ bot.

    Key operations:
      reset()                — reinitialise to uniform 1.0 for all combos
      zero_blocked(cards)    — set weight = 0 for hands conflicting with known cards
      update_on_action(...)  — Bayesian update after opponent folds / calls / raises
      to_3card_range(...)    — expand to 3-card space after auction win
      sample_equity(...)     — compute range-weighted equity against opponent range
      get_likely_hands(k)    — top-k combos by weight (for diagnostics)
    """

    def __init__(self):
        # weights[i] corresponds to _ALL_COMBOS[i] — (card_a_idx, card_b_idx)
        self.weights: list[float] = [1.0] * len(_ALL_COMBOS)
        # Map (i,j) → position in _ALL_COMBOS for O(1) lookup
        self._idx: dict = {c: n for n, c in enumerate(_ALL_COMBOS)}

    def reset(self):
        self.weights = [1.0] * len(_ALL_COMBOS)

    def zero_blocked(self, cards: list):
        """
        Zero out all combos that share a card with `cards`.
        Called whenever new board / hole cards become known.
        """
        blocked = set()
        for c in cards:
            s = str(c)
            if s in _CARD_TO_IDX:
                blocked.add(_CARD_TO_IDX[s])
        for n, (a, b) in enumerate(_ALL_COMBOS):
            if a in blocked or b in blocked:
                self.weights[n] = 0.0

    def update_on_action(self, action: str, board: list,
                         known_opp_cards: list = None):
        """
        Bayesian update after observing opponent action on this board.
        action: 'fold' | 'call' | 'check' | 'raise'

        MIT formula (main_bot.cpp update_range):
          For opponent (unknown hand):
            strat = (1 - WEIGHT_UNIFORM) * action_prob + WEIGHT_UNIFORM * (1 - action_prob)
            weight[i] *= strat

        We use a fast rank-based heuristic for action_prob to avoid running
        eval7 on all 1326 combos (which is too slow for 2s time limit).
        Only updates postflop (board ≥ 3 cards) since preflop has no board context.
        """
        if action not in ('fold', 'call', 'check', 'raise'):
            return
        # Skip preflop — no board to evaluate against
        if not board or len(board) < 3:
            return

        blocked = set()
        for c in (known_opp_cards or []) + list(board):
            s = str(c)
            if s in _CARD_TO_IDX:
                blocked.add(_CARD_TO_IDX[s])

        board_vals  = sorted([_RANK_VALUE[str(c)[0]] for c in board])
        board_suits = [str(c)[1] for c in board]
        max_bv      = max(board_vals) if board_vals else 6

        # Fast rank-based heuristic — avoids 1326 eval7.evaluate() calls.
        # MIT regularisation: strat = (1-U)*action_prob + U*(1-action_prob)
        # Weights multiply across updates; NO normalisation (normalising
        # collapses all weights to ~0.001 and destroys the range model).
        for n, (a, b) in enumerate(_ALL_COMBOS):
            if self.weights[n] == 0.0 or a in blocked or b in blocked:
                if a in blocked or b in blocked:
                    self.weights[n] = 0.0
                continue

            ca, cb = _ALL_EVAL7_CARDS[a], _ALL_EVAL7_CARDS[b]
            ra, rb = _RANK_VALUE[str(ca)[0]], _RANK_VALUE[str(cb)[0]]
            sa, sb = str(ca)[1], str(cb)[1]

            raw = (ra + rb) / 24.0
            if ra in board_vals:
                raw += 0.20 * (ra / 12.0)
            if rb in board_vals:
                raw += 0.20 * (rb / 12.0)
            if ra == rb:
                raw += 0.15
            for suit in (sa, sb):
                if board_suits.count(suit) >= 2:
                    raw += 0.10
                    break
            if ra > max_bv:
                raw += 0.08
            if rb > max_bv:
                raw += 0.08

            norm = min(1.0, max(0.0, raw))

            if action == 'raise':
                action_prob = 0.15 + 0.85 * norm
            elif action == 'fold':
                action_prob = 0.85 - 0.85 * norm
            elif action == 'call':
                dist = abs(norm - 0.50) * 2.0
                action_prob = 0.60 - 0.40 * dist
            else:  # check
                dist = abs(norm - 0.45) * 2.0
                action_prob = 0.55 - 0.30 * dist

            action_prob = max(0.01, min(0.99, action_prob))

            # MIT regularised Bayesian update (main_bot.cpp update_range)
            strat = ((1.0 - WEIGHT_UNIFORM) * action_prob
                     + WEIGHT_UNIFORM * (1.0 - action_prob))
            self.weights[n] *= strat

    def get_likely_hands(self, k: int = 10) -> list:
        """
        Return top-k (combo_tuple, eval7_cards, weight) sorted by weight.
        Used for diagnostics and range-weighted equity sampling.
        """
        indexed = [(w, n) for n, w in enumerate(self.weights) if w > 0]
        indexed.sort(reverse=True)
        result = []
        for w, n in indexed[:k]:
            a, b = _ALL_COMBOS[n]
            result.append((_ALL_COMBOS[n],
                           [_ALL_EVAL7_CARDS[a], _ALL_EVAL7_CARDS[b]],
                           w))
        return result

    def total_weight(self) -> float:
        return sum(self.weights)

    def get_sampling_data(self, blocked: set) -> tuple:
        """
        Returns (cands, ws): paired lists of (eval7.Card, eval7.Card) tuples
        and their corresponding weights, for use with random.choices.
        blocked: set of card string representations (e.g. {'Ah','Kd'}) to exclude.
        Reuses the cached _ALL_EVAL7_CARDS and _ALL_COMBOS globals — no allocation.
        """
        cands: list = []
        ws:    list = []
        for n, (a, b) in enumerate(_ALL_COMBOS):
            w = self.weights[n]
            if w <= 0.0:
                continue
            ca, cb = _ALL_EVAL7_CARDS[a], _ALL_EVAL7_CARDS[b]
            if str(ca) in blocked or str(cb) in blocked:
                continue
            cands.append((ca, cb))
            ws.append(w)
        return cands, ws

    def sample_equity(self, my_hand: list, board: list,
                      n_sample: int = 60) -> float:
        """
        Range-weighted equity estimate.
        Samples top-n_sample opponent hands by weight, computes our equity
        against each using eval7, returns probability-weighted average.
        Uses fast random runouts for incomplete boards.
        """
        top_hands = self.get_likely_hands(n_sample)
        if not top_hands:
            return 0.5

        board_e7 = [eval7.Card(str(c)) for c in board]
        my_e7    = [eval7.Card(str(c)) for c in my_hand]
        need     = 5 - len(board)

        blocked  = set(str(c) for c in my_hand + board)

        total_weight = 0.0
        weighted_wins = 0.0

        for _combo, opp_e7, w in top_hands:
            # Skip if opp hand conflicts with our cards
            if any(str(c) in blocked for c in opp_e7):
                continue

            if need == 0:
                # River — deterministic evaluation
                ms = eval7.evaluate(my_e7  + board_e7)
                os = eval7.evaluate(opp_e7 + board_e7)
                eq = 1.0 if ms > os else (0.5 if ms == os else 0.0)
                weighted_wins += w * eq
                total_weight  += w
            else:
                opp_strs = {str(x) for x in opp_e7}
                deck_remaining = [c for c in _ALL_EVAL7_CARDS
                                  if str(c) not in blocked
                                  and str(c) not in opp_strs]
                n_trials = min(30, len(deck_remaining) // max(need, 1))
                if n_trials <= 0:
                    continue
                wins = 0
                random.shuffle(deck_remaining)
                for t in range(n_trials):
                    run_cards = deck_remaining[t * need:(t + 1) * need]
                    if len(run_cards) < need:
                        break
                    full_board = board_e7 + run_cards
                    ms = eval7.evaluate(my_e7  + full_board)
                    os = eval7.evaluate(opp_e7 + full_board)
                    wins += 1.0 if ms > os else (0.5 if ms == os else 0.0)
                weighted_wins += w * wins / n_trials
                total_weight  += w

        if total_weight == 0:
            return 0.5
        return weighted_wins / total_weight


# ─────────────────────────────────────────────────────────────────────────────
# OppBidTracker — 4-value opponent bid model  (MIT auction.cpp port)
# ─────────────────────────────────────────────────────────────────────────────

class OppBidTracker:
    """
    Tracks opponent auction bids with the MIT 4-value model (auction.cpp).

    MIT implementation details:
      • v_is_excessive_bidder starts TRUE and is set FALSE only when
        (stack - villain_bid) > REASONABLE_DIST_FROM_MAX (=10).
      • abs_min/max tracked as raw chip values.
      • pct_min/max tracked as (bid / pot) ratio.
      • ABS_BIDDING_EPSILON = 2  (tight abs range threshold)
      • POT_PERCENTAGE_BIDDING_EPSILON = 0.1  (tight pct range threshold)
    """

    ABS_BIDDING_EPSILON  = 2      # MIT: abs range < 2 → tight
    PCT_BIDDING_EPSILON  = 0.10   # MIT: pct range < 0.1 → tight
    REASONABLE_DIST      = 10     # MIT: REASONABLE_DIST_FROM_MAX

    def __init__(self):
        self.bid_count: int    = 0
        self.abs_min:   int    = STARTING_STACK
        self.abs_max:   int    = -1
        self.pct_min:   float  = float(STARTING_STACK)
        self.pct_max:   float  = -1.0
        self.is_excessive: bool = True   # MIT: starts True
        self._bids: list[int]  = []

    def update_exploits(self, hero_bid: int, villain_bid: int,
                        bid_plus_pot: int):
        """
        MIT auction.cpp update_exploits — exact port.
        bid_plus_pot is the total pot AFTER both bids are applied.
        """
        # Recover the pot before bids
        if hero_bid == villain_bid:
            pot = bid_plus_pot - hero_bid - villain_bid
        elif hero_bid > villain_bid:
            pot = bid_plus_pot - villain_bid
        else:
            pot = bid_plus_pot - hero_bid
        pot = max(pot, 1)

        stack = STARTING_STACK - (pot // 2)

        # Excessive check: if opponent leaves > REASONABLE_DIST from stack
        if (stack - villain_bid) > self.REASONABLE_DIST:
            self.is_excessive = False

        self.abs_min = min(self.abs_min, villain_bid)
        self.abs_max = max(self.abs_max, villain_bid)

        bid_to_pot = villain_bid / float(pot)
        self.pct_min = min(self.pct_min, bid_to_pot)
        self.pct_max = max(self.pct_max, bid_to_pot)

        self.bid_count += 1
        self._bids.append(villain_bid)

    def n_obs(self) -> int:
        return self.bid_count

    def adapt_bid(self, default_bid: int, pot: int, stack: int) -> int:
        """
        MIT auction.cpp get_bid exploit logic — exact port.
        Called after computing default_bid from geometric series formula.
        """
        if default_bid > stack:
            return stack

        if self.bid_count < BID_OBS_MIN:
            return max(0, min(default_bid, stack))

        if self.is_excessive:
            # Opponent bids near all-in every time — match them
            return min(stack, max(default_bid, stack - self.REASONABLE_DIST))

        abs_bid_diff = self.abs_max - self.abs_min
        if abs(abs_bid_diff) < self.ABS_BIDDING_EPSILON:
            # Very tight absolute range — bid above their min
            return min(stack, max(default_bid, self.abs_min))

        pct_diff = self.pct_max - self.pct_min
        if abs(pct_diff) < self.PCT_BIDDING_EPSILON:
            # Tight pot-fraction range — undercut via floor of their min
            return min(stack, max(default_bid,
                                  int(math.floor(self.pct_min * pot))))

        return max(0, min(default_bid, stack))

    @property
    def avg_bid(self) -> float:
        return sum(self._bids) / len(self._bids) if self._bids else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Player bot
# ─────────────────────────────────────────────────────────────────────────────

class Player(BaseBot):
    """
    MIT-Architecture SneakBot.

    Layered on normal_bid_use.py's proven strategy engine with the following
    MIT-sourced improvements:
      • Bayesian HandRange tracking for opponent (updated after every action)
      • Range-weighted equity (blended with Monte-Carlo)
      • MIT geometric-series auction bid formula
      • 4-value OppBidTracker with exploit mode
    """

    def __init__(self):
        super().__init__()

        # ── Static lookup tables ──────────────────────────────────────────
        self._preflop_table: dict = _build_preflop_equity_table()

        # ── Game-level aggregate counters ────────────────────────────────
        self.opp_folds         : int   = 0
        self.opp_raises        : int   = 0
        self.opp_calls         : int   = 0
        self.opp_checks        : int   = 0
        self.opp_actions_total : int   = 0

        # ── Auction tracking ─────────────────────────────────────────────
        self._bid_tracker      = OppBidTracker()
        self._auction_my_bid   : int   = 0
        self._auction_wins     : int   = 0
        self._auction_total    : int   = 0
        self.auction_results   : list  = []

        # ── Range tracking ────────────────────────────────────────────────
        self.opp_range = HandRange()

        # ── Per-hand state (reset in on_hand_start) ──────────────────────
        self.preflop_equity    : float = 0.5
        self.won_auction       : bool  = False
        self.opp_won_auction   : bool  = False
        self.info_advantage    : float = 0.0
        self.revealed_opp_card        = None
        self.opp_card_board_str: float = 0.0
        self.opp_analysis             = None
        self.prev_opp_wager    : int   = 0
        self.prev_opp_chips    : int   = STARTING_STACK
        self.hand_street_history: list = []
        self._last_range_update_action: str = ''

    # ── Aggregate fold / raise rates ─────────────────────────────────────
    @property
    def opp_fold_rate(self) -> float:
        if self.opp_actions_total < 5:
            return 0.25
        return self.opp_folds / self.opp_actions_total

    @property
    def opp_raise_rate(self) -> float:
        if self.opp_actions_total < 5:
            return 0.30
        return self.opp_raises / self.opp_actions_total

    @property
    def opp_avg_bid(self) -> float:
        return self._bid_tracker.avg_bid

    @property
    def opp_call_rate(self) -> float:
        """Fraction of actions that were calls. Defaults to 0.30 until 5 obs."""
        if self.opp_actions_total < 5:
            return 0.30
        return self.opp_calls / self.opp_actions_total

    @property
    def opp_aggression_factor(self) -> float:
        """
        Aggression Factor = raises / max(calls, 1).
        Standard HUD metric (Poker Tracker, Hold'em Manager documentation).
        AF > 1.5: aggressive player; AF < 0.8: passive/calling station.
        Defaults to 1.0 (neutral) until 5 observations.
        """
        if self.opp_actions_total < 5:
            return 1.0
        return self.opp_raises / max(self.opp_calls, 1)

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle callbacks
    # ─────────────────────────────────────────────────────────────────────

    def on_game_start(self, game_info: GameInfo) -> None:
        pass

    def on_hand_start(self, game_info: GameInfo,
                      current_state: PokerState) -> None:
        s = current_state
        key = _hand_canonical_key(s.my_hand)
        self.preflop_equity = self._preflop_table.get(key, 0.50)

        self.won_auction        = False
        self.opp_won_auction    = False
        self.info_advantage     = 0.0
        self.revealed_opp_card  = None
        self.opp_card_board_str = 0.0
        self.opp_analysis       = None
        self._auction_my_bid    = 0
        self.prev_opp_wager     = s.opp_wager
        self.prev_opp_chips     = s.opp_chips
        self.hand_street_history = []
        self._last_range_update_action = ''

        # Reset opponent range to uniform and block known cards
        self.opp_range.reset()
        self.opp_range.zero_blocked(s.my_hand + s.board)

    def on_hand_end(self, game_info: GameInfo,
                    current_state: PokerState) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _make_raise(self, state: PokerState, target: int):
        """Clamp raise to legal bounds; fall back to call/check if needed."""
        if not state.can_act(ActionRaise):
            return ActionCall() if state.can_act(ActionCall) else ActionCheck()
        lo, hi = state.raise_bounds
        return ActionRaise(max(lo, min(hi, int(target))))

    def _infer_opp_action(self, state: PokerState):
        """
        Infer opponent's last action from wager / chip deltas.
        Records aggregate counts AND triggers Bayesian range update.
        """
        if state.street not in self.hand_street_history:
            self.hand_street_history.append(state.street)
            self.prev_opp_wager = state.opp_wager
            self.prev_opp_chips = state.opp_chips
            # Block newly revealed board cards in range
            if state.board:
                self.opp_range.zero_blocked(state.board)
            return

        delta_wager = state.opp_wager - self.prev_opp_wager
        inferred    = None

        if delta_wager > 0:
            if state.opp_wager > state.my_wager:
                self.opp_raises += 1
                inferred = 'raise'
            else:
                self.opp_calls += 1
                inferred = 'call'
            self.opp_actions_total += 1
        elif delta_wager == 0 and self.prev_opp_chips == state.opp_chips:
            self.opp_checks += 1
            inferred = 'check'
            self.opp_actions_total += 1

        self.prev_opp_wager = state.opp_wager
        self.prev_opp_chips = state.opp_chips

        if inferred and inferred != self._last_range_update_action:
            self.opp_range.update_on_action(
                inferred, state.board,
                list(state.opp_revealed_cards) if state.opp_revealed_cards else None
            )
            self._last_range_update_action = inferred

    def _blend_equity(self, mc_eq: float, my_hand: list,
                      board: list, time_bank: float,
                      range_used: bool = False) -> float:
        """
        Blend Monte-Carlo equity with range-weighted equity.
        Skip range sampling entirely if time is tight or insufficient data.
        MIT insight: range is most useful mid-game with several updates.
        """
        if time_bank < 5.0 or len(board) < 3:
            return mc_eq
        if self.opp_actions_total < 2:
            return mc_eq  # not enough observations for meaningful range
        n_obs = min(self.opp_actions_total, 8)
        blend = min(RANGE_BLEND_W, 0.08 + 0.05 * n_obs)
        try:
            range_eq = self.opp_range.sample_equity(my_hand, board, n_sample=40)
        except Exception:
            return mc_eq
        return (1.0 - blend) * mc_eq + blend * range_eq

    def _adjust_equity_for_aggression(self, equity: float,
                                       cost: int, pot: int,
                                       s) -> float:
        """
        Discount or boost equity based on opponent's Aggression Factor (AF).

        Research basis:
          Sklansky & Malmuth (1999) "The Theory of Poker" — facing raises from
          passive players signals stronger ranges than identical raises from
          aggressive players (Bayes-optimal read on bet frequency).
          Poker Tracker / Hold'em Manager AF definition: AF = raises / calls.

        Logic:
          • AF > 1.5 facing raise  → opp raises often, range is wide → small discount
          • AF 0.8-1.5 facing raise → neutral → tiny discount
          • AF < 0.8 facing raise  → rare raise from passive = polarised strong → larger discount
          • Facing check, AF > 1.5 → bluff check-backs likely → small equity boost
        Only active when opp_actions_total >= 8 (enough sample).
        """
        if self.opp_actions_total < 8:
            return equity
        af = self.opp_aggression_factor
        facing_raise = cost > 0 and s.opp_wager > s.my_wager
        if facing_raise:
            bet_frac = cost / max(pot + cost, 1)
            if af > 1.5:
                # Wide raise range — only slight discount
                discount = 0.02 + 0.03 * bet_frac
            elif af >= 0.8:
                discount = 0.01 + 0.02 * bet_frac
            else:
                # Passive opp rarely raises — their range is very strong
                discount = 0.04 + 0.05 * bet_frac
            equity = max(0.0, equity - discount)
        elif cost == 0 and af > 1.5:
            # Aggressive opp checking back — likely a bluff trap or weak
            equity = min(1.0, equity + 0.03)
        return equity

    # ─────────────────────────────────────────────────────────────────────
    # Main decision entry
    # ─────────────────────────────────────────────────────────────────────

    def get_move(self, game_info: GameInfo,
                 current_state: PokerState):
        s = current_state
        self._infer_opp_action(s)

        if s.street == 'auction':
            return self._auction_action(game_info, s)
        if s.street == 'pre-flop':
            return self._preflop_action(game_info, s)
        return self._postflop_action(game_info, s)

    # ─────────────────────────────────────────────────────────────────────
    # Auction strategy  (MIT geometric series formula + 4-value exploit)
    # ─────────────────────────────────────────────────────────────────────

    def _auction_action(self, game_info: GameInfo, s: PokerState):
        """
        MIT-inspired auction bidding.

        ═══════════════════════════════════════════════════════════════════
        BID FORMULA  (MIT auction.cpp / auction.h)
        ═══════════════════════════════════════════════════════════════════
        The core idea: the value of seeing the opponent's card is related
        to how much our equity changes when we see it.  The geometric
        series derivation gives:

            equity_diff   = E[eq | see card] - E[eq | no info]
                          ≈ max(board_avg_loss, hand_specific_loss)

            equity_with_bid = (1 / (1 - equity_diff) - 1) * pot

        This represents the pot-fraction we recover from information
        over the remaining streets via the infinite geometric series
        for compounding equity gains.

            default_bid = ceil(equity_with_bid × BID_MULTIPLIER=1.8) + 1

        The ×1.8 multiplier induces aggressive bidding to ensure we
        outbid a rational opponent who knows their own equity_diff.

        Board texture, uncertainty, and strength adjustments are then
        applied (same logic as VOI model, now as multipliers).

        ═══════════════════════════════════════════════════════════════════
        4-VALUE EXPLOIT  (MIT auction.cpp OppBidTracker)
        ═══════════════════════════════════════════════════════════════════
        After ≥ BID_OBS_MIN bids observed:
          • Excessive bidder  → near-all-in to guarantee win
          • Tight chip range  → bid above their abs_max
          • Tight pct range   → bid above their pct_max × pot
        """
        iters  = MC_ITERS_FAST if game_info.time_bank < 4 else 300
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards,
                                    iters=iters)
        pot    = s.pot
        stack  = s.my_chips

        # ── Equity difference estimate ───────────────────────────────────
        # MIT uses precomputed equity_loss tables (HandEquitiesThirdCard).
        # We approximate using uncertainty and board texture.
        # Key formula: equity_diff = max(board_avg_eq_loss, hand_specific_eq_loss)
        #
        # Calibration target: MIT's median equity_diff ≈ 0.04-0.08.
        # Higher board texture (draws, pairs) → higher equity_diff.
        # Uncertainty = 4p(1-p) peaks at 1.0 when equity = 0.50.

        base_uncertainty   = 4.0 * equity * (1.0 - equity)

        # Board complexity: paired boards, flush/straight textures have
        # higher equity_diff because seeing opponent's card changes more.
        board_complexity = 1.0
        my_suits    = [str(c)[1] for c in s.my_hand]
        board_suits = [str(c)[1] for c in s.board]
        board_ranks = [str(c)[0] for c in s.board]
        board_vals  = sorted([_RANK_VALUE[r] for r in board_ranks])
        my_vals     = sorted([_RANK_VALUE[str(c)[0]] for c in s.my_hand])

        # Monotone board → knowing suit is very valuable
        if len(set(board_suits)) == 1:
            board_complexity += 0.30
        # Two-tone
        elif len(set(board_suits)) == 2:
            board_complexity += 0.15
        # Paired board → trips/boat danger
        if len(board_ranks) != len(set(board_ranks)):
            board_complexity += 0.20
        # Connected board → straight draws
        if board_vals:
            spread = board_vals[-1] - board_vals[0]
            if spread <= 4:
                board_complexity += 0.15
            elif spread <= 6:
                board_complexity += 0.08

        # equity_diff ≈ uncertainty × base_rate × board_complexity
        # Calibrated to produce MIT-range values: 0.02 - 0.12 typically
        equity_diff = base_uncertainty * 0.06 * board_complexity
        equity_diff = max(0.01, min(equity_diff, 0.40))   # safety clamp

        # ── MIT geometric series formula ─────────────────────────────────
        # MIT: equity_with_bid = ((1 / (1 - equity_difference)) - 1) * pot
        # Then: default_bid = ceil(equity_with_bid * BID_MULTIPLIER) + 1
        # Special case from MIT: if (stack - pot) < 0, skip the multiplier
        if equity_diff > 0:
            equity_with_bid = ((1.0 / (1.0 - equity_diff)) - 1.0) * pot
        else:
            equity_with_bid = 0.0

        stack_minus_pot = stack - pot
        if stack_minus_pot < 0:
            default_bid = int(math.ceil(equity_with_bid)) + 1
        else:
            default_bid = int(math.ceil(equity_with_bid * BID_MULTIPLIER)) + 1

        # ── Hand-specific adjustment ─────────────────────────────────────
        # Our hand's interaction with the board: flush draws and straight
        # draws in OUR hand make seeing opponent's card more valuable.
        flush_bonus = False
        for suit in my_suits:
            if board_suits.count(suit) >= 2:
                default_bid = int(default_bid * 1.20)
                flush_bonus = True
                break

        all_vals = sorted(my_vals + board_vals)
        for i in range(len(all_vals) - 3):
            if all_vals[i + 3] - all_vals[i] <= 4:
                default_bid = int(default_bid * 1.10)
                break

        # ── Strength asymmetry cut-offs ──────────────────────────────────
        # Less aggressive cuts — even strong/weak hands benefit from info
        if equity > 0.85:
            default_bid = int(default_bid * 0.30)
        elif equity > 0.78:
            default_bid = int(default_bid * 0.50)
        if equity < 0.15:
            default_bid = int(default_bid * 0.20)
        elif equity < 0.22:
            default_bid = int(default_bid * 0.45)

        # ── Floor bids (competitive participation) ───────────────────────
        # Must outbid opponents who use VOI model (avg ~37 chips).
        # Floors scaled to equity tier — winning auction is very valuable.
        if equity >= 0.72:
            default_bid = max(default_bid, BIG_BLIND * 5)      # 100
        elif equity >= 0.62:
            default_bid = max(default_bid, BIG_BLIND * 4)      # 80
        elif equity >= 0.55:
            default_bid = max(default_bid, int(BIG_BLIND * 3)) # 60
        elif equity >= 0.48:
            default_bid = max(default_bid, BIG_BLIND * 2)      # 40
        elif equity >= 0.40:
            default_bid = max(default_bid, BIG_BLIND)           # 20
        elif equity >= 0.32:
            default_bid = max(default_bid, BIG_BLIND // 2)      # 10

        # ── Ceiling (chip preservation — MIT style) ──────────────────────
        # MIT caps at stack only: if (default_bid > stack) return stack;
        # We add a softer ceiling: don't bid more than 50% of stack or
        # the geometric formula output, whichever is higher.
        max_sensible = min(stack, max(int(pot * 0.50), int(stack * 0.25)))
        default_bid  = min(default_bid, max_sensible)

        # ── 4-value exploit adaptation ───────────────────────────────────
        final_bid = self._bid_tracker.adapt_bid(default_bid, pot, stack)

        self._auction_my_bid = final_bid
        return ActionBid(final_bid)

    def _detect_auction_result(self, s: PokerState):
        """
        Called once on first post-auction action.
        Determines who won the auction, records opponent's bid for tracking
        via MIT's update_exploits(hero_bid, villain_bid, bid_plus_pot).
        """
        if s.opp_revealed_cards:
            self.won_auction       = True
            self.revealed_opp_card = s.opp_revealed_cards[0]
            self.opp_analysis      = _analyze_opp_card(
                self.revealed_opp_card, s.board, s.my_hand)
            self.opp_card_board_str = self.opp_analysis['score']

            chips_in_total   = STARTING_STACK - s.my_chips
            chips_in_betting = s.my_wager
            chips_in_auction = chips_in_total - chips_in_betting

            if self._auction_my_bid > 0 and chips_in_auction >= self._auction_my_bid * 0.90:
                # Tie — both bid same amount
                self.info_advantage  = 0.0
                self.opp_won_auction = True
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid,
                                             'won': False, 'tie': True})
                opp_bid = self._auction_my_bid
            else:
                # Outright win — chips_in_auction = opp's bid (second-price)
                self.info_advantage  = 1.0
                self.opp_won_auction = False
                self._auction_wins  += 1
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid,
                                             'won': True, 'tie': False})
                opp_bid = max(0, int(chips_in_auction))

            # MIT update_exploits(hero_bid, villain_bid, bid_plus_pot)
            self._bid_tracker.update_exploits(
                self._auction_my_bid, opp_bid, s.pot)
            # Pin opponent's revealed card in range
            self.opp_range.zero_blocked([self.revealed_opp_card])

        else:
            self.won_auction = False
            expected_pot_no_auction = (
                (STARTING_STACK - s.my_chips  - s.my_wager) +
                (STARTING_STACK - s.opp_chips - s.opp_wager)
            )
            if s.pot > expected_pot_no_auction + 5:
                self.opp_won_auction = True
                self.info_advantage  = -1.0
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid,
                                             'won': False, 'tie': False})
                opp_bid = max(0, int((s.pot - expected_pot_no_auction) / 2))
                self._bid_tracker.update_exploits(
                    self._auction_my_bid, opp_bid, s.pot)
            else:
                self.opp_won_auction = False
                self.info_advantage  = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Pre-flop  (unchanged from normal_bid_use.py — table is proven good)
    # ─────────────────────────────────────────────────────────────────────

    def _preflop_action(self, game_info: GameInfo, s: PokerState):
        eq   = self.preflop_equity
        cost = s.cost_to_call

        ELITE    = 0.75
        PREMIUM  = 0.65
        STRONG   = 0.60
        ABOVE_AV = 0.55
        PLAYABLE = 0.50

        if cost > 0:
            pot_odds = cost / max(s.pot + cost, 1)

            if eq >= ELITE:
                return self._make_raise(s, s.opp_wager + max(cost * 4, BIG_BLIND * 10))
            if eq >= PREMIUM:
                if random.random() < 0.65:
                    return self._make_raise(s, s.opp_wager + max(cost * 3, BIG_BLIND * 7))
                return ActionCall()
            if eq >= STRONG:
                if pot_odds < 0.38 or cost <= BIG_BLIND * 5:
                    if random.random() < 0.25:
                        return self._make_raise(s, s.opp_wager + max(cost * 2, BIG_BLIND * 5))
                    return ActionCall()
                if random.random() < 0.18:
                    return ActionCall()
                return ActionFold()
            if eq >= ABOVE_AV:
                if cost <= BIG_BLIND * 4:
                    return ActionCall()
                if cost <= BIG_BLIND * 7 and random.random() < 0.30:
                    return ActionCall()
                return ActionFold()
            if eq >= PLAYABLE:
                if cost <= BIG_BLIND * 2:
                    return ActionCall()
                return ActionFold()
            if cost <= BIG_BLIND and random.random() < 0.12:
                return ActionCall()
            return ActionFold()

        # Open action
        if eq >= ELITE:
            return self._make_raise(s, BIG_BLIND * 5)
        if eq >= PREMIUM:
            return self._make_raise(s, BIG_BLIND * 4)
        if eq >= STRONG:
            return self._make_raise(s, int(BIG_BLIND * 3.5))
        if eq >= ABOVE_AV:
            if random.random() < 0.80:
                return self._make_raise(s, int(BIG_BLIND * 2.5))
            return ActionCheck()
        if eq >= PLAYABLE:
            if random.random() < 0.50:
                return self._make_raise(s, BIG_BLIND * 2)
            return ActionCheck()
        return ActionCheck()

    # ─────────────────────────────────────────────────────────────────────
    # Post-flop dispatch
    # ─────────────────────────────────────────────────────────────────────

    def _postflop_action(self, game_info: GameInfo, s: PokerState):
        # Detect auction result on first post-auction call
        if s.street == 'flop' and not self.won_auction and not self.opp_won_auction:
            self._detect_auction_result(s)

        # Re-analyse opponent card vs current board each street
        if self.won_auction and self.revealed_opp_card and self.info_advantage > 0:
            self.opp_analysis      = _analyze_opp_card(
                self.revealed_opp_card, s.board, s.my_hand)
            self.opp_card_board_str = self.opp_analysis['score']

        # Base MC equity — use range-weighted sampling only when range is mature.
        # Require >= 8 observed actions so the heuristic has had enough
        # multiplicative updates to meaningfully separate strong/weak combos.
        iters     = MC_ITERS_FAST if game_info.time_bank < 3.5 else MC_ITERS
        use_range = (not s.opp_revealed_cards
                     and game_info.time_bank > 8.0
                     and len(s.board) >= 3
                     and self.opp_actions_total >= 8)
        mc_eq  = monte_carlo_equity(s.my_hand, s.board,
                                    s.opp_revealed_cards, iters=iters,
                                    opp_range=self.opp_range if use_range else None)

        # Range-weighted blend (MIT innovation)
        equity = self._blend_equity(mc_eq, s.my_hand, s.board,
                                    game_info.time_bank)
        pot  = s.pot
        cost = s.cost_to_call

        if self.opp_analysis and self.info_advantage > 0:
            return self._postflop_with_info(s, equity, pot, cost)
        if self.opp_won_auction and self.info_advantage < 0:
            return self._postflop_vs_info(s, equity, pot, cost)
        return self._postflop_neutral(s, equity, pot, cost)

    # ─────────────────────────────────────────────────────────────────────
    # Post-flop: WE SEE OPPONENT'S CARD  (6 threat modes)
    # ─────────────────────────────────────────────────────────────────────

    def _postflop_with_info(self, s: PokerState, equity: float,
                            pot: int, cost: int):
        a          = self.opp_analysis
        threat     = a['threat_level']
        draw_eq    = a['draw_equity_per_card']
        strong_frac = a['strong_second_fraction']

        # All-in protection
        if cost > 0 and s.street == 'river':
            if cost / max(s.my_chips, 1) > 0.50 and equity < 0.65:
                if s.can_act(ActionFold):
                    return ActionFold()

        # MODE 1: DANGEROUS
        if threat == 'dangerous':
            if cost > 0:
                if equity >= 0.78:
                    return self._make_raise(s, int(pot * 0.80 + cost))
                pot_odds = cost / max(pot + cost, 1)
                if equity >= pot_odds + 0.08 and equity >= 0.52:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()
            if equity >= 0.78:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.70)))
            if equity >= 0.60:
                return ActionCheck()
            return ActionCheck()

        # MODE 2: STRONG
        if threat == 'strong':
            if cost > 0:
                pot_odds = cost / max(pot + cost, 1)
                if equity >= 0.72:
                    return self._make_raise(s, int(pot * 0.75 + cost))
                if equity >= 0.52:
                    return ActionCall()
                if equity >= pot_odds + 0.05:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()
            if equity >= 0.70:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.65)))
            if equity >= 0.55:
                if random.random() < 0.55:
                    return self._make_raise(s, max(BIG_BLIND, int(pot * 0.40)))
                return ActionCheck()
            return ActionCheck()

        # MODE 3: DRAW / DRAW_WEAK
        if threat in ('draw', 'draw_weak'):
            denial_bet = _compute_draw_denial_bet(pot, draw_eq, s.street)
            if cost > 0:
                pot_odds = cost / max(pot + cost, 1)
                if equity >= 0.62:
                    return self._make_raise(s, max(denial_bet + cost,
                                                   int(pot * 0.70 + cost)))
                if equity >= 0.48:
                    return ActionCall()
                if equity >= pot_odds + 0.05:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()
            if equity >= 0.52:
                bet = max(denial_bet, int(pot * 0.55))
                return self._make_raise(s, max(BIG_BLIND, bet))
            if equity >= 0.38 and s.street != 'river':
                if random.random() < 0.70:
                    return self._make_raise(s, max(BIG_BLIND,
                                                   int(denial_bet * 0.75)))
            return ActionCheck()

        # MODE 4: MARGINAL
        if threat == 'marginal':
            if cost > 0:
                pot_odds = cost / max(pot + cost, 1)
                if equity >= 0.68:
                    return self._make_raise(s, int(pot * 0.65 + cost))
                if equity >= 0.48:
                    if random.random() < 0.25:
                        return self._make_raise(s, int(pot * 0.50 + cost))
                    return ActionCall()
                if equity >= pot_odds + 0.03:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()
            if equity >= 0.65:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.60)))
            if equity >= 0.48:
                if random.random() < 0.60:
                    return self._make_raise(s, max(BIG_BLIND, int(pot * 0.40)))
                return ActionCheck()
            if equity >= 0.32 and s.street in ('flop', 'turn'):
                if random.random() < 0.35:
                    return self._make_raise(s, max(BIG_BLIND, int(pot * 0.30)))
            return ActionCheck()

        # MODE 5: LOW THREAT
        if threat == 'low_threat':
            if cost > 0:
                pot_odds = cost / max(pot + cost, 1)
                if equity >= 0.65:
                    return self._make_raise(s, int(pot * 0.70 + cost))
                if equity >= 0.45:
                    if random.random() < 0.25:
                        return self._make_raise(s, int(pot * 0.45 + cost))
                    return ActionCall()
                if equity >= pot_odds + 0.02:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()
            if equity >= 0.58:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.60)))
            if equity >= 0.42:
                if random.random() < 0.60:
                    return self._make_raise(s, max(BIG_BLIND, int(pot * 0.45)))
                return ActionCheck()
            if s.street in ('flop', 'turn') and random.random() < 0.35:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.35)))
            return ActionCheck()

        # MODE 6: AIR — maximum exploitation
        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)
            if equity >= 0.58:
                return self._make_raise(s, int(pot * 0.80 + cost))
            if equity >= 0.42:
                if random.random() < 0.25:
                    return self._make_raise(s, int(pot * 0.50 + cost))
                return ActionCall()
            if strong_frac < 0.25 and equity >= pot_odds - 0.03:
                return ActionCall()
            if equity >= pot_odds + 0.02:
                return ActionCall()
            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # No bet to face — relentless pressure vs air
        if equity >= 0.52:
            size = 0.75 if s.street == 'river' else 0.55
            return self._make_raise(s, max(BIG_BLIND, int(pot * size)))
        if equity >= 0.32:
            if s.street in ('flop', 'turn'):
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.45)))
            if s.street == 'river' and random.random() < 0.50:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.55)))
            return ActionCheck()
        if s.street == 'flop' and random.random() < 0.70:
            return self._make_raise(s, max(BIG_BLIND, int(pot * 0.40)))
        if s.street == 'turn':
            barrel = 0.50 if strong_frac < 0.20 else 0.30
            if random.random() < barrel:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.50)))
        if s.street == 'river':
            bluff = 0.35 if strong_frac < 0.15 else 0.18
            if self.opp_fold_rate > 0.30:
                bluff += 0.10
            if random.random() < bluff:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.65)))
        return ActionCheck()

    # ─────────────────────────────────────────────────────────────────────
    # Post-flop: OPPONENT HAS INFO  (defensive mode)
    # ─────────────────────────────────────────────────────────────────────

    def _postflop_vs_info(self, s: PokerState, equity: float,
                          pot: int, cost: int):
        if cost > 0 and s.street == 'river':
            commit = cost / max(s.my_chips, 1)
            if commit > 0.40 and equity < 0.60:
                if s.can_act(ActionFold):
                    return ActionFold()
            if commit > 0.25 and equity < 0.50:
                if s.can_act(ActionFold):
                    return ActionFold()

        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)
            if equity >= 0.75:
                return self._make_raise(s, int(pot * 0.65 + cost))
            if equity >= 0.58:
                return ActionCall()
            if equity >= pot_odds + 0.12:
                return ActionCall()
            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        if equity >= 0.75:
            return self._make_raise(s, max(BIG_BLIND, int(pot * 0.55)))
        if equity >= 0.60:
            if random.random() < 0.40:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.35)))
            return ActionCheck()
        if equity >= 0.50 and random.random() < 0.08:
            return self._make_raise(s, max(BIG_BLIND, int(pot * 0.30)))
        return ActionCheck()

    # ─────────────────────────────────────────────────────────────────────
    # Post-flop: NO INFO ADVANTAGE  (range-enhanced GTO-adjacent play)
    # ─────────────────────────────────────────────────────────────────────

    def _postflop_neutral(self, s: PokerState, equity: float,
                          pot: int, cost: int):
        """
        Standard play, now enhanced with range-weighted equity blend.
        Raises from the range model tilt equity estimates, improving
        marginal call/fold decisions especially on later streets.
        """
        if cost > 0 and s.street == 'river':
            if cost / max(s.my_chips, 1) > 0.60 and equity < 0.65:
                if s.can_act(ActionFold):
                    return ActionFold()

        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)
            if equity >= 0.70:
                return self._make_raise(s, int(pot * 0.70 + cost))
            if equity >= 0.55:
                if random.random() < 0.20:
                    return self._make_raise(s, int(pot * 0.50 + cost))
                return ActionCall()
            if equity >= pot_odds + 0.05:
                return ActionCall()
            if equity >= pot_odds - 0.02 and cost <= pot * 0.35:
                return ActionCall()
            if s.street == 'flop' and random.random() < 0.08:
                return self._make_raise(s, int(pot * 0.65 + cost))
            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        if equity >= 0.70:
            return self._make_raise(s, max(BIG_BLIND, int(pot * 0.65)))
        if equity >= 0.55:
            if random.random() < 0.55:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.45)))
            return ActionCheck()
        if equity >= 0.40:
            if random.random() < 0.15:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.35)))
            return ActionCheck()
        if s.street == 'river' and random.random() < 0.12:
            if self.opp_fold_rate > 0.35:
                return self._make_raise(s, max(BIG_BLIND, int(pot * 0.65)))
        if s.street == 'flop' and random.random() < 0.15:
            return self._make_raise(s, max(BIG_BLIND, int(pot * 0.40)))
        return ActionCheck()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_bot(Player(), parse_args())
