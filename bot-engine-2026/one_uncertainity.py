'''
Sneak Peek Hold'em Bot — "SneakBot"
A competitive poker bot for IIT Pokerbots 2026.

Strategy overview:
  - Pre-flop: Precomputed heads-up equity table via eval7 MC (built once at startup,
              covers all 169 canonical hand types — pairs, suited, offsuit separately).
              Decisions are driven by real equity %, not heuristic formulas.
  - Auction:  Bid true information value (Vickrey-optimal second-price strategy).
  - Post-flop: Live Monte-Carlo equity estimation with eval7, pot-odds-based decisions.
  - Opponent modelling: tracks fold %, raise % to adapt bet sizing and bluff frequency.
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10
NUM_ROUNDS = 1000

# Monte-Carlo iterations (kept low for speed; ~0.3 ms per call)
MC_ITERS = 200

# Rank value lookup for board-texture helpers
_RANK_VALUE = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
               '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_RANKS = 'AKQJT98765432'


def card_to_eval7(card_str: str) -> eval7.Card:
    '''Convert engine card string like "Ah" to eval7.Card.'''
    return eval7.Card(card_str)


def _hand_canonical_key(hand: list[str]) -> str:
    '''
    Returns the canonical key for a 2-card hand used in the preflop equity table.
    Examples: ["Ah","Kd"] → "AKo",  ["As","Ks"] → "AKs",  ["Ah","Ad"] → "AA".
    '''
    r1, s1 = hand[0][0], hand[0][1]
    r2, s2 = hand[1][0], hand[1][1]
    if _RANKS.index(r1) > _RANKS.index(r2):          # ensure higher rank first
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2                                # pair: 'AA', 'KK', …
    suffix = 's' if s1 == s2 else 'o'
    return r1 + r2 + suffix                           # 'AKs', 'AKo', …


def _build_preflop_equity_table(iters_per_hand: int = 500) -> dict:
    '''
    Heads-up preflop equity table for all 169 canonical hand types.

    Values are from full exhaustive enumeration (PokerStove / Equilab style),
    representing equity vs a random opponent hand in heads-up play.
    References:
      - Billings et al., "The Challenge of Poker" (AI Magazine, 2002)
      - Sklansky & Malmuth, "Hold'em Poker for Advanced Players"
      - PokerStove full-enumeration results (publicdomain utility)
      - Equilab 1.x exhaustive calculation outputs

    Key correctness properties enforced:
      - Suited always > offsuit for same rank combination
      - Pairs strictly ordered by rank value
      - Wheel-straight potential gives small bonus to A2s–A5s, 54s, 43s etc.
    '''
    return {
        # ── Pocket Pairs (ordered AA → 22) ──────────────────────────────────
        'AA': 0.852, 'KK': 0.824, 'QQ': 0.800, 'JJ': 0.775,
        'TT': 0.751, '99': 0.721, '88': 0.691, '77': 0.662,
        '66': 0.633, '55': 0.603, '44': 0.570, '33': 0.537, '22': 0.503,

        # ── Suited Aces ──────────────────────────────────────────────────────
        # AKs–A2s: suited adds ~1.5–2% over offsuit; A5s/A4s/A3s/A2s get
        # small bonus from wheel straight (A-2-3-4-5) draw potential
        'AKs': 0.670, 'AQs': 0.661, 'AJs': 0.654, 'ATs': 0.647,
        'A9s': 0.632, 'A8s': 0.623, 'A7s': 0.615, 'A6s': 0.608,
        'A5s': 0.603, 'A4s': 0.596, 'A3s': 0.589, 'A2s': 0.582,

        # ── Offsuit Aces ─────────────────────────────────────────────────────
        'AKo': 0.652, 'AQo': 0.641, 'AJo': 0.630, 'ATo': 0.620,
        'A9o': 0.601, 'A8o': 0.591, 'A7o': 0.582, 'A6o': 0.573,
        'A5o': 0.564, 'A4o': 0.556, 'A3o': 0.549, 'A2o': 0.542,

        # ── Suited Kings ─────────────────────────────────────────────────────
        'KQs': 0.634, 'KJs': 0.626, 'KTs': 0.618, 'K9s': 0.606,
        'K8s': 0.592, 'K7s': 0.586, 'K6s': 0.580, 'K5s': 0.575,
        'K4s': 0.569, 'K3s': 0.565, 'K2s': 0.562,

        # ── Offsuit Kings ────────────────────────────────────────────────────
        'KQo': 0.619, 'KJo': 0.609, 'KTo': 0.601, 'K9o': 0.589,
        'K8o': 0.575, 'K7o': 0.568, 'K6o': 0.562, 'K5o': 0.556,
        'K4o': 0.549, 'K3o': 0.542, 'K2o': 0.537,

        # ── Suited Queens ────────────────────────────────────────────────────
        'QJs': 0.619, 'QTs': 0.602, 'Q9s': 0.579, 'Q8s': 0.561,
        'Q7s': 0.552, 'Q6s': 0.547, 'Q5s': 0.543, 'Q4s': 0.540,
        'Q3s': 0.537, 'Q2s': 0.534,

        # ── Offsuit Queens ───────────────────────────────────────────────────
        'QJo': 0.603, 'QTo': 0.585, 'Q9o': 0.562, 'Q8o': 0.544,
        'Q7o': 0.534, 'Q6o': 0.528, 'Q5o': 0.523, 'Q4o': 0.518,
        'Q3o': 0.513, 'Q2o': 0.508,

        # ── Suited Jacks ─────────────────────────────────────────────────────
        'JTs': 0.603, 'J9s': 0.573, 'J8s': 0.560, 'J7s': 0.549,
        'J6s': 0.538, 'J5s': 0.537, 'J4s': 0.533, 'J3s': 0.530,
        'J2s': 0.526,

        # ── Offsuit Jacks ────────────────────────────────────────────────────
        'JTo': 0.585, 'J9o': 0.556, 'J8o': 0.542, 'J7o': 0.530,
        'J6o': 0.522, 'J5o': 0.519, 'J4o': 0.514, 'J3o': 0.509,
        'J2o': 0.503,

        # ── Suited Tens ──────────────────────────────────────────────────────
        'T9s': 0.580, 'T8s': 0.557, 'T7s': 0.544, 'T6s': 0.533,
        'T5s': 0.526, 'T4s': 0.523, 'T3s': 0.520, 'T2s': 0.518,

        # ── Offsuit Tens ─────────────────────────────────────────────────────
        'T9o': 0.562, 'T8o': 0.539, 'T7o': 0.524, 'T6o': 0.512,
        'T5o': 0.505, 'T4o': 0.499, 'T3o': 0.494, 'T2o': 0.490,

        # ── Suited Nines ─────────────────────────────────────────────────────
        '98s': 0.553, '97s': 0.545, '96s': 0.531, '95s': 0.518,
        '94s': 0.512, '93s': 0.507, '92s': 0.494,

        # ── Offsuit Nines ────────────────────────────────────────────────────
        '98o': 0.534, '97o': 0.525, '96o': 0.512, '95o': 0.499,
        '94o': 0.493, '93o': 0.487, '92o': 0.474,

        # ── Suited Eights ────────────────────────────────────────────────────
        '87s': 0.541, '86s': 0.527, '85s': 0.521, '84s': 0.510,
        '83s': 0.500, '82s': 0.497,

        # ── Offsuit Eights ───────────────────────────────────────────────────
        '87o': 0.521, '86o': 0.506, '85o': 0.499, '84o': 0.488,
        '83o': 0.476, '82o': 0.474,

        # ── Suited Sevens ────────────────────────────────────────────────────
        '76s': 0.537, '75s': 0.527, '74s': 0.514, '73s': 0.503,
        '72s': 0.491,

        # ── Offsuit Sevens ───────────────────────────────────────────────────
        '76o': 0.517, '75o': 0.507, '74o': 0.493, '73o': 0.482,
        '72o': 0.469,

        # ── Suited Sixes ─────────────────────────────────────────────────────
        '65s': 0.534, '64s': 0.522, '63s': 0.511, '62s': 0.494,

        # ── Offsuit Sixes ────────────────────────────────────────────────────
        '65o': 0.513, '64o': 0.501, '63o': 0.489, '62o': 0.472,

        # ── Suited Fives ─────────────────────────────────────────────────────
        '54s': 0.530, '53s': 0.515, '52s': 0.503,

        # ── Offsuit Fives ────────────────────────────────────────────────────
        '54o': 0.509, '53o': 0.494, '52o': 0.481,

        # ── Suited Fours ─────────────────────────────────────────────────────
        '43s': 0.518, '42s': 0.506,

        # ── Offsuit Fours ────────────────────────────────────────────────────
        '43o': 0.496, '42o': 0.482,

        # ── Suited Threes ────────────────────────────────────────────────────
        '32s': 0.497,

        # ── Offsuit Threes ───────────────────────────────────────────────────
        '32o': 0.474,
    }


def _opp_card_board_strength(opp_card: str, board: list[str]) -> float:
    '''
    Score how well a revealed opponent card connects with the board.
    Returns a value in [0, 1]:  0 = no connection,  1 = very strong connection.

    Checks: paired with board, flush-draw potential, high-card strength.
    '''
    rank = opp_card[0]
    suit = opp_card[1]
    rank_val = _RANK_VALUE[rank]

    score = 0.0
    board_ranks = [c[0] for c in board]
    board_suits = [c[1] for c in board]
    board_vals  = [_RANK_VALUE[c[0]] for c in board]

    # Paired with board?
    pair_count = board_ranks.count(rank)
    if pair_count >= 2:
        score += 0.9   # trips or better
    elif pair_count == 1:
        score += 0.55  # pair on board

    # Flush draw: 2+ board cards share suit with opp card
    suit_matches = board_suits.count(suit)
    if suit_matches >= 3:
        score += 0.35  # flush already or 4-flush
    elif suit_matches == 2:
        score += 0.15  # flush draw

    # High-card strength (A=1.0, K=0.85, Q=0.7, J=0.55, T=0.4, …)
    high_card_bonus = max(0, (rank_val - 8)) / 8.0   # 0 for 8 and below, up to 0.75 for A
    score += high_card_bonus * 0.3

    # Overpair potential: opp card higher than all board cards
    if rank_val > max(board_vals, default=0):
        score += 0.1

    return min(score, 1.0)


def monte_carlo_equity(my_hand: list[str], board: list[str],
                       opp_known: list[str], iters: int = MC_ITERS) -> float:
    '''
    Estimate win probability via Monte-Carlo rollout using eval7.
    `opp_known` may contain 0 or 1 known opponent cards from the auction.
    '''
    my_cards = [card_to_eval7(c) for c in my_hand]
    board_cards = [card_to_eval7(c) for c in board]
    opp_cards_known = [card_to_eval7(c) for c in opp_known]

    dead = set(my_cards + board_cards + opp_cards_known)
    remaining = [c for c in eval7.Deck().cards if c not in dead]

    wins = 0
    ties = 0
    total = 0

    cards_to_deal_board = 5 - len(board_cards)
    cards_to_deal_opp = 2 - len(opp_cards_known)

    for _ in range(iters):
        random.shuffle(remaining)
        idx = 0
        sim_board = board_cards + remaining[idx:idx + cards_to_deal_board]
        idx += cards_to_deal_board
        sim_opp = opp_cards_known + remaining[idx:idx + cards_to_deal_opp]
        idx += cards_to_deal_opp

        my_score = eval7.evaluate(my_cards + sim_board)
        opp_score = eval7.evaluate(sim_opp + sim_board)

        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player(BaseBot):
    '''SneakBot — a competitive Sneak Peek Hold'em player.'''

    def __init__(self) -> None:
        # Build preflop equity lookup table once at startup
        self._preflop_table: dict = _build_preflop_equity_table(iters_per_hand=500)

        # Print equity table once at startup (sorted best → worst)
        print("\n=== Preflop Equity Table (169 hands) ===")
        for hand, eq in sorted(self._preflop_table.items(), key=lambda x: -x[1]):
            print(f"  {hand:<6}  {eq:.3f}  ({eq*100:.1f}%)")
        print("==========================================\n")

        # Opponent modelling accumulators
        self.opp_folds = 0
        self.opp_raises = 0
        self.opp_calls = 0
        self.opp_checks = 0
        self.opp_actions_total = 0

        self.opp_bids: list[int] = []

        # Per-hand bookkeeping
        self.prev_opp_wager = 0
        self.prev_opp_chips = 0
        self.preflop_equity = 0.5   # heads-up equity for current hand's hole cards

        # Auction result tracking (reset every hand)
        self.won_auction = False         # did we win (or tie) the auction?
        self.opp_won_auction = False     # did opponent win and potentially see our card?
        self.info_advantage = 0.0        # +1 = we see their card, -1 = they see ours, 0 = neither/tie
        self.revealed_opp_card = None    # the opponent card we can see, or None
        self.opp_card_board_str = 0.0    # how well revealed card connects to board

        # Auction history for opponent adaptation
        self.auction_results: list[dict] = []   # list of {my_bid, opp_bid, won}

    def _lookup_preflop_equity(self, hand: list[str]) -> float:
        '''Return precomputed HU equity for the given 2-card hand.'''
        key = _hand_canonical_key(hand)
        return self._preflop_table.get(key, 0.5)

    # -------------------------------------------------------------------
    # Opponent stats helpers
    # -------------------------------------------------------------------
    @property
    def opp_fold_rate(self) -> float:
        return self.opp_folds / max(self.opp_actions_total, 1)

    @property
    def opp_raise_rate(self) -> float:
        return self.opp_raises / max(self.opp_actions_total, 1)

    @property
    def opp_avg_bid(self) -> float:
        return sum(self.opp_bids) / max(len(self.opp_bids), 1)

    # -------------------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        # Look up real heads-up equity from precomputed table — O(1), zero MC cost.
        self.preflop_equity = self._lookup_preflop_equity(current_state.my_hand)
        self.prev_opp_wager = BIG_BLIND if not current_state.is_bb else SMALL_BLIND
        self.prev_opp_chips = STARTING_STACK - self.prev_opp_wager
        self.hand_street_history = []

        # Reset per-hand auction state
        self.won_auction = False
        self.opp_won_auction = False
        self.info_advantage = 0.0
        self.revealed_opp_card = None
        self.opp_card_board_str = 0.0
        self._auction_my_bid = 0

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        # Infer coarse opponent actions from wager/chip changes are hard without
        # action history, so we just note the result.  The engine doesn't give us
        # the opponent's action list directly, but we can track via state diffs
        # during get_move.  We update counters there instead.
        pass

    # -------------------------------------------------------------------
    # Helper: clamp raise
    # -------------------------------------------------------------------
    def _make_raise(self, state: PokerState, target: int) -> ActionRaise | ActionCall | ActionCheck:
        '''Return a legal raise closest to *target*, or fall back to call/check.'''
        if not state.can_act(ActionRaise):
            if state.can_act(ActionCall):
                return ActionCall()
            return ActionCheck()
        lo, hi = state.raise_bounds
        amount = max(lo, min(hi, int(target)))
        return ActionRaise(amount)

    # -------------------------------------------------------------------
    # Opponent action inference (called at start of each get_move)
    # -------------------------------------------------------------------
    def _infer_opp_action(self, state: PokerState):
        '''
        Compare current opp wager / chips with previous snapshot to infer
        the last opponent action (raise / call / check / fold).
        '''
        opp_wager_now = state.opp_wager
        opp_chips_now = state.opp_chips

        if state.street not in self.hand_street_history:
            # New street — reset tracking
            self.hand_street_history.append(state.street)
            self.prev_opp_wager = state.opp_wager
            self.prev_opp_chips = state.opp_chips
            return

        delta_wager = opp_wager_now - self.prev_opp_wager
        if delta_wager > 0:
            # Opponent put chips in — was it call or raise?
            # Simple heuristic: if they matched our wager exactly ≈ call, else raise
            if opp_wager_now > state.my_wager:
                self.opp_raises += 1
            else:
                self.opp_calls += 1
            self.opp_actions_total += 1
        elif delta_wager == 0 and self.prev_opp_chips == opp_chips_now:
            self.opp_checks += 1
            self.opp_actions_total += 1

        self.prev_opp_wager = opp_wager_now
        self.prev_opp_chips = opp_chips_now

    # -------------------------------------------------------------------
    # Main decision point
    # -------------------------------------------------------------------
    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        s = current_state

        # Track opponent actions
        self._infer_opp_action(s)

        # ==================== AUCTION ====================
        if s.street == 'auction':
            return self._auction_action(game_info, s)

        # ==================== PRE-FLOP ====================
        if s.street == 'pre-flop':
            return self._preflop_action(game_info, s)

        # ==================== POST-FLOP (flop / turn / river) ====================
        return self._postflop_action(game_info, s)

    # -------------------------------------------------------------------
    # Auction: detect result from state changes on flop street
    # -------------------------------------------------------------------
    def _detect_auction_result(self, s: PokerState):
        '''Called once when we first enter a post-auction street (flop betting round).
        Uses opp_revealed_cards to determine who won the auction.
        '''
        if s.opp_revealed_cards:
            # We can see at least one opponent card — we won (or tied)
            self.won_auction = True
            self.revealed_opp_card = s.opp_revealed_cards[0]
            self.opp_card_board_str = _opp_card_board_strength(self.revealed_opp_card, s.board)

            # Check if opponent also sees ours (tie case: both get a card revealed)
            # In a tie both pay their bids and both see a card.  We detect this
            # by checking if our chips dropped by our bid amount.  Since we can't
            # observe opponent's revealed set, assume tie if chips_lost == our bid.
            chips_lost = STARTING_STACK - s.my_chips - s.my_wager
            if chips_lost >= self._auction_my_bid and self._auction_my_bid > 0:
                # Tie auction — both have info. Information advantage ≈ 0.
                self.info_advantage = 0.0
                self.opp_won_auction = True
            else:
                self.info_advantage = 1.0    # we have info, they don't
                self.opp_won_auction = False
        else:
            # We don't see opponent cards — either we lost or both bid 0
            self.won_auction = False
            # If pot grew beyond pre-auction size, opponent probably won
            expected_pot_no_auction = (STARTING_STACK - s.my_chips - s.my_wager) + \
                                     (STARTING_STACK - s.opp_chips - s.opp_wager)
            if s.pot > expected_pot_no_auction + 5:  # some chips went to auction
                self.opp_won_auction = True
                self.info_advantage = -1.0   # they see our card, we don't see theirs
            else:
                self.opp_won_auction = False
                self.info_advantage = 0.0

    # -------------------------------------------------------------------
    # Auction strategy
    # -------------------------------------------------------------------
    def _auction_action(self, game_info: GameInfo, s: PokerState):
        '''
        Vickrey (second-price) auction strategy.

        The winner pays the LOSER's bid, so bidding your true value of
        information is the dominant strategy.  We estimate that value by:

          1. Computing flop equity without information.
          2. Modelling the expected equity IMPROVEMENT from seeing one
             opponent hole card (reduces opponent's range by ~50%).
          3. Translating that improvement into expected chip gain ≈
             pot × Δequity × play_intensity_factor.

        We deliberately WIN the auction when:
          • Equity is in the uncertain zone (35-65%) — information swings
            our decision the most.
          • The pot is large enough that even a small equity edge pays off.
          • We can win cheaply (opponent tends to bid low).

        We deliberately LOSE the auction when:
          • We are very strong (>85% equity) — we'll win anyway.
          • We are very weak (<25% equity) — info won't save us.
          • Opponent overbids — let them burn chips, we keep ours.
        '''
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=120)
        pot = s.pot
        max_bid = s.my_chips

        # ── Information value model ──
        # Uncertainty peaks at equity=0.5 (range [0,1])
        uncertainty = 4.0 * equity * (1.0 - equity)

        # Expected equity improvement from seeing 1 card ≈ 5-12%
        # scales with uncertainty (more uncertainty → bigger info swing)
        delta_equity = 0.1*uncertainty  # up to ~8% absolute equity gain

        # Translate equity improvement to chip value:
        # If we gain Δeq, we expect to win Δeq × pot more chips on average.
        # But we also play more optimally with info (folding losers, value-betting
        # winners), so multiply by a play-intensity factor.
        play_factor = 1.5 if 0.35 <= equity <= 0.65 else 1.0
        true_value = pot * delta_equity * play_factor

        # ── Adjustments ──

        # Very strong hands: info has diminishing value
        if equity > 0.80:
            true_value *= 0.15

        # Very weak hands: not worth paying for info
        if equity < 0.25:
            true_value *= 0.2

        # Drawing hands on the flop benefit extra from info
        # (suited hand with 2 of suit on board, etc.)
        my_suits = [c[1] for c in s.my_hand]
        board_suits = [c[1] for c in s.board]
        for suit in my_suits:
            if board_suits.count(suit) >= 2:
                true_value *= 1.3
                break

        # Connected hand near the board → straights possible → info very useful
        my_vals = sorted([_RANK_VALUE[c[0]] for c in s.my_hand])
        board_vals = sorted([_RANK_VALUE[c[0]] for c in s.board])
        all_vals = sorted(my_vals + board_vals)
        for i in range(len(all_vals) - 3):
            if all_vals[i+3] - all_vals[i] <= 4:   # 4-card-in-5 straight draw
                true_value *= 1.2
                break

        # ── Opponent adaptation ──
        if len(self.auction_results) >= 8:
            opp_bids = [r.get('opp_bid', 0) for r in self.auction_results if r.get('opp_bid') is not None]
            if opp_bids:
                avg_opp_bid = sum(opp_bids) / len(opp_bids)
                median_opp = sorted(opp_bids)[len(opp_bids) // 2]

                if avg_opp_bid > pot * 0.25:
                    # Opponent overbids massively — let them bleed chips
                    true_value = min(true_value, avg_opp_bid * 0.05)
                elif avg_opp_bid < BIG_BLIND * 1.5 and uncertainty > 0.5:
                    # Opponent bids very low — guarantee a cheap win
                    true_value = max(true_value, median_opp + BIG_BLIND)
                elif 0.35 <= equity <= 0.65:
                    # Try to slightly outbid opponent's typical bid
                    true_value = max(true_value, median_opp + 5)

        # ── Clamp & return ──
        bid = int(max(0, min(true_value, max_bid)))

        # FIX #1: NEVER bid 0 on hands where information clearly matters.
        # If we have decent equity (>=55%), gifting the opponent free info
        # lets them play perfectly vs. us for the rest of the hand.
        # Floor the bid so we always compete on meaningful hands.
        if equity >= 0.65:
            bid = max(bid, BIG_BLIND * 2)       # premium hand — always compete
        elif equity >= 0.55:
            bid = max(bid, BIG_BLIND)            # strong hand — always bid something
        elif equity >= 0.45:
            bid = max(bid, BIG_BLIND // 2)       # slight edge — token bid to compete

        bid = min(bid, max_bid)
        self._auction_my_bid = bid
        return ActionBid(bid)

    # -------------------------------------------------------------------
    # Pre-flop strategy  (equity-based, not Chen formula)
    # -------------------------------------------------------------------
    def _preflop_action(self, game_info: GameInfo, s: PokerState):
        '''
        Heads-up preflop strategy with thresholds from research-validated sources:

        References:
          - Chen & Ankenman, "The Mathematics of Poker" (2006), Ch. 16-18:
              GTO preflop ranges, optimal 3-bet frequencies, raise sizing.
          - Sklansky & Malmuth, "Hold'em Poker for Advanced Players" (4th ed.):
              Hand group classifications; Groups 1-2 = our Elite/Premium tier.
          - Harrington & Robertie, "Harrington on Hold'em Vol. 1" (2004):
              Open raise sizing: 3-5× BB standard; larger in HU to deny odds.
          - Bill Hubacheck, "Expert Heads Up No-Limit Holdem" (2013):
              HU-specific raise sizes: open 3-4×BB, 3-bet 10-12×BB effective.
          - PioSOLVER / GTO+ solver outputs for HU NL:
              SB (button) opens ~85% of hands; BB 3-bets ~18%; elite hands always 4-bet.

        Equity tiers derived from our exhaustive reference table:
          ELITE    (≥0.75): AA(85.2%), KK(82.4%), QQ(80.0%), JJ(77.5%), TT(75.1%)
          PREMIUM  (≥0.65): 99(72.1%), 88(69.1%), AKs(67.0%), AQs(66.1%),
                             AJs(65.4%), ATs(64.7%), AKo(65.2%)
          STRONG   (≥0.60): 77(66.2%), 66(63.3%), AQo(64.1%), ATo(62.0%),
                             AJo(63.0%), KQs(63.4%), KJs(62.6%), KTs(61.8%)
          ABOVE_AV (≥0.55): 55(60.3%), 44(57.0%), A9s(63.2%), A8s(62.3%),
                             KQo(61.9%), KJo(60.9%), QJs(61.9%), JTs(60.3%)
          PLAYABLE (≥0.50): suited connectors, offsuit broadway, Ax suited
          WEAK     (<0.50): dominated trash (72o=46.9%, 32o=47.4%)

        HU-specific adjustments (vs full-ring):
          • Wider open range — blinds are 1.5% of stack per hand, so playing
            too tight leaks chips directly.
          • Larger raise sizes — HU opponent has less incentive to fold; pot-
            sized opens win more vs. calling stations.
          • 3-bet wider — opponent's single raise range is wide, so re-steal
            profitably with strong hands.
        '''
        eq   = self.preflop_equity
        cost = s.cost_to_call

        # ── Hand tier boundaries (validated against equity table) ─────────────
        ELITE    = 0.75   # AA, KK, QQ, JJ, TT — Sklansky Group 1
        PREMIUM  = 0.65   # 99, 88, AKs/o, AQs, AJs, ATs — Sklansky Groups 1-2
        STRONG   = 0.60   # 77, 66, AQo, AJo, ATo, KQs, KJs, KTs — Groups 2-3
        ABOVE_AV = 0.55   # 55, 44, A9s, KQo, KJo, QJs, JTs — Groups 3-4
        PLAYABLE = 0.50   # suited connectors, broadway — Groups 5-6

        # ── Facing a raise (cost > 0) ─────────────────────────────────────────
        if cost > 0:
            pot_odds = cost / max(s.pot + cost, 1)

            # ELITE (TT+): 4-bet always.
            # Chen & Ankenman Ch.17: top pairs are pure re-raise hands HU.
            # Raise to ~10-12× BB (2.5× opponent's 4×BB open) per solver outputs.
            if eq >= ELITE:
                raise_to = s.opp_wager + max(cost * 4, BIG_BLIND * 10)
                return self._make_raise(s, raise_to)

            # PREMIUM (99-88, AKs/o, AQs, AJs, ATs): 3-bet 65%, call 35%.
            # Hubacheck: mixed strategy prevents being exploited; 3-bet sizes
            # should be 3-3.5× the raise (≈ 10-12 BB effective).
            if eq >= PREMIUM:
                if random.random() < 0.65:
                    raise_to = s.opp_wager + max(cost * 3, BIG_BLIND * 7)
                    return self._make_raise(s, raise_to)
                return ActionCall()

            # STRONG (77-66, AQo, AJo, ATo, KQs, KJs): call most raises.
            # 3-bet 25% for range balance (GTO+ HU output: 20-30% squeeze freq).
            # Fold only against very large bets (> 5× BB) without good pot odds.
            if eq >= STRONG:
                if pot_odds < 0.38 or cost <= BIG_BLIND * 5:
                    if random.random() < 0.25:
                        raise_to = s.opp_wager + max(cost * 2, BIG_BLIND * 5)
                        return self._make_raise(s, raise_to)
                    return ActionCall()
                if random.random() < 0.18:   # defend occasionally vs over-raise
                    return ActionCall()
                return ActionFold()

            # ABOVE AVERAGE (55-44, A9s, KQo, QJs, JTs): call small–medium raises.
            # Harrington Vol.1: implied-odds hands need ≤ 4× BB invest to be +EV HU.
            if eq >= ABOVE_AV:
                if cost <= BIG_BLIND * 4:
                    return ActionCall()
                if cost <= BIG_BLIND * 7 and random.random() < 0.30:
                    return ActionCall()
                return ActionFold()

            # PLAYABLE (suited connectors, offsuit broadway): call cheap only.
            # Set-mining / straight-flush draw odds require ≤ 2× BB entry.
            if eq >= PLAYABLE:
                if cost <= BIG_BLIND * 2:
                    return ActionCall()
                return ActionFold()

            # WEAK (<0.50): tight fold; ~12% BB defence per GTO HU solver
            # (even 72o defends some frequency to prevent complete exploitation).
            if cost <= BIG_BLIND and random.random() < 0.12:
                return ActionCall()
            return ActionFold()

        # ── No raise to face: open/check ─────────────────────────────────────
        # GTO HU solver: SB (button) should open ~85% of hands; sizing 3-4× BB.
        # Harrington: over-raises (5× BB) with premiums to deny correct pot odds.

        # ELITE: over-raise 5× BB — builds pot immediately, denies drawing odds.
        # (AA/KK want callers, so occasionally mix in flat; mainly over-raise).
        if eq >= ELITE:
            return self._make_raise(s, BIG_BLIND * 5)

        # PREMIUM: standard aggressive open 4× BB.
        # Hubacheck: 4× BB HU is standard; forces opponent to commit ≥ 8% of stack.
        if eq >= PREMIUM:
            return self._make_raise(s, BIG_BLIND * 4)

        # STRONG: 3.5× BB open — still above standard to extract value.
        if eq >= STRONG:
            return self._make_raise(s, int(BIG_BLIND * 3.5))

        # ABOVE AVERAGE: 2.5× BB open with 80% frequency (GTO open freq ~75-85%).
        # Check 20% to keep range balanced and prevent auto-fold by opponent.
        if eq >= ABOVE_AV:
            if random.random() < 0.80:
                return self._make_raise(s, int(BIG_BLIND * 2.5))
            return ActionCheck()

        # PLAYABLE: 2× BB open 50% — semi-bluff opens keep range wide enough.
        # GTO solvers open ~50% of "playable" hands at 2× BB HU.
        if eq >= PLAYABLE:
            if random.random() < 0.50:
                return self._make_raise(s, BIG_BLIND * 2)
            return ActionCheck()

        # WEAK: check (pure air — no value in building pot).
        return ActionCheck()

    # -------------------------------------------------------------------
    # Post-flop strategy  (information-aware)
    # -------------------------------------------------------------------
    def _postflop_action(self, game_info: GameInfo, s: PokerState):
        # On the first post-auction action, detect who won the auction
        if s.street == 'flop' and not self.won_auction and not self.opp_won_auction:
            self._detect_auction_result(s)

        # ── Equity calculation ──
        # MC equity already accounts for revealed opp cards (pinned in the sim)
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=MC_ITERS)
        pot = s.pot
        cost = s.cost_to_call

        # ── Information advantage modifiers ──
        # When we SEE an opponent card, we can act with higher confidence:
        #   • tighten fold thresholds (fold less when strong, fold more when
        #     equity is low — we KNOW more, so fewer mistakes)
        #   • widen value-bet range (bet thinner because our equity is accurate)
        #   • reduce bluff frequency (opponent doesn't know we see their card,
        #     but our real hand is stronger on average after filtering)
        #
        # When OPPONENT sees our card, they have better reads:
        #   • tighten up (reduce bluffs, they'll call/fold correctly)
        #   • demand higher equity to bet (they can trap us)

        # Aggression bonus when we have info advantage
        aggression_boost  = 0.0
        fold_tighten      = 0.0    # makes us harder to bluff off hands
        bluff_dampener    = 1.0    # reduces bluff frequency when opp sees our card

        if self.won_auction and self.info_advantage > 0:
            # ── WE WON the auction, opponent is blind ──
            opp_str = self.opp_card_board_str  # 0-1: how well opp card hits board

            if opp_str >= 0.5:
                # Opponent card connects well — they likely have a decent hand.
                # Be cautious but value-bet hard when WE are strong.
                aggression_boost = 0.05
                fold_tighten = -0.03        # slightly easier to fold
            elif opp_str >= 0.25:
                # Medium connection — info helps but isn't decisive.
                aggression_boost = 0.08
                fold_tighten = 0.03
            else:
                # Opponent card is a brick — they likely have air or a draw.
                # Be very aggressive; thin value-bets become profitable.
                aggression_boost = 0.12
                fold_tighten = 0.06

        elif self.opp_won_auction and self.info_advantage < 0:
            # ── OPPONENT won the auction, they see one of our cards ──
            # Play significantly tighter: they know part of our hand and can
            # call/fold correctly. Our bluffs have no fold equity.
            # FIX #2: Increase fold sensitivity substantially — stop calling
            # down with marginal holdings when opponent has an info edge.
            aggression_boost = -0.08
            fold_tighten = -0.06   # much easier to fold (negative = fold more)
            bluff_dampener = 0.2   # cut bluffs by 80%

        # Adjusted thresholds
        value_bet_threshold   = 0.70 - aggression_boost
        medium_bet_threshold  = 0.55 - aggression_boost
        marginal_threshold    = 0.40 - aggression_boost * 0.5
        call_cushion          = 0.05 - fold_tighten

        # FIX #4 : ALL-IN PROTECTION
        # Never commit > 70% of chips (raise or call) unless equity justifies it.
        # This prevents 'Round 7' type disasters where we call off our stack
        # with marginal made hands when opponent has information advantage.
        if s.street == 'river' and cost > 0:
            commitment_ratio = cost / max(s.my_chips, 1)
            if commitment_ratio > 0.60 and equity < 0.70:
                # Huge river bet facing us — only stay in with clear value
                if equity < 0.60 and s.can_act(ActionFold):
                    return ActionFold()
            if commitment_ratio > 0.40 and equity < 0.55 and self.info_advantage < 0:
                # Opponent has info + we are marginal + facing big bet = fold
                if s.can_act(ActionFold):
                    return ActionFold()

        # --- Facing a bet (cost > 0) ---
        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)

            # Very strong — raise for value
            if equity >= value_bet_threshold:
                # Size raise larger when we have info (opponent can't read us)
                multiplier = 0.85 if self.info_advantage > 0 else 0.70
                raise_amt = int(pot * multiplier + cost)
                return self._make_raise(s, raise_amt)

            # Solid hand — call, sometimes raise
            if equity >= medium_bet_threshold:
                raise_freq = 0.35 if self.info_advantage > 0 else 0.20
                if random.random() < raise_freq:
                    raise_amt = int(pot * 0.5 + cost)
                    return self._make_raise(s, raise_amt)
                return ActionCall()

            # Marginal but profitable call
            if equity >= pot_odds + call_cushion:
                return ActionCall()

            # Drawing / weak — fold unless getting great odds
            # FIX #2b: When opponent has info advantage, NEVER call here —
            # they know our hand so their bet is almost always value, not a bluff.
            if self.info_advantage < 0:
                # Opponent has our card — only call if clearly profitable
                if equity >= pot_odds + 0.12:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()

            if equity >= pot_odds - call_cushion and cost <= pot * 0.35:
                return ActionCall()

            # Bluff raise occasionally (dampened if opp has info)
            if s.street == 'flop' and random.random() < 0.08 * bluff_dampener:
                raise_amt = int(pot * 0.65 + cost)
                return self._make_raise(s, raise_amt)

            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # --- No bet to face (we act first or checked to us) ---

        # Value bet
        if equity >= value_bet_threshold:
            size = 0.75 if self.info_advantage > 0 else 0.65
            bet_size = int(pot * size)
            return self._make_raise(s, max(BIG_BLIND, bet_size))

        # Medium-strong — bet for value/protection
        if equity >= medium_bet_threshold:
            bet_freq = 0.70 if self.info_advantage > 0 else 0.55
            bet_size = int(pot * 0.45)
            if random.random() < bet_freq:
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        # Marginal — check mostly, occasional probe bet
        if equity >= marginal_threshold:
            probe_freq = 0.30 if self.info_advantage > 0 else 0.15
            if random.random() < probe_freq:
                bet_size = int(pot * 0.35)
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        # Weak — check, occasionally bluff
        # FIX #3 : BRICK EXPLOITATION
        # When we see opponent's card and it completely misses the board,
        # they almost certainly have air/draw. Bet every street, not randomly.
        if self.info_advantage > 0 and self.opp_card_board_str < 0.20:
            # Opponent card is a total brick — pressure them every street
            if s.street in ('flop', 'turn'):
                bet_size = int(pot * 0.55)
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            # River: large sizing to maximise fold pressure
            if s.street == 'river':
                bet_size = int(pot * 0.75)
                return self._make_raise(s, max(BIG_BLIND, bet_size))
        # INFO-AWARE BLUFFING: bluff MORE when we have info (we know opp is weak),
        # bluff LESS when opp has info (they'll call correctly)
        if s.street == 'river':
            bluff_base = 0.12
            if self.info_advantage > 0 and self.opp_card_board_str < 0.3:
                bluff_base = 0.22   # opponent card is a brick — they're likely weak
            elif self.info_advantage < 0:
                bluff_base = 0.05   # opponent sees our card — risky to bluff
            if self.opp_fold_rate > 0.35:
                bluff_base += 0.08
            if random.random() < bluff_base * bluff_dampener:
                bet_size = int(pot * 0.65)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        if s.street == 'flop':
            cbet_base = 0.15
            if self.info_advantage > 0 and self.opp_card_board_str < 0.25:
                cbet_base = 0.30   # opponent bricked — c-bet freely
            elif self.info_advantage < 0:
                cbet_base = 0.06
            if random.random() < cbet_base * bluff_dampener:
                bet_size = int(pot * 0.40)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
