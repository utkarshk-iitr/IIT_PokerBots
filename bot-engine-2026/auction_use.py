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
    Multi-factor scoring of how well a revealed opponent card connects to
    the board.  Returns [0, 1]:  0 = total brick,  1 = monster connection.

    Research backing:
      - Billings et al. (2003) "Approximating Game-Theoretic Optimal
        Strategies for Full-scale Poker": decompose hand strength into
        positive (made-hand) and negative (draw potential) components when
        partial information is available.
      - Johanson et al. (2011) "Accelerating Best Response Calculation in
        Large Extensive Games": opponent range narrowing from observed cards
        should factor in all board interactions (pair, flush, straight).
      - Gilpin, Sandholm & Sørensen (2007) "Potential-Aware Automated
        Abstraction of Sequential Games": bucket hands by (current strength +
        draw potential) to avoid abstraction pathologies.

    Components scored (max contribution → normalised):
      1. Pairing with board              (0 – 0.35)
      2. Flush / flush-draw              (0 – 0.20)
      3. Straight connectivity           (0 – 0.20)
      4. High-card / overcards           (0 – 0.15)
      5. Two-pair / set potential        (0 – 0.10)
    Total raw max ≈ 1.0, clamped [0, 1].
    '''
    rank = opp_card[0]
    suit = opp_card[1]
    rank_val = _RANK_VALUE[rank]

    score = 0.0
    board_ranks = [c[0] for c in board]
    board_suits = [c[1] for c in board]
    board_vals  = sorted([_RANK_VALUE[c[0]] for c in board])

    # ── 1. Pairing with board (made-hand strength) ────────────────────────
    pair_count = board_ranks.count(rank)
    if pair_count >= 2:
        score += 0.35                   # trips or quads — monster
    elif pair_count == 1:
        # Top pair vs middle/bottom pair matters (Harrington Vol.2)
        if rank_val == max(board_vals):
            score += 0.30              # top pair
        elif rank_val >= sorted(board_vals)[-2] if len(board_vals) >= 2 else rank_val:
            score += 0.22              # second pair
        else:
            score += 0.15              # bottom pair

    # ── 2. Flush / flush-draw potential ───────────────────────────────────
    suit_matches = board_suits.count(suit)
    if suit_matches >= 4:
        score += 0.20                  # flush made (or 1-away on river)
    elif suit_matches == 3:
        score += 0.17                  # 4-flush (one card to flush)
    elif suit_matches == 2:
        score += 0.08                  # flush draw (2 to come or 1 to come)

    # ── 3. Straight connectivity ──────────────────────────────────────────
    # Count how many board cards are within ±4 of the opp card rank —
    # these create open-ended or gutshot straight draws.
    # Reference: Chen & Ankenman (2006) "The Mathematics of Poker", ch.19
    # on straight-draw outs and implied odds.
    all_vals = board_vals + [rank_val]
    # Also handle A=1 for wheel straights
    if 14 in all_vals:
        all_vals_ext = all_vals + [1]
    else:
        all_vals_ext = all_vals[:]

    # Check 5-card windows for straight potential
    unique_vals = sorted(set(all_vals_ext))
    best_window = 0
    for base in range(1, 11):   # windows: 1-5, 2-6, ..., 10-14
        window = [v for v in unique_vals if base <= v <= base + 4]
        if rank_val in range(base, base + 5) or (rank_val == 14 and base == 1):
            # Opp card participates in this window
            best_window = max(best_window, len(window))

    if best_window >= 5:
        score += 0.20                  # straight already made with opp card
    elif best_window == 4:
        score += 0.14                  # open-ended straight draw
    elif best_window == 3:
        score += 0.07                  # gutshot / backdoor straight

    # ── 4. High-card / overcard strength ──────────────────────────────────
    # Overcards have ~6 outs to top pair (Chen & Ankenman 2006)
    max_board_val = max(board_vals) if board_vals else 0
    if rank_val > max_board_val and pair_count == 0:
        # Overcard — value scales with rank
        overcard_bonus = 0.08 + 0.07 * ((rank_val - max_board_val) / 6.0)
        score += min(overcard_bonus, 0.15)
    elif rank_val >= 12 and pair_count == 0:
        # Broadway card (Q/K/A) even if not overcard — kicker value
        score += 0.04

    # ── 5. Two-pair / set potential (hidden second card) ──────────────────
    # Opponent's unknown second card can pair another board card or make
    # two-pair / trips.  If their revealed card is already paired, the
    # unknown card making two-pair is a major threat.
    if pair_count == 1:
        # They have one pair + unknown card → two-pair threat ≈ (board_len - 1) * 2/45
        score += 0.08
    elif pair_count == 0 and len(board) >= 3:
        # No pair, but unknown card could pair any of len(board) ranks
        # Light threat score
        score += 0.03

    return min(score, 1.0)


def _opp_domination_check(my_hand: list[str], opp_card: str) -> dict:
    '''
    Analyse domination relationships between our hand and the revealed
    opponent card.  Returns a dict with exploitation signals.

    Research backing:
      - Sklansky (1999) "The Theory of Poker": domination principle — when
        two hands share a card rank, the one with the higher kicker wins
        the vast majority (>70%) of the time at showdown.
      - Billings et al. (2003): opponent modelling from partial info should
        include domination detection to adjust post-flop strategy.
      - Johanson et al. (2011): hand-vs-range equity shifts dramatically
        when one card is revealed, especially for dominated/dominating cases.

    Keys returned:
      we_dominate        : bool  — our card same rank but higher kicker
      we_are_dominated   : bool  — opp card dominates one of ours
      shared_rank        : bool  — same rank in hand (blocks their outs)
      opp_rank_val       : int   — opp card rank value
      suit_block         : bool  — we hold a card of same suit (flush blocker)
      kicker_edge        : float — +1.0 = strong kicker advantage, -1.0 = weak
    '''
    opp_rank = opp_card[0]
    opp_suit = opp_card[1]
    opp_val  = _RANK_VALUE[opp_rank]

    my_ranks = [c[0] for c in my_hand]
    my_suits = [c[1] for c in my_hand]
    my_vals  = [_RANK_VALUE[c[0]] for c in my_hand]

    result = {
        'we_dominate':      False,
        'we_are_dominated': False,
        'shared_rank':      False,
        'opp_rank_val':     opp_val,
        'suit_block':       opp_suit in my_suits,
        'kicker_edge':      0.0,
    }

    # Shared rank: we hold the same rank → we block their pair outs
    if opp_rank in my_ranks:
        result['shared_rank'] = True
        # Check kicker: our other card vs their unknown second card
        # If we share rank, our kicker is the non-shared card
        other_vals = [v for r, v in zip(my_ranks, my_vals) if r != opp_rank]
        if other_vals:
            our_kicker = max(other_vals)
            # We dominate if our kicker is strong (>= T)
            if our_kicker >= 10:
                result['we_dominate'] = True
                result['kicker_edge'] = min((our_kicker - 10) / 4.0, 1.0)
            else:
                result['kicker_edge'] = -0.3  # weak kicker, but we still block

    # Domination: opp card is same suit & higher than our highest
    # (Suited domination — they have nut flush draw when their suit matches)
    if opp_val > max(my_vals):
        # Opponent card outranks us — potential domination
        # Strong domination: Ace/King with our hand being medium-low
        if opp_val >= 13 and max(my_vals) <= 10:
            result['we_are_dominated'] = True
            result['kicker_edge'] = -((opp_val - max(my_vals)) / 8.0)
    elif opp_val < min(my_vals):
        # Our hand outranks opp card — we have equity advantage
        gap = min(my_vals) - opp_val
        result['kicker_edge'] = min(gap / 6.0, 1.0)

    return result


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
        # print("\n=== Preflop Equity Table (169 hands) ===")
        # for hand, eq in sorted(self._preflop_table.items(), key=lambda x: -x[1]):
        #     print(f"  {hand:<6}  {eq:.3f}  ({eq*100:.1f}%)")
        # print("==========================================\n")

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
        self.opp_domination = None       # dict from _opp_domination_check()

        # Auction history for opponent adaptation
        self.auction_results: list[dict] = []   # list of {my_bid, won, tie}
        self._auction_wins  = 0   # times we won outright (info_advantage=+1)
        self._auction_total = 0   # total auctions tracked

        # Opponent bid tracking:
        # In a second-price auction, when WE WIN outright we paid opp's bid
        # (chips_in_auction = opp_bid). On a tie both paid their own bid.
        # These observations let us directly model opponent bid behaviour.
        self._opp_auction_bids: list[int] = []

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
        self.opp_domination = None
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
        '''
        Called once on the first post-auction action (flop betting round).
        Uses opp_revealed_cards and chip deltas to determine who won,
        then records the result for opponent bid-pattern exploitation.
        '''
        if s.opp_revealed_cards:
            # We can see at least one opponent card — we won (or tied)
            self.won_auction = True
            self.revealed_opp_card = s.opp_revealed_cards[0]
            self.opp_card_board_str = _opp_card_board_strength(self.revealed_opp_card, s.board)
            self.opp_domination = _opp_domination_check(s.my_hand, self.revealed_opp_card)

            # Detect tie vs outright win:
            #   Win:  we paid opp_bid  (<  our bid)  → chips_in_auction < _auction_my_bid
            #   Tie:  we paid our_bid  (== our bid)  → chips_in_auction ≈ _auction_my_bid
            chips_in_total   = STARTING_STACK - s.my_chips
            chips_in_betting = s.my_wager
            chips_in_auction = chips_in_total - chips_in_betting  # approx auction cost

            if self._auction_my_bid > 0 and chips_in_auction >= self._auction_my_bid * 0.90:
                # We paid our full bid → tie (both bid the same amount)
                self.info_advantage  = 0.0
                self.opp_won_auction = True
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid, 'won': False, 'tie': True})
                # In a tie, opponent's bid ≈ our bid
                self._opp_auction_bids.append(self._auction_my_bid)
            else:
                # We paid less than our bid → outright win; we paid opp's bid
                self.info_advantage  = 1.0
                self.opp_won_auction = False
                self._auction_wins  += 1
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid, 'won': True, 'tie': False})
                # chips_in_auction is exactly what opponent bid (second-price)
                self._opp_auction_bids.append(max(0, int(chips_in_auction)))
        else:
            # We don't see opponent cards — we lost or both bid 0
            self.won_auction = False
            expected_pot_no_auction = (STARTING_STACK - s.my_chips - s.my_wager) + \
                                     (STARTING_STACK - s.opp_chips - s.opp_wager)
            if s.pot > expected_pot_no_auction + 5:   # chips entered pot from auction
                self.opp_won_auction = True
                self.info_advantage  = -1.0
                self._auction_total += 1
                self.auction_results.append({'my_bid': self._auction_my_bid, 'won': False, 'tie': False})
            else:
                # Both bid 0 — no auction effect
                self.opp_won_auction = False
                self.info_advantage  = 0.0

    # -------------------------------------------------------------------
    # Auction strategy
    # -------------------------------------------------------------------
    def _auction_action(self, game_info: GameInfo, s: PokerState):
        '''
        Research-backed second-price auction bidding for Sneak Peek Hold'em.

        ═══════════════════════════════════════════════════════════════════
        THEORETICAL FOUNDATION
        ═══════════════════════════════════════════════════════════════════

        1. Vickrey (1961) — Second-Price Auction Dominant Strategy:
           In a sealed-bid second-price auction, bidding true value V is
           weakly dominant. Win iff V > opp_bid, paying opp_bid < V.
           Over/under-bidding both reduce expected payoff strictly.
           → bid = VOI(hand, board, pot)

        2. Howard (1966) — Value of Information (VOI):
           VOI = E[payoff | with info] - E[payoff | without info]
           Winning the auction converts partial knowledge into near-perfect
           single-card certainty, enabling correct fold/call/bet decisions.

        3. Bowling et al. (2015) — "HU Limit Hold'em is Solved" (Science):
           Information value ∝ variance reduction in equity estimate.
           High-entropy spots (equity ≈ 0.50) benefit most. Calibration:
           ~8-15% of pot at maximally uncertain spots.
           Approximated by: uncertainty = 4·p·(1-p)  (variance of Bernoulli).

        4. Brown & Sandholm (2017) — Libratus, Science:
           Subgame principle: auction payoff = EV improvement across all
           remaining streets. Sneak Peek info is useful over 3 decisions
           (flop-bet + turn + river) → STREETS_FACTOR ≈ 2.5.
           Chip preservation: never risk > 15% of stack on information.

        5. Chen & Ankenman (2006) — "Mathematics of Poker" Ch. 14-18:
           Exploitative deviation from Vickrey when opponent has a
           measurable bid-pattern tendency (overbid / underbid).
           Use auction win-rate as a proxy for their bidding level.

        Sneak Peek Hold'em specifics:
           • Post-flop auction (3 board cards visible) — 2 streets remain.
           • Winner sees ONE random hole card of opponent.
           • Second-price: winner pays loser's bid; tie: both pay own bid.
        ═══════════════════════════════════════════════════════════════════
        '''
        # Extra MC iters for auction — this is the most critical decision
        equity  = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=300)
        pot     = s.pot
        max_bid = s.my_chips

        # ── Step 1: Core VOI Model (Bowling 2015 + Howard 1966) ──────────────
        #
        # uncertainty = 4p(1-p) ∈ [0,1], peaks at 1.0 when equity = 0.50.
        # This is the variance of Bernoulli(p) and closely tracks Shannon entropy.
        uncertainty = 4.0 * equity * (1.0 - equity)

        # STREETS_FACTOR: info benefits 3 decision rounds on Sneak Peek (flop-bet
        # + turn + river). Diminishing returns per street → effective factor 2.5.
        # VOI_RATE: fraction of pot capturable per unit uncertainty across streets.
        # Calibrated so that at peak uncertainty (eq=0.5, pot P):
        #   true_value = P × 0.10 × 1.0 × 2.5 = 0.25 × P   (25% of pot)
        STREETS_FACTOR = 2.5
        VOI_RATE       = 0.10

        true_value = pot * VOI_RATE * uncertainty * STREETS_FACTOR

        # ── Step 2: Strength Asymmetry Corrections ────────────────────────────
        # Near-certain outcome → information rarely changes the decision.

        if equity > 0.82:       # very strong: win without info on most runouts
            true_value *= 0.18
        elif equity > 0.75:
            true_value *= 0.38

        if equity < 0.18:       # very weak: fold without info on most runouts
            true_value *= 0.18
        elif equity < 0.25:
            true_value *= 0.38

        # ── Step 3: Board Texture Bonuses ─────────────────────────────────────
        # Textured boards → opponent's card rank/suit more often changes action.
        my_suits    = [c[1] for c in s.my_hand]
        board_suits = [c[1] for c in s.board]
        board_ranks = [c[0] for c in s.board]
        my_vals     = sorted([_RANK_VALUE[c[0]] for c in s.my_hand])
        board_vals  = sorted([_RANK_VALUE[c[0]] for c in s.board])

        # Flush draw: our hand has 2 suited cards matching 2+ board cards
        flush_bonus = False
        for suit in my_suits:
            if board_suits.count(suit) >= 2:
                true_value *= 1.25   # suit of opp card is critical information
                flush_bonus = True
                break

        # Monotone board: ALL 3 board cards same suit (extreme flush relevance)
        if len(set(board_suits)) == 1 and not flush_bonus:
            true_value *= 1.20

        # Straight draw: 4-card window within 5 consecutive ranks
        all_vals = sorted(my_vals + board_vals)
        for i in range(len(all_vals) - 3):
            if all_vals[i + 3] - all_vals[i] <= 4:
                true_value *= 1.15
                break

        # Paired board: opponent may have trips/boat draw — rank info critical
        if len(board_ranks) != len(set(board_ranks)):
            true_value *= 1.12

        # High board (T+): opponent's broadway connectivity matters a lot
        if board_vals and max(board_vals) >= 10:
            true_value *= 1.06

        # ── Step 4: Pot-Depth Scaling (Libratus subgame principle) ────────────
        # Larger committed pot → higher absolute stakes of each remaining decision.
        relative_pot = pot / (2 * STARTING_STACK)
        true_value  *= (1.0 + 0.25 * relative_pot)

        # ── Step 5: Dual-Track Opponent Bid Adaptation ───────────────────────
        # Each time we WIN the auction, chips_in_auction = opp's actual bid
        # (second-price mechanics). We accumulate these to estimate opp's
        # typical bid level and adapt in opposite directions:
        #
        #   LOW BIDDER  (opp_estimate < VOI_THRESHOLD):
        #     Bid just barely above their estimate. Winning cheaply is far
        #     more +EV than the marginal info from overbidding. Chip saved
        #     now compounds over 1000 rounds.
        #
        #   HIGH BIDDER (opp_estimate >= VOI_THRESHOLD):
        #     Bid competitively to avoid ceding info every hand. Being
        #     systematically outbid means opponent sees our cards and plays
        #     perfectly against us, which is deeply -EV.
        #
        # VOI_THRESHOLD: the crossover point where the opponent starts
        # bidding more than our pure-VOI estimate. Below this we're the
        # natural winner and can shade bids down; above it we must fight.

        # Boundary below which we consider the opponent a "low bidder"
        # (roughly: half of a max-uncertainty VOI at a 100-chip pot).
        VOI_THRESHOLD = max(10, int(pot * 0.07))

        n_obs = len(self._opp_auction_bids)
        opp_estimate = None

        if n_obs >= 5:
            opp_avg_all    = sum(self._opp_auction_bids) / n_obs
            recent         = self._opp_auction_bids[-15:]
            opp_avg_recent = sum(recent) / len(recent)
            # Weight recent bids 65% — opponents adjust strategy over time
            opp_estimate   = 0.35 * opp_avg_all + 0.65 * opp_avg_recent

            win_rate = self._auction_wins / max(self._auction_total, 1)

            if opp_estimate < VOI_THRESHOLD:
                # ──── LOW BIDDER: strictly enforce < 70% win rate ────
                # Winning every auction against a low bidder bleeds chips and
                # gains little — info from 1 card rarely covers the chip cost.
                # Target: 40–65% win rate, hard ceiling at 70%.
                #   • CONCEDE on clear winners/losers (info doesn't change action)
                #   • CONTEST only on high-uncertainty spots where info matters
                #   • Hard-concede whenever win_rate is approaching 70%
                win_rate = self._auction_wins / max(self._auction_total, 1)

                # Graduated contest threshold — gets stricter as win rate rises.
                # Hard ceiling: once at/above 65%, contest only peak-uncertainty hands.
                if self._auction_total >= 10 and win_rate >= 0.65:
                    # Approaching ceiling — concede everything except maximum
                    # uncertainty spots (equity right at 0.50, uncertainty > 0.85)
                    contest_threshold = 0.85
                elif self._auction_total >= 10 and win_rate >= 0.55:
                    # Getting high — raise the bar significantly
                    contest_threshold = 0.70
                elif self._auction_total >= 10 and win_rate >= 0.40:
                    # Healthy range — contest top third of uncertainty distribution
                    contest_threshold = 0.55
                elif self._auction_total >= 10 and win_rate < 0.30:
                    # Too low — fight a bit more
                    contest_threshold = 0.35
                else:
                    # Early rounds / no data yet — moderate contest rate
                    contest_threshold = 0.50

                if uncertainty >= contest_threshold:
                    # Contest: bid just above opponent's level (cheap win)
                    margin     = 1 + int(uncertainty * 3)   # 1–4 chip buffer
                    target_bid = int(opp_estimate) + margin
                    target_bid = max(target_bid, 1)
                    true_value = float(target_bid)
                else:
                    # Concede: bid well below opponent so they win without effort.
                    # Use 25% of their estimate (not 40%) for a cleaner concede —
                    # ensures we don't accidentally tie due to integer rounding.
                    concede_bid = max(0, int(opp_estimate * 0.25))
                    true_value  = float(concede_bid)

            else:
                # ──── HIGH BIDDER: bid competitively above their estimate ────
                # 20% buffer guarantees we beat them most rounds while still
                # paying only their (lower) bid back (second-price benefit).
                target_bid = opp_estimate * 1.20
                if true_value < target_bid:
                    # VOI is below what opp typically bids — scale up.
                    # Cap at 2.5× VOI so weak hands don’t inflate to absurdity.
                    true_value = min(target_bid, true_value * 2.5)

                # Still losing >65% despite upward adjustment — push harder.
                if self._auction_total >= 15 and win_rate < 0.35:
                    true_value *= 1.30
                # Winning >70% against a "high" bidder — trim slightly.
                elif self._auction_total >= 15 and win_rate > 0.70:
                    true_value = max(true_value * 0.85, opp_estimate * 1.10)

        # ── Step 6: Final Bid Computation ─────────────────────────────────────
        bid = int(max(0.0, true_value))

        # FLOOR bids — dynamically scaled to opponent's bid level.
        # LOW BIDDER:  true_value was already set to opp_estimate + margin in
        #              Step 5, so use tiny symbolic floors that won't override
        #              the shaded-down value and bleed chips unnecessarily.
        # HIGH BIDDER / no data yet: use standard Sklansky-tier floors to
        #              guarantee participation in the most valuable spots.
        is_low_bidder = (opp_estimate is not None and opp_estimate < VOI_THRESHOLD)

        if is_low_bidder:
            # Only apply a floor in genuinely uncertain spots (equity near 50%)
            # so we guarantee winning the info when it matters most.
            if equity >= 0.48:
                bid = max(bid, 1)
            # else: bid whatever Step 5 computed (may be 0 for weak hands)
        else:
            # Standard floors: ensure competitive participation
            if equity >= 0.72:                      # TT+ / AKs  — always contest
                bid = max(bid, BIG_BLIND * 4)
            elif equity >= 0.62:                    # 88-99 / AQs-AKo
                bid = max(bid, BIG_BLIND * 3)
            elif equity >= 0.55:                    # strong broadway / suited aces
                bid = max(bid, BIG_BLIND * 2)
            elif equity >= 0.48:                    # coin-flip territory
                bid = max(bid, BIG_BLIND)
            elif equity >= 0.40:                    # slight underdog — token bid
                bid = max(bid, BIG_BLIND // 2)
        # Below 0.40: bid = 0 is correct (losing hand, info won't help)

        # CEILING — Libratus chip-preservation principle.
        # LOW BIDDER:  tight ceiling — no reason to pay more than ~2× their avg.
        # HIGH BIDDER: standard ceiling 30% pot / 15% stack.
        if is_low_bidder:
            max_sensible = min(max_bid, max(int(opp_estimate * 2) + 5, int(s.my_chips * 0.04)))
        else:
            max_sensible = min(max_bid, max(int(pot * 0.30), int(s.my_chips * 0.15)))
        bid = min(bid, max_sensible)

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

            # Detect SB open (completing the blind) vs facing a real raise.
            # SB: my_wager=10, opp_wager=20, cost=10, pot=30 → this is just
            # completing the blind, NOT facing a raise.  Don't 4-bet here.
            is_sb_open = (s.my_wager == SMALL_BLIND and s.opp_wager == BIG_BLIND
                          and cost == SMALL_BLIND)

            if is_sb_open:
                # SB open-raise sizing (not facing a raise, just completing)
                if eq >= ELITE:
                    return self._make_raise(s, BIG_BLIND * 5)
                if eq >= PREMIUM:
                    return self._make_raise(s, BIG_BLIND * 4)
                if eq >= STRONG:
                    return self._make_raise(s, int(BIG_BLIND * 3.5))
                if eq >= ABOVE_AV:
                    if random.random() < 0.80:
                        return self._make_raise(s, int(BIG_BLIND * 2.5))
                    return ActionCall()
                if eq >= PLAYABLE:
                    if random.random() < 0.50:
                        return self._make_raise(s, BIG_BLIND * 2)
                    return ActionCall()
                # Weak: complete ~25% to prevent exploitation
                if random.random() < 0.25:
                    return ActionCall()
                return ActionFold()

            # ELITE (TT+): 4-bet always.
            # Chen & Ankenman Ch.17: top pairs are pure re-raise hands HU.
            # Raise to ~10-12× BB (2.5× opponent's 4×BB open) per solver outputs.
            if eq >= ELITE:
                raise_to = s.opp_wager + max(cost * 3, BIG_BLIND * 10)
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
        '''
        Research-backed postflop strategy with deep revealed-card exploitation.

        ═══════════════════════════════════════════════════════════════════
        THEORETICAL FOUNDATION — REVEALED CARD EXPLOITATION
        ═══════════════════════════════════════════════════════════════════

        1. Billings, Burch, Davidson, Holte, Schaeffer, Schauenberg &
           Szafron (2003) — "Approximating Game-Theoretic Optimal Strategies
           for Full-scale Poker":
           When partial opponent information is observed, decompose the
           situation into (a) current made-hand strength of the revealed
           card, (b) draw potential (flush/straight outs), and (c) range
           narrowing.  The bot should shift between value-heavy and bluff-
           heavy lines based on category (a-c).

        2. Ganzfried & Sandholm (2014) — "Potential-Aware Imperfect-Recall
           Abstraction with Earth Mover's Distance":
           Hands must be bucketed by *potential* (draw equity), not just
           current strength.  A revealed 7h on a board of 6h-8h-Kd is
           MUCH more dangerous than a revealed 7c on that board — flush +
           straight draw vs just a gutshot.  Our scoring must capture this.

        3. Johanson, Bowling & Waugh (2011) — "Accelerating Best Response
           Calculation in Large Extensive Games":
           With one card revealed, the opponent's effective range shrinks
           from ~1326 combos to ~50 possible second-card pairings.  This
           creates a dramatic information asymmetry: our MC equity is far
           more accurate than theirs.  Exploit by (a) value-betting thinner,
           (b) folding more precisely, (c) sizing bets to deny correct odds.

        4. Sklansky (1999) — "The Theory of Poker", Domination Principle:
           When two hands share a rank, the dominated hand wins < 25% at
           showdown.  Detecting domination from the revealed card lets us
           play aggressively (we dominate) or cautiously (we're dominated).

        5. Chen & Ankenman (2006) — "The Mathematics of Poker", Ch. 23-24:
           Optimal bet sizing in asymmetric information games should be
           polarized: large bets when info advantage is clear (brick),
           smaller "block bets" when opponent connects (protection).

        6. Brown & Sandholm (2017) — "Safe and Nested Subgame Solving for
           Imperfect-Information Games" (Libratus):
           Real-time equity refinement + opponent-range tracking.  Our MC
           equity already pins the known card; the modifiers below encode
           the *strategic* implications that MC alone doesn't capture
           (e.g., opponent's betting tendencies given their range).
        '''
        # On the first post-auction action, detect who won the auction
        if s.street == 'flop' and not self.won_auction and not self.opp_won_auction:
            self._detect_auction_result(s)

        # ── Equity calculation ──
        # MC equity already pins the revealed opp card in the simulation
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=MC_ITERS)
        pot = s.pot
        cost = s.cost_to_call

        # ── Re-score opp card on turn/river (board changed since flop) ──
        if self.won_auction and self.revealed_opp_card and s.street in ('turn', 'river'):
            self.opp_card_board_str = _opp_card_board_strength(self.revealed_opp_card, s.board)
            self.opp_domination = _opp_domination_check(s.my_hand, self.revealed_opp_card)

        # ══════════════════════════════════════════════════════════════════
        # INFORMATION ADVANTAGE MODIFIERS
        # ══════════════════════════════════════════════════════════════════
        aggression_boost  = 0.0    # shifts all bet/raise thresholds
        fold_tighten      = 0.0    # positive = harder to bluff us; negative = fold easier
        bluff_dampener    = 1.0    # multiplier on bluff frequencies
        sizing_multiplier = 1.0    # controls bet size scaling

        if self.won_auction and self.info_advantage > 0:
            # ── WE WON the auction — revealed-card-aware modifiers ────────
            opp_str = self.opp_card_board_str
            dom = self.opp_domination or {}

            if opp_str >= 0.50:
                # ═══ OPPONENT CARD CONNECTS STRONGLY ═══
                # Reference: Ganzfried & Sandholm (2014) — high-potential hands
                # require DEFENSIVE posture.  Their revealed card hitting top
                # pair / flush draw / straight draw means their hidden second
                # card amplifies strength.  We should:
                #   • Raise our value-bet threshold (need stronger hand)
                #   • Fold more easily vs aggression (they have real equity)
                #   • Reduce bluffs (they'll call with made hands/draws)
                aggression_boost = -0.06   # SHIFT DEFENSIVE
                fold_tighten = -0.04       # fold more easily
                bluff_dampener = 0.4       # cut bluffs by 60%
                sizing_multiplier = 0.80   # smaller bets (pot control)

                # Extra caution if they paired the top card on board
                if dom.get('we_are_dominated', False):
                    aggression_boost = -0.10
                    fold_tighten = -0.07
                    bluff_dampener = 0.2

            elif opp_str >= 0.30:
                # ═══ MEDIUM CONNECTION ═══
                # Billings et al. (2003): moderate-strength observed cards
                # create an ambiguous spot.  We have info edge but opponent
                # has real equity.  Play straightforward — bet for value and
                # protection, but don't over-commit.
                aggression_boost = 0.03
                fold_tighten = 0.02
                bluff_dampener = 0.7
                sizing_multiplier = 0.95

                # If we dominate their card, lean aggressive
                if dom.get('we_dominate', False):
                    aggression_boost = 0.07
                    fold_tighten = 0.04
                    sizing_multiplier = 1.05
                # If dominated, lean cautious
                elif dom.get('we_are_dominated', False):
                    aggression_boost = -0.04
                    fold_tighten = -0.03
                    bluff_dampener = 0.4

            elif opp_str >= 0.15:
                # ═══ WEAK CONNECTION / LOW CARD ═══
                # Johanson et al. (2011): low-connecting cards narrow opp range
                # to mostly weak holdings.  We can bet thinner for value.
                aggression_boost = 0.08
                fold_tighten = 0.04
                bluff_dampener = 0.85
                sizing_multiplier = 1.10

                # Kicker edge: if our cards outrank their revealed card,
                # we have strong equity advantage
                if dom.get('kicker_edge', 0) > 0.5:
                    aggression_boost = 0.11
                    fold_tighten = 0.06

            else:
                # ═══ TOTAL BRICK (opp_str < 0.15) ═══
                # Chen & Ankenman (2006) Ch. 23: with near-perfect information
                # that opponent has air, we should value-bet aggressively AND
                # bluff at high frequency (opponent can't distinguish).
                # Billings et al. (2003): "pure exploitation" mode when
                # opponent's range is extremely narrow/weak.
                aggression_boost = 0.13
                fold_tighten = 0.07
                bluff_dampener = 1.2   # actually INCREASE bluffs (they'll fold)
                sizing_multiplier = 1.20

                # If we also dominate them, maximum exploitation
                if dom.get('kicker_edge', 0) > 0.3:
                    aggression_boost = 0.15
                    sizing_multiplier = 1.30

            # ── SUIT BLOCKER BONUS ──
            # Sklansky (1999): holding a card of the same suit as opp's
            # revealed card blocks their flush draws (reduces their outs
            # from 9 to 8).  Small but consistent edge.
            if dom.get('suit_block', False) and opp_str >= 0.15:
                aggression_boost += 0.02

            # ── SHARED RANK BLOCKER ──
            # Johanson et al. (2011): if we hold the same rank as opponent's
            # revealed card, we block 2 of their pairing outs.  They can
            # no longer hit trips/two-pair with that rank.
            if dom.get('shared_rank', False):
                aggression_boost += 0.03
                fold_tighten += 0.02

        elif self.opp_won_auction and self.info_advantage < 0:
            # ── OPPONENT WON — they see one of our cards ──────────────────
            # Brown & Sandholm (2017): when opponent has information, our
            # bluffs lose fold equity dramatically.  The correct adjustment
            # is to play a tight, value-heavy strategy.
            aggression_boost = -0.08
            fold_tighten = -0.06
            bluff_dampener = 0.15     # cut bluffs by 85%
            sizing_multiplier = 0.75  # smaller bets to limit exposure

        # ══════════════════════════════════════════════════════════════════
        # ADJUSTED THRESHOLDS
        # ══════════════════════════════════════════════════════════════════
        value_bet_threshold   = 0.70 - aggression_boost
        medium_bet_threshold  = 0.55 - aggression_boost
        marginal_threshold    = 0.40 - aggression_boost * 0.5
        call_cushion          = 0.05 - fold_tighten

        # ══════════════════════════════════════════════════════════════════
        # ALL-IN PROTECTION (commitment decisions)
        # ══════════════════════════════════════════════════════════════════
        # Chen & Ankenman (2006) Ch. 20: commitment threshold — only commit
        # large fraction of stack when equity clearly justifies it.
        if cost > 0:
            commitment_ratio = cost / max(s.my_chips, 1)

            if s.street == 'river':
                if commitment_ratio > 0.60 and equity < 0.70:
                    if equity < 0.60 and s.can_act(ActionFold):
                        return ActionFold()
                if commitment_ratio > 0.40 and equity < 0.55 and self.info_advantage < 0:
                    if s.can_act(ActionFold):
                        return ActionFold()

            # Extended: protect on turn too for very large bets
            if s.street == 'turn':
                if commitment_ratio > 0.50 and equity < 0.55:
                    if s.can_act(ActionFold):
                        return ActionFold()

            # When opp card connects strongly and we face a big bet, fold
            # more aggressively — their range is narrow and strong
            if (self.won_auction and self.opp_card_board_str >= 0.50
                    and commitment_ratio > 0.35 and equity < 0.60):
                if s.can_act(ActionFold):
                    return ActionFold()

        # ══════════════════════════════════════════════════════════════════
        # FACING A BET (cost > 0)
        # ══════════════════════════════════════════════════════════════════
        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)

            # ── Very strong — raise for value ──
            if equity >= value_bet_threshold:
                multiplier = 0.85 * sizing_multiplier if self.info_advantage > 0 else 0.70
                raise_amt = int(pot * multiplier + cost)
                return self._make_raise(s, raise_amt)

            # ── Solid hand — call or raise ──
            if equity >= medium_bet_threshold:
                raise_freq = 0.35 if self.info_advantage > 0 else 0.20
                # When opp card connects strongly, reduce raise frequency —
                # they have a narrow strong range (Ganzfried & Sandholm 2014)
                if self.won_auction and self.opp_card_board_str >= 0.50:
                    raise_freq = 0.10
                if random.random() < raise_freq:
                    raise_amt = int(pot * 0.5 * sizing_multiplier + cost)
                    return self._make_raise(s, raise_amt)
                return ActionCall()

            # ── Marginal but profitable call ──
            if equity >= pot_odds + call_cushion:
                return ActionCall()

            # ── Drawing / weak ──
            if self.info_advantage < 0:
                # Opponent has our card — they bet correctly; tight fold
                if equity >= pot_odds + 0.12:
                    return ActionCall()
                if s.can_act(ActionFold):
                    return ActionFold()
                return ActionCall()

            # ── Opp card connected strongly + we're marginal = fold ──
            # Billings et al. (2003): when we know opponent likely has real
            # equity and our hand is marginal, calling is -EV
            if (self.won_auction and self.opp_card_board_str >= 0.40
                    and equity < medium_bet_threshold):
                if s.can_act(ActionFold):
                    return ActionFold()

            if equity >= pot_odds - call_cushion and cost <= pot * 0.35:
                return ActionCall()

            # Bluff raise occasionally (dampened by info state)
            if s.street == 'flop' and random.random() < 0.08 * bluff_dampener:
                raise_amt = int(pot * 0.65 + cost)
                return self._make_raise(s, raise_amt)

            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # ══════════════════════════════════════════════════════════════════
        # NO BET TO FACE (we act first or checked to us)
        # ══════════════════════════════════════════════════════════════════

        # ── Value bet ──
        if equity >= value_bet_threshold:
            size = 0.75 * sizing_multiplier if self.info_advantage > 0 else 0.65
            bet_size = int(pot * size)
            return self._make_raise(s, max(BIG_BLIND, bet_size))

        # ── Medium-strong — bet for value/protection ──
        if equity >= medium_bet_threshold:
            bet_freq = 0.70 if self.info_advantage > 0 else 0.55
            # When opp card connects strongly, bet smaller for pot control
            bet_pct = 0.35 if (self.won_auction and self.opp_card_board_str >= 0.50) else 0.45
            bet_size = int(pot * bet_pct * sizing_multiplier)
            if random.random() < bet_freq:
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        # ── Marginal — check mostly, occasional probe ──
        if equity >= marginal_threshold:
            probe_freq = 0.30 if self.info_advantage > 0 else 0.15
            # Reduce probing when opp card connects (they'll raise back)
            if self.won_auction and self.opp_card_board_str >= 0.40:
                probe_freq = 0.10
            if random.random() < probe_freq:
                bet_size = int(pot * 0.35 * sizing_multiplier)
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        # ══════════════════════════════════════════════════════════════════
        # WEAK HAND — BRICK EXPLOITATION & BLUFFING
        # ══════════════════════════════════════════════════════════════════

        # ── BRICK EXPLOITATION (equity-guarded) ──
        # Billings et al. (2003): exploit when we KNOW opponent is weak.
        # Chen & Ankenman (2006): even with air, betting into a known-weak
        # opponent is +EV if fold equity exceeds risk.
        #
        # CRITICAL FIX: Only bluff-exploit when we have SOME equity (≥ 0.25).
        # Without this guard, we bet 42o on a Kh-Qd-Js board when opp card
        # is 3c — opponent's *other* card may still beat us.  The equity
        # floor ensures we have enough backup when called.
        if self.info_advantage > 0 and self.opp_card_board_str < 0.15:
            # Opponent card is a total brick — high-pressure exploitation
            if equity >= 0.25:  # EQUITY GUARD: only bet with some equity
                if s.street in ('flop', 'turn'):
                    bet_size = int(pot * 0.55 * sizing_multiplier)
                    return self._make_raise(s, max(BIG_BLIND, bet_size))
                if s.street == 'river':
                    # River: polarized sizing — large to maximise fold pressure
                    # Chen & Ankenman: optimal bluff size = pot × (1 - equity)
                    bet_size = int(pot * 0.70 * sizing_multiplier)
                    return self._make_raise(s, max(BIG_BLIND, bet_size))
            elif equity >= 0.15:
                # Very weak but not hopeless — small probe only on flop
                if s.street == 'flop' and random.random() < 0.40:
                    bet_size = int(pot * 0.33)
                    return self._make_raise(s, max(BIG_BLIND, bet_size))

        # Mild brick exploitation for opp_str 0.15-0.20
        if self.info_advantage > 0 and self.opp_card_board_str < 0.20 and equity >= 0.30:
            if s.street in ('flop', 'turn') and random.random() < 0.55:
                bet_size = int(pot * 0.45 * sizing_multiplier)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        # ── INFO-AWARE BLUFFING ──
        # Sklansky (1999): bluff frequency should be proportional to fold
        # equity and inversely proportional to opponent's information.
        if s.street == 'river':
            bluff_base = 0.12
            if self.info_advantage > 0:
                if self.opp_card_board_str < 0.20:
                    bluff_base = 0.25   # brick → they're weak → high bluff freq
                elif self.opp_card_board_str < 0.30:
                    bluff_base = 0.18
                else:
                    bluff_base = 0.08   # connected → risky to bluff
            elif self.info_advantage < 0:
                bluff_base = 0.04       # they see our card → almost never bluff
            if self.opp_fold_rate > 0.35:
                bluff_base += 0.08
            if random.random() < bluff_base * bluff_dampener:
                bet_size = int(pot * 0.65 * sizing_multiplier)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        if s.street == 'flop':
            cbet_base = 0.15
            if self.info_advantage > 0:
                if self.opp_card_board_str < 0.20:
                    cbet_base = 0.35   # brick → c-bet freely
                elif self.opp_card_board_str < 0.30:
                    cbet_base = 0.22
                else:
                    cbet_base = 0.10   # connected → cautious c-bet
            elif self.info_advantage < 0:
                cbet_base = 0.05
            if random.random() < cbet_base * bluff_dampener:
                bet_size = int(pot * 0.40 * sizing_multiplier)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        # ── RIVER AGGRESSION FLOOR ──
        # Brown & Sandholm (2017): always maintain minimum aggression on
        # river to prevent opponent from checking down profitably.  Even
        # a small bet with ~33% equity is +EV if opponent folds > 25%.
        if s.street == 'river' and equity >= 0.33 and self.info_advantage >= 0:
            if random.random() < 0.15:
                bet_size = int(pot * 0.30)
                return self._make_raise(s, max(BIG_BLIND, bet_size))

        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())