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
    Precompute heads-up equity for all 169 canonical starting hand types via
    Monte-Carlo simulation using eval7.  Called once at bot startup.

    500 iters × 169 hand types ≈ 84,500 evaluations — typically < 1 second.
    Returns dict mapping canonical key → float equity in [0, 1].
    '''
    all_cards = eval7.Deck().cards   # list of 52 eval7.Card objects
    table: dict = {}

    for i, r1 in enumerate(_RANKS):
        for j in range(i, len(_RANKS)):
            r2 = _RANKS[j]
            is_pair = (r1 == r2)

            if is_pair:
                candidates = [(r1 + r2, [eval7.Card(r1 + 's'), eval7.Card(r1 + 'h')])]
            else:
                candidates = [
                    (r1 + r2 + 's', [eval7.Card(r1 + 's'), eval7.Card(r2 + 's')]),
                    (r1 + r2 + 'o', [eval7.Card(r1 + 's'), eval7.Card(r2 + 'h')]),
                ]

            for key, hand_ev7 in candidates:
                dead = set(hand_ev7)
                deck = [c for c in all_cards if c not in dead]   # 50 cards
                wins = ties = total = 0
                for _ in range(iters_per_hand):
                    random.shuffle(deck)
                    opp   = deck[:2]
                    board = deck[2:7]
                    my_sc  = eval7.evaluate(hand_ev7 + board)
                    opp_sc = eval7.evaluate(opp      + board)
                    if my_sc > opp_sc:
                        wins += 1
                    elif my_sc == opp_sc:
                        ties += 1
                    total += 1
                table[key] = (wins + 0.5 * ties) / total

    return table


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
        delta_equity = 0.08 * uncertainty  # up to ~8% absolute equity gain

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
        self._auction_my_bid = bid
        return ActionBid(bid)

    # -------------------------------------------------------------------
    # Pre-flop strategy  (equity-based, not Chen formula)
    # -------------------------------------------------------------------
    def _preflop_action(self, game_info: GameInfo, s: PokerState):
        '''
        Thresholds derived from actual HU preflop equity ranges:
          >= 0.65 : Premium  — AA(85%), KK(82%), QQ(80%), JJ(77%), AKs(67%), AKo(65%)
          >= 0.58 : Strong   — TT(75%), 99(72%), AQs(66%), AQo(64%), AJs(65%), KQs(63%)
          >= 0.53 : Above avg— 88(69%), 77(66%), ATs(64%), KJs(61%), KQo(61%)
          >= 0.49 : Marginal — 66(63%), 55(59%), KTo, QJo, suited connectors
          <  0.49 : Weak     — dominated hands, large off-suit gaps
        '''
        eq   = self.preflop_equity   # float in [0, 1]
        cost = s.cost_to_call

        # --- Facing a raise (cost > 0) ---
        if cost > 0:
            pot_odds = cost / max(s.pot + cost, 1)

            # Premium: always 3-bet
            if eq >= 0.65:
                raise_to = s.opp_wager + max(cost * 3, BIG_BLIND * 4)
                return self._make_raise(s, raise_to)

            # Strong: call, occasionally 3-bet for balance
            if eq >= 0.58:
                if pot_odds < 0.35 or cost <= BIG_BLIND * 4:
                    if eq >= 0.62 and random.random() < 0.3:
                        return self._make_raise(s, s.opp_wager + BIG_BLIND * 3)
                    return ActionCall()
                if random.random() < 0.12:   # pot odds bad but float occasionally
                    return ActionCall()
                return ActionFold()

            # Above average: call small raises; fold big ones
            if eq >= 0.53:
                if cost <= BIG_BLIND * 3:
                    return ActionCall()
                if cost <= BIG_BLIND * 5 and random.random() < 0.25:
                    return ActionCall()
                return ActionFold()

            # Marginal: call very small raises; fold big ones
            if eq >= 0.49:
                if cost <= BIG_BLIND * 2:
                    return ActionCall()
                return ActionFold()

            # Weak: fold (occasional blind defence bluff)
            if cost <= BIG_BLIND and random.random() < 0.15:
                return ActionCall()
            return ActionFold()

        # --- No raise to face (BB option after SB limp, or SB openable) ---
        # Premium: pot-sized open raise
        if eq >= 0.65:
            return self._make_raise(s, BIG_BLIND * 4)

        # Strong: standard 2.5x open
        if eq >= 0.58:
            return self._make_raise(s, BIG_BLIND * 3)

        # Above average: mixed raise/check to balance range
        if eq >= 0.53:
            if random.random() < 0.6:
                return self._make_raise(s, BIG_BLIND * 2)
            return ActionCheck()

        # Marginal: check or occasionally min-raise
        if eq >= 0.49:
            if random.random() < 0.2:
                return self._make_raise(s, BIG_BLIND * 2)
            return ActionCheck()

        # Weak: check
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
            # Play tighter: less bluffing, higher equity thresholds to bet.
            aggression_boost = -0.05
            fold_tighten = -0.02
            bluff_dampener = 0.4     # cut bluffs by 60%

        # Adjusted thresholds
        value_bet_threshold   = 0.70 - aggression_boost
        medium_bet_threshold  = 0.55 - aggression_boost
        marginal_threshold    = 0.40 - aggression_boost * 0.5
        call_cushion          = 0.05 - fold_tighten

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
