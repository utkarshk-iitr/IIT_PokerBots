'''
Sneak Peek Hold'em Bot — "SneakBot"
A competitive poker bot for IIT Pokerbots 2026.

Strategy overview:
  - Pre-flop: Chen-formula-based hand ranking for open/call/raise decisions.
  - Auction:  Bid proportional to hand equity (Vickrey-optimal: bid true value).
  - Post-flop: Monte-Carlo equity estimation with eval7, pot-odds-based decisions.
  - Opponent modelling: tracks fold %, raise %, and average bid to adapt dynamically.
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

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def card_to_eval7(card_str: str) -> eval7.Card:
    '''Convert engine card string like "Ah" to eval7.Card.'''
    return eval7.Card(card_str)


def chen_score(hand: list[str]) -> float:
    '''
    Modified Chen formula for 2-card hand strength.
    Returns a numeric score; higher is better.
    '''
    rank_values = {
        'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
        '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5,
        '4': 2, '3': 1.5, '2': 1
    }
    rank_order = 'AKQJT98765432'

    r1, s1 = hand[0][0], hand[0][1]
    r2, s2 = hand[1][0], hand[1][1]

    v1 = rank_values[r1]
    v2 = rank_values[r2]

    score = max(v1, v2)

    # Pair bonus
    if r1 == r2:
        score = max(score * 2, 5)
        return score

    # Suited bonus
    suited = s1 == s2
    if suited:
        score += 2

    # Gap penalty
    idx1 = rank_order.index(r1)
    idx2 = rank_order.index(r2)
    gap = abs(idx1 - idx2) - 1  # 0 = connected

    if gap == 0:
        pass  # connected — no penalty
    elif gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap == 3:
        score -= 4
    else:
        score -= 5

    # Straight bonus for low connected cards
    if gap <= 1 and max(v1, v2) < 7:
        score += 1

    return score


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
        self.my_chen = 0.0

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
        self.my_chen = chen_score(current_state.my_hand)
        self.prev_opp_wager = BIG_BLIND if not current_state.is_bb else SMALL_BLIND
        self.prev_opp_chips = STARTING_STACK - self.prev_opp_wager
        self.hand_street_history = []

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
    # Auction strategy
    # -------------------------------------------------------------------
    def _auction_action(self, game_info: GameInfo, s: PokerState):
        '''
        Second-price auction: optimal to bid true value of information.
        The winner pays the LOSER's bid, so bidding true value is dominant.
        We estimate the value of information based on equity uncertainty
        and pot size, then bid accordingly.
        '''
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=100)

        # Information is most valuable at medium equity (~0.5) where the
        # decision is hardest; less valuable when we're clearly ahead/behind.
        uncertainty = 4.0 * equity * (1.0 - equity)  # peaks at 0.5, range [0,1]

        # Base value: fraction of pot proportional to information value.
        # In a Vickrey auction, bid your true value.
        # The effect of seeing 1 opponent card ~= 5-15% equity swing.
        # Value that at roughly pot × information_leverage.
        info_leverage = 0.10 * uncertainty  # ~0-10% of pot
        base_value = s.pot * info_leverage

        # Scale up slightly for medium-equity hands where peeking tips decisions
        if 0.35 <= equity <= 0.65:
            base_value *= 1.5

        # Don't bid more than we can afford
        max_bid = s.my_chips
        bid = int(max(0, min(base_value, max_bid)))

        # Very strong hands don't need information
        if equity > 0.85:
            bid = max(0, int(bid * 0.2))

        # Adapt to opponent's average bid — if they consistently overbid,
        # keep ours low and let them overpay.  If they bid low, we can
        # sometimes snipe cheaply.
        if len(self.opp_bids) >= 10:
            avg_opp = self.opp_avg_bid
            if avg_opp > s.pot * 0.3:
                # Opponent overbids — stay low, they'll burn chips
                bid = min(bid, int(avg_opp * 0.05))
            elif avg_opp < BIG_BLIND and 0.35 <= equity <= 0.65:
                # Opponent bids very low — we can win auction cheaply
                bid = max(bid, int(avg_opp + BIG_BLIND))

        return ActionBid(max(0, bid))

    # -------------------------------------------------------------------
    # Pre-flop strategy
    # -------------------------------------------------------------------
    def _preflop_action(self, game_info: GameInfo, s: PokerState):
        chen = self.my_chen
        cost = s.cost_to_call

        # --- Facing a raise (cost > 0) ---
        if cost > 0:
            pot_odds = cost / max(s.pot + cost, 1)

            # Premium hands — 3-bet / re-raise
            if chen >= 10:
                raise_to = s.pot + cost * 2
                return self._make_raise(s, raise_to)

            # Strong hands — call, sometimes raise
            if chen >= 7:
                if pot_odds < 0.35 or cost <= BIG_BLIND * 3:
                    if chen >= 8 and random.random() < 0.3:
                        return self._make_raise(s, s.opp_wager + BIG_BLIND * 3)
                    return ActionCall()
                # Pot odds too bad for a medium-strong hand
                if random.random() < 0.15:
                    return ActionCall()  # occasional float
                return ActionFold()

            # Mediocre hands — call small raises
            if chen >= 5:
                if cost <= BIG_BLIND * 2:
                    return ActionCall()
                if random.random() < 0.1:
                    return ActionCall()
                return ActionFold()

            # Weak hands — mostly fold
            if cost <= BIG_BLIND and random.random() < 0.25:
                return ActionCall()
            return ActionFold()

        # --- No raise to face (we can check or raise) ---
        # This happens when we are BB and SB just called
        if chen >= 9:
            raise_to = BIG_BLIND * 3
            return self._make_raise(s, raise_to)

        if chen >= 6:
            if random.random() < 0.35:
                return self._make_raise(s, BIG_BLIND * 2.5)
            return ActionCheck()

        # Weak — just check
        return ActionCheck()

    # -------------------------------------------------------------------
    # Post-flop strategy
    # -------------------------------------------------------------------
    def _postflop_action(self, game_info: GameInfo, s: PokerState):
        equity = monte_carlo_equity(s.my_hand, s.board, s.opp_revealed_cards, iters=MC_ITERS)
        pot = s.pot
        cost = s.cost_to_call

        # --- Facing a bet (cost > 0) ---
        if cost > 0:
            pot_odds = cost / max(pot + cost, 1)

            # We need equity > pot_odds to call profitably
            if equity >= 0.7:
                # Very strong — raise for value
                raise_amt = int(pot * 0.75 + cost)
                return self._make_raise(s, raise_amt)

            if equity >= 0.55:
                # Solid hand — call, sometimes raise
                if random.random() < 0.25:
                    raise_amt = int(pot * 0.5 + cost)
                    return self._make_raise(s, raise_amt)
                return ActionCall()

            if equity >= pot_odds + 0.05:
                # Marginal but profitable call
                return ActionCall()

            # Drawing / weak — fold unless getting great odds
            if equity >= pot_odds - 0.05 and cost <= pot * 0.3:
                return ActionCall()

            # Bluff raise occasionally on the flop
            if s.street == 'flop' and random.random() < 0.08:
                raise_amt = int(pot * 0.65 + cost)
                return self._make_raise(s, raise_amt)

            if s.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        # --- No bet to face (we act first or checked to us) ---
        if equity >= 0.7:
            # Value bet
            bet_size = int(pot * 0.7)
            return self._make_raise(s, max(BIG_BLIND, bet_size))

        if equity >= 0.55:
            # Medium-strong — bet for value/protection
            bet_size = int(pot * 0.45)
            if random.random() < 0.6:
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        if equity >= 0.4:
            # Marginal — check mostly, occasional small bet
            if random.random() < 0.2:
                bet_size = int(pot * 0.35)
                return self._make_raise(s, max(BIG_BLIND, bet_size))
            return ActionCheck()

        # Weak — check, occasionally bluff
        if s.street == 'river' and random.random() < 0.12:
            # River bluff
            bet_size = int(pot * 0.6)
            if self.opp_fold_rate > 0.35:
                bet_size = int(pot * 0.8)
            return self._make_raise(s, max(BIG_BLIND, bet_size))

        if s.street == 'flop' and random.random() < 0.15:
            # Continuation bet / bluff on flop
            bet_size = int(pot * 0.4)
            return self._make_raise(s, max(BIG_BLIND, bet_size))

        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
