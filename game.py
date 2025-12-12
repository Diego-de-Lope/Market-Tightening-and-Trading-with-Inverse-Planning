"""
Game class for Trade or Tighten market game.

TODO: POMDP extension - extend state to include belief states
TODO: POMDP extension - add observation generation methods
TODO: Inverse planning - add methods to track observable vs hidden information
"""

class Game:
    """
    Represents the market state and game mechanics.

    Attributes:
        bid: current best bid (float or None)
        ask: current best ask (float or None)
        turn: current trader index (0, 1, 2 for A, B, C)
        start_player: trader who started current round
        history: list of past public actions
        round: round counter (increments only after a trade)
    """

    min_tighten = 1.0  # Minimum bid/ask improvement for tighten

    def __init__(self, bid=None, ask=None, turn=0, start_player=0, history=None, round=0):

        self.bid = bid
        self.ask = ask
        self.turn = turn
        self.start_player = start_player
        self.history = history if history is not None else []
        self.round = round

    def copy(self):
        """
        Create a copy of the game state.

        Returns:
            Game: A new Game instance with copied state
        """
        return Game(
            bid=self.bid,
            ask=self.ask,
            turn=self.turn,
            start_player=self.start_player,
            history=self.history.copy(),
            round=self.round
        )

    def get_valid_actions(self):
        """
        Return list of valid actions for current state.

        Args:
            agent_index: index of the agent whose turn it is

        Returns:
            list: List of valid action dictionaries
        """
        valid_actions = []

        # tighten market actions
        if self.bid is not None:
            valid_actions.append({"move": "tighten", "side": "bid"})
        if self.ask is not None:
            valid_actions.append({"move": "tighten", "side": "ask"})

        # If no bid/ask exists, agent can set initial quote
        if self.bid is None and self.ask is None:
            valid_actions.append({"move": "tighten", "side": "bid"})
            valid_actions.append({"move": "tighten", "side": "ask"})

        # execute trade actions
        if self.ask is not None:
            valid_actions.append({"move": "trade", "side": "buy"})
        if self.bid is not None:
            valid_actions.append({"move": "trade", "side": "sell"})

        return valid_actions

    def transition(self, action, agent_index):
        """
        Pure deterministic transition function.

        Args:
            action: action dictionary (tighten or trade)
            agent_index: index of agent taking the action

        Returns:
            Game: New Game instance with updated state

        Raises:
            ValueError: If action is invalid
        """
        new_game = self.copy()

        if action["move"] == "tighten":
            side = action["side"]
            price = action.get("price")

            if side == "bid":
                if price is None:
                    raise ValueError("Tighten bid action must include 'price'")
                if self.bid is not None and price <= self.bid:
                    raise ValueError(f"New bid {price} must be > current bid {self.bid}")
                if self.ask is not None and price >= self.ask:
                    raise ValueError(f"New bid {price} must be < current ask {self.ask}")
                new_game.bid = price
            elif side == "ask":
                if price is None:
                    raise ValueError("Tighten ask action must include 'price'")
                if self.ask is not None and price >= self.ask:
                    raise ValueError(f"New ask {price} must be < current ask {self.ask}")
                if self.bid is not None and price <= self.bid:
                    raise ValueError(f"New ask {price} must be > current bid {self.bid}")
                new_game.ask = price
            else:
                raise ValueError(f"Invalid side for tighten: {side}")

            # advance turn
            new_game.turn = (new_game.turn + 1) % 3
            new_game.history.append(action)

        elif action["move"] == "trade":
            side = action["side"]

            # validate trade is possible
            if side == "buy" and self.ask is None:
                raise ValueError("Cannot buy: no ask exists")
            if side == "sell" and self.bid is None:
                raise ValueError("Cannot sell: no bid exists")

            # reset to new round dynamics
            new_game.bid = None
            new_game.ask = None
            new_game.round += 1
            new_game.start_player = (agent_index + 1) % 3
            new_game.turn = new_game.start_player
            new_game.history.append(action)

        else:
            raise ValueError(f"Invalid action type: {action['type']}")

        return new_game

    def is_terminal(self):
        """
        Check if game should end (optional, for future use).

        Returns:
            bool: True if game should end (currently always False)
        """
        # TODO: POMDP extension - add terminal conditions if needed
        return False

    def __repr__(self):
        return (f"Game(bid={self.bid}, ask={self.ask}, turn={self.turn}, "
                f"start_player={self.start_player}, round={self.round}, "
                f"history_len={len(self.history)})")
