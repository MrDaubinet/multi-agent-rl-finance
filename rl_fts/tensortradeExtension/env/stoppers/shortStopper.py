
from tensortrade.env.generic import Stopper, TradingEnv
# from rl_fts.tensortradeExtension.actions.sh import SH

class ShortStopper(Stopper):
    """A stopper that stops an episode if the agents deposit margin goes below the maintenance margin limit

    Attributes
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    """

    def __init__(self):
        super().__init__()

    def borrow_limit_reached(self, env: 'TradingEnv'):
        action_scheme = env.action_scheme
        maintenance_margin = action_scheme.maintenance_margin
        deposit_margin = action_scheme.deposit_margin.total_balance.as_float()
        current_short_value = action_scheme.borrow_asset.convert(action_scheme.exchange_pair)
        margin_requirement = (1 + maintenance_margin)
        if current_short_value > 0:
            margin_threshold = margin_requirement * current_short_value
            if deposit_margin < margin_threshold:
                return True
        return False

    def cash_minimum_hit(self, env: 'TradingEnv'):
        action_scheme: SH = env.action_scheme
        return action_scheme.cash.balance.as_float() < action_scheme.minimum_short_deposit

    def stop(self, env: 'TradingEnv') -> bool:
        c1 = self.borrow_limit_reached(env)
        # c2 = self.cash_minimum_hit(env)
        stop = False
        if c1:
            stop = True
        return stop
