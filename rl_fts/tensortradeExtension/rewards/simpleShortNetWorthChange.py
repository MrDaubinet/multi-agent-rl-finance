from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.env.generic import TradingEnv

from rl_fts.tensortradeExtension.actions.sh import SH

class SimpleShortNetWorthChange(TensorTradeRewardScheme):
    """A simple reward scheme that rewards the agent based on the change in its networth
    """

    def __init__(self):
        self.net_worth_history = []
        self.previous_net_worth = 0

    def reward(self, env: TradingEnv) -> float:
        return self.get_reward(env.action_scheme)

    def get_reward(self, short_action_scheme: SH) -> float:
        """
        Parameters
        ----------
        short_action_scheme : `SH`
            The action scheme used for managing simple short positions.
        Returns
        -------
        float
            The difference in networth as profit / loss 
        """
        asset_balance = short_action_scheme.asset.balance.convert(short_action_scheme.exchange_pair)
        cash_balance = short_action_scheme.cash.balance
        deposit_margin = short_action_scheme.deposit_margin.balance
        borrowed_cash = short_action_scheme.borrow_quantity.convert(short_action_scheme.exchange_pair)
        net_worth = (asset_balance + cash_balance + deposit_margin - borrowed_cash).as_float()
        net_worth_change = net_worth - self.previous_net_worth
        self.previous_net_worth = net_worth
        self.net_worth_history.append(net_worth)

        return net_worth_change

    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.net_worth_history = []
        self.previous_net_worth = 0
        

