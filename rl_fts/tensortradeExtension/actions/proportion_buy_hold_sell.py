from tensortrade.env.default.actions import TensorTradeActionScheme
from gym.spaces import Discrete, Tuple, Box
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import ExchangePair, TradingPair

from tensortrade.oms.orders import (
    Order,
    proportion_order,
)

class PBSH(TensorTradeActionScheme):
    """
        An action scheme which has both continuos and discrete outputs.
        The options are to buy, sell, or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset
        # USD-TTC
        traing_pair = TradingPair(self.cash.balance.instrument, self.asset.balance.instrument)
        # sine-wave-USD-TTC
        self.exchange_pair = ExchangePair(self.cash.exchange, traing_pair)

        self.listeners = []
        self.action = -1
        self.proportion = -1

    @property
    def action_space(self):
        """
          0 -> Buy everything
          1 -> Sell everything
          Note:
            1. If the action is the same as the previous action, we implement a hold
            2. we do nothing until the agent requests a buy
        """
        return Tuple((Discrete(2), Box(1, 100, shape=(1,))))

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, actions: Tuple, portfolio: 'Portfolio') -> 'Order':
        action = actions[0]
        proportion = actions[1][0]
        order = None

        if abs(action - self.action) > 0:
            if action == 0:
                src = self.cash
                tgt = self.asset
            else:
                src = self.asset
                tgt = self.cash

            if src.balance == 0:  # We need to check, regardless of the proposed order, if we have balance in 'src'
                return []  # Otherwise just return an empty order list

            order = proportion_order(portfolio, src, tgt, proportion / 100)
            self.action = action
            self.proportion = proportion

        for listener in self.listeners:
            listener.on_action(actions)

        return [order]

    def reset(self):
        super().reset()
        self.action = -1
