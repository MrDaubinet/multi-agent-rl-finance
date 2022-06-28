from tensortrade.env.default.actions import TensorTradeActionScheme
from gym.spaces import Discrete
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Quantity, ExchangePair, TradingPair
from tensortrade.env.generic import TradingEnv

import logging

from typing import Any

from tensortrade.oms.orders import (
    Order,
    proportion_order,
    market_order,
    TradeSide,
    TradeType
)

class SH(TensorTradeActionScheme):
    """A simple discrete action scheme where the only options available are to short
    or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    margin : `Wallet`
        The wallet to hold funds in the margin account.
    borrow_limit : `float`
        A limit specifying the amount that the agent can borrow as
        a percentage of the security's purchased price.
    maintenance_margin : `float`
        The percentage of borrowed funds that has to be available in the margin account
    margin_interest : `float`
        The interest applied per time step for this short position
    Notes :
        A stopper (or a custom Tensortrade Environment) must be implemented end the episode if
            the maintenance_margin is exceeded after an action.
        The minimum or initial margin must be at least $2,000 in cash or securities
        There are specific rules which dictate short selling
            uptick rule in the US
            capital adequacy norms rule in the UK
            (I will exclude this rule from my experiments)
        Any dividends that would have been received in the short position are owed to the investor
            (I will exclude this rule from my experiments)
        There is a rule stipulating that the cash produced from shorting a position (selling a borrowed asset)
        cannot be used to purchase other assets
    """

    registered_name = "sh"

    def __init__(self,
            cash: 'Wallet',
            asset: 'Wallet',
            broker_asset: 'Wallet',
            broker_cash: 'Wallet',
            deposit_margin: 'Wallet',
            profit_wallet: 'Wallet',
            broker_portfolio: 'Portfolio',
            borrow_limit: 'float' = 0.5,
            maintenance_margin: 'float' = 0.25,
            margin_interest: 'float' = (0.06/365),
            broker_fee: 'float' = 0.01,
            minimum_short_deposit: 'float' = None
        ):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.broker_asset = broker_asset
        self.broker_cash = broker_cash
        self.deposit_margin = deposit_margin
        self.profit_wallet = profit_wallet
        broker_portfolio=broker_portfolio
        self.borrow_limit = borrow_limit
        self.maintenance_margin = maintenance_margin
        self.margin_interest = margin_interest
        self.minimum_short_deposit = minimum_short_deposit
        self.broker_fee = broker_fee
        
        self.commission = cash.exchange.options.commission

        self.borrow_price = None
        self.borrow_quantity = None
        self.cash_quantity = None

        self.short_enter_order = None
        self.short_exit_order = None
        
        self.complete_exit_order = False
        self.complete_enter_order = False
        self.short_remainder = False

        # USD-TTC
        traing_pair_usd_ttc = TradingPair(self.cash.balance.instrument, self.broker_asset.instrument)
        # sine-wave-USD-TTC
        self.exchange_pair_usd_ttc = ExchangePair(self.cash.exchange, traing_pair_usd_ttc)

        self.listeners = []
        self.action = -1

        # TODO: implement the profit wallet

    @property
    def action_space(self):
        """
          0 -> Enter short position (Buy everything)
          1 -> Exit short position (Sell everything)
          Note:
            If the action is the same as the previous action, we implement a hold
        """
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def perform(self, env: 'TradingEnv', action: 'Any') -> None:
        """Performs the action on the given environment.

        Under the TT action scheme, the subclassed action scheme is expected
        to provide a method for getting a list of orders to be submitted to
        the broker for execution in the OMS.

        Parameters
        ----------
        env : 'TradingEnv'
            The environment to perform the action on.
        action : Any
            The specific action selected from the action space.
        """
        orders = self.get_orders(action, self.portfolio)
        all_orders_completed = True

        for order in orders:
            if order:
                if not order.is_complete:
                    logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                    self.broker.submit(order)
                else:
                    all_orders_completed = False

        self.broker.update()

        # If the last action created an exit order
        if self.complete_exit_order and all_orders_completed:
            # complete the short transfers
            self.completeShort()

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        """Gets the list of orders to be submitted for the given action.

        Parameters
        ----------
        action : Any
            The action to be interpreted.
        portfolio : 'Portfolio'
            The portfolio defined for the environment.

        Returns
        -------
        List[Order]
            A list of orders to be submitted to the broker.
        """
        
        if self.complete_enter_order or self.complete_exit_order:
            print("This may be an issue -> complete_exit_order = True")
            return [self.short_enter_order, self.short_exit_order]

        # order = None
        self.short_enter_order = None
        self.short_exit_order = None

        # only place an order if the action has changed (if the action hasn't changed, then we are holding)
        if abs(action - self.action) > 0:
            # if we are entering a short position
            if action == 0:
                self.short_enter_order = self.enterShort(portfolio)
            # if we are exiting the short
            if action == 1 and self.action == 0:
                # create a market order to buy back the same quantity of stock with cash (from cash to asset)
                self.short_exit_order = self.exitShort(portfolio)
            self.action = action
        else:
            # if we are currently in a short position
            if self.action == 0:
                self.short_exit_order = self.maintainShort()
        for listener in self.listeners:
            listener.on_action(action)

        return [self.short_enter_order, self.short_exit_order]

    def enterShort(self, portfolio):
        # lets check that we actually have positive cash in the cash wallet
        if self.cash.balance.as_float() == 0:
            return None  # Otherwise just return an empty order list
            # We can add other requirements here like the minimum value
        if self.minimum_short_deposit is not None and self.cash.balance < self.minimum_short_deposit:
            return None

        # The borrowed asset must be created  (1 / borrow_limit * cash value) and placed in the brokers wallet
        borrow_size = 1 / self.borrow_limit * self.cash.balance.as_float()
        self.cash_quantity = Quantity(self.cash.balance.instrument, borrow_size)
        self.borrow_quantity = self.cash_quantity.convert(
            self.exchange_pair_usd_ttc)
        # When the agent enters the short position, the borrow limit must be transfered (cash -> deposit_margin)
        self.transfer(
            source=self.cash,
            target=self.deposit_margin,
            quantity=self.cash.balance,
            commission=self.cash.balance * self.commission,
            reason="SHORT - BORROW DEPOSIT"
        )
        # Add the funds to the broker wallet (deposit)
        self.broker_asset.deposit(
            self.borrow_quantity,
            "Adding security to brokers wallet so that it can be \
            transfered to us to sell in a short position"
        )
        # transfer the borrowed funds from the broker (broker_asset -> asset)
        self.transfer(
            source=self.broker_asset,
            target=self.asset,
            quantity=self.broker_asset.balance,
            commission=0 * self.commission,
            reason="SHORT - BORROWED ASSET"
        )
        # broker fee for entering short
        self.transfer(
            source=self.deposit_margin,
            target=self.broker_cash,
            quantity=self.deposit_margin.balance * self.broker_fee,
            commission=0,
            reason="SHORT - BROKER FEE"
        )
        # The borrowed asset must be sold (asset -> cash)
        short_enter_order = proportion_order(
            portfolio,
            source=self.asset,
            target=self.cash,
            proportion=1
        )
        return short_enter_order

    def exitShort(self, portfolio: 'Portfolio'):
        short_exit_order = None
        current_asset_quantity = self.cash.balance.convert(self.exchange_pair_usd_ttc)
        # we need to check if we have enough cash to cover the borrow quantity
        if  current_asset_quantity > self.borrow_quantity: 
            # The short must be exited (cash -> asset)
            short_exit_order = Order(
                step=portfolio.clock.step,
                side=TradeSide.BUY,
                trade_type=TradeType.MARKET,
                exchange_pair=self.exchange_pair_usd_ttc,
                price=self.exchange_pair_usd_ttc.price,
                quantity=self.borrow_quantity.convert(self.exchange_pair_usd_ttc),
                portfolio=portfolio,
            )
        else:
            # transfer from deposit margin (deposit_margin -> cash)
            borrow_remainder_cash = (self.borrow_quantity - current_asset_quantity).convert(self.exchange_pair_usd_ttc)
            short_exit_commission = self.borrow_quantity.convert(self.exchange_pair_usd_ttc) * self.commission
            # infalte for commision on order
            borrow_remainder_cash += short_exit_commission
            # inflate for commision on transfer
            borrow_remainder_cash += borrow_remainder_cash * self.commission
            self.transfer(
                source=self.deposit_margin,
                target=self.cash,
                quantity=(borrow_remainder_cash+borrow_remainder_cash*self.commission),
                commission=borrow_remainder_cash * self.commission,
                reason="SHORT - BORROW REMAINDER"
            )
            # exit the short with our total cash value (cash -> asset)
            short_exit_order = Order(
                step=portfolio.clock.step,
                side=TradeSide.BUY,
                trade_type=TradeType.MARKET,
                exchange_pair=self.exchange_pair_usd_ttc,
                price=self.exchange_pair_usd_ttc.price,
                quantity=self.cash.balance,
                portfolio=portfolio,
            )
        # complete the exit order wallet transfers after the order is complete
        self.complete_exit_order = True
        return short_exit_order

    def completeShort(self):
        # transfer asset to broker (asset -> broker_asset)
        self.transfer(
            source=self.asset,
            target=self.broker_asset,
            quantity=self.asset.balance,
            commission=0,
            reason="SHORT - RETURN ASSET"
        )
        # broker fee for exiting short
        self.transfer(
            source=self.deposit_margin,
            target=self.broker_cash,
            quantity=self.deposit_margin.balance * self.broker_fee,
            commission=0,
            reason="SHORT - BROKER FEE"
        )
        # claim back deposit (transfer into cash)
        self.transfer(
            source=self.deposit_margin,
            target=self.cash,
            quantity=self.deposit_margin.balance,
            commission=self.deposit_margin.balance * self.commission,
            reason="SHORT - RETURN DEPOSIT"
        )
        self.complete_exit_order = False

    def maintainShort(self):
        # interest must be deducted from the short value
        interest: Quantity = self.cash_quantity * self.margin_interest
        self.transfer(
            source=self.deposit_margin,
            target=self.broker_cash,
            quantity=interest,
            commission=0,
            reason="SHORT - BORROWED INTEREST"
        )
        # if the deposit value is less than maintenance_margin % of the borrowed assets value
        if (self.deposit_margin.total_balance.as_float() - interest.as_float()) / self.borrow_quantity.convert(self.exchange_pair_usd_ttc).as_float() < self.maintenance_margin:
            # exit short
            return self.exitShort()
        return None

    def transfer(self, 
        source: Wallet, 
        target: Wallet, 
        quantity: Quantity, 
        commission: Quantity, 
        reason: str):
        '''
            A function used to tranfer funds from one wallet to another by calling 
            withdraw and deposit on each.
        '''
        if commission:
            # withdraw the commision
            source.withdraw(quantity=commission, reason=reason+" COMMISSION")
        # withdraw transfer quantity from source
        source.withdraw(quantity=quantity-commission, reason=reason)
        # deposit transfer quantity to target
        target.deposit(quantity=quantity-commission, reason=reason)

    def reset(self):
        super().reset()
        self.broker_asset.reset()
        self.broker_cash.reset()
        self.deposit_margin.reset()
        self.profit_wallet.reset()
        self.action = -1
