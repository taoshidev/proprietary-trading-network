from vali_objects.vali_dataclasses.order import Order

class PriceUtils:
    """Class for utilities for price calculations"""
    @staticmethod
    def currency_converter(
            amount: float,
            base: str,
            quote: str,
            from_currency: str,
            to_currency: str,
            bid_price: float = 0,
            ask_price: float = 0
    ) -> float:
        """
        Convert between currencies in a given trading pair using available bid/ask prices

        Args:
            amount: Amount to convert
            trade_pair: TradePair object containing base and quote currencies
            from_currency: 3-letter ISO currency code to convert from (e.g. "USD")
            to_currency: 3-letter ISO currency code to convert to (e.g. "EUR")
            bid_price: Best available buying price for the currency pair (optional)
            ask_price: Best available selling price for the currency pair (optional)

        Returns:
            Converted amount in target currency

        Raises:
            ValueError: For invalid currency inputs, when required price is not provided,
                       or when provided prices are not positive
        """

        # Validate inputs
        if from_currency == to_currency:
            return amount

        if base is None or quote is None:
            raise ValueError(f"Trade pair {base}/{quote} does not have valid base/quote currencies")

        valid_currencies = {base, quote}

        if from_currency not in valid_currencies or to_currency not in valid_currencies:
            raise ValueError(f"Currencies must belong to {base}/{quote}. "
                             f"Converting: {from_currency}->{to_currency}")

        # Validate provided prices are positive
        if bid_price <= 0 and ask_price <= 0:
            raise ValueError("Bid price or ask price must be positive")

        # Determine conversion direction and required price
        if from_currency == base and to_currency == quote:
            # Sell base at bid price
            if bid_price == 0:
                raise ValueError("Bid price required for base->quote conversion")
            return amount * bid_price

        if from_currency == quote and to_currency == base:
            # Buy base with quote at ask price
            if ask_price == 0:
                raise ValueError("Ask price required for quote->base conversion")
            return amount / ask_price

        raise ValueError(f"Invalid conversion direction for {base}/{quote}: "
                         f"{from_currency}->{to_currency}")

    @staticmethod
    def convert_currency_for_order(amount: float, order: Order, to_USD: bool = True) -> float:
        if not order.trade_pair.is_forex:
            return amount

        if to_USD:
            if order.quote_usd_bid == 0:
                base = order.trade_pair.quote
                quote = "USD"
            else:
                base = "USD"
                quote = order.trade_pair.quote

            currency_amount = PriceUtils.currency_converter(
                amount=amount,
                base=base,
                quote=quote,
                from_currency=order.trade_pair.quote,
                to_currency='USD',
                bid_price=order.quote_usd_bid,
                ask_price=order.usd_quote_ask,
            )
        else:
            if order.base_usd_ask == 0:
                base = order.trade_pair.quote
                quote = "USD"
            else:
                base = "USD"
                quote = order.trade_pair.quote

            currency_amount = PriceUtils.currency_converter(
                amount=amount,
                base=base,
                quote=quote,
                from_currency='USD',
                to_currency=order.trade_pair.base,
                bid_price=order.usd_base_bid,
                ask_price=order.base_usd_ask,
            )
        return currency_amount


