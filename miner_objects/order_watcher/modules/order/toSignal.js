export function toSignal(order, leverage = 0) {
  return {
    trade_pair: order.info.symbol,
    order_type: order.side.toUpperCase() === "BUY" ? "LONG" : "SHORT",
    leverage: leverage,
  };
}
