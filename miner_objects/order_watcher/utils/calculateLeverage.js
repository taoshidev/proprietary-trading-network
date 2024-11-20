export function calculateLeverage(order, balance) {
	const currentOrderValue = order.price * order.amount;

	const accountBalanceUSD = balance.total.USDC + balance.total.USDT + (balance.total.BTC * balance.free.USDT / balance.free.BTC) + (balance.total.ETH * balance.free.USDT / balance.free.ETH);

	const openPositionValueUSD = Object.values(balance.used).reduce((acc, value) => acc + value, 0);

	// return leverage
	return currentOrderValue / (openPositionValueUSD + accountBalanceUSD + currentOrderValue);
}