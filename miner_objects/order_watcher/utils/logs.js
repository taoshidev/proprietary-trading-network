export const logUserBalance = (balance) => {
  const balances = ["USDC", "BTC", "ETH", "USDT"].map((currency) => ({
    Currency: currency,
    Free: balance[currency].free,
    Used: balance[currency].used,
    Total: balance[currency].total,
    Debt: balance[currency].debt,
  }));

  console.log("\nUser Balances:");
  console.table(balances);
};
