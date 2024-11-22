export function isFilled(order) {
  return (
    order?.info?.orderStatus === "Filled" && order?.info?.state === "filled"
  );
}
