function formatMarketCap(num: number) {
  const trillion = 10 ** 6;
  const billion = 10 ** 3;
  const million = 1;

  function truncateToTwoDecimalPlaces(number: number) {
    return Math.floor(number * 100) / 100;
  }

  if (num >= trillion) {
    return `${truncateToTwoDecimalPlaces(num / trillion)}T`;
  }
  if (num >= billion) {
    return `${truncateToTwoDecimalPlaces(num / billion)}B`;
  }
  if (num >= million) {
    return `${truncateToTwoDecimalPlaces(num / million)}M`;
  }
  return num.toFixed(2);
}

export default formatMarketCap;
