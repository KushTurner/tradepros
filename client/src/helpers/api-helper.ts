import { todaysDate } from './date-helper';

const {
  VITE_BASE_URL,
  VITE_FINNHUB_TOKEN,
  VITE_AWS_API_URL,
  VITE_AWS_API_KEY,
} = import.meta.env;

export function fetchStockProfile(stockId: string) {
  return fetch(
    `${VITE_BASE_URL}/stock/profile2?symbol=${stockId.toUpperCase()}&token=${VITE_FINNHUB_TOKEN}`
  ).then((response) => response.json());
}

export function fetchStockQuote(stockId: string) {
  return fetch(
    `${VITE_BASE_URL}/quote?symbol=${stockId.toUpperCase()}&token=${VITE_FINNHUB_TOKEN}`
  ).then((response) => response.json());
}

export function fetchStockFinancials(stockId: string) {
  return fetch(
    `${VITE_BASE_URL}/stock/metric?symbol=${stockId.toUpperCase()}&metric=all&token=${VITE_FINNHUB_TOKEN}`
  ).then((response) => response.json());
}

export function fetchStockPrediction(stockId: string) {
  return fetch(
    `${VITE_AWS_API_URL}/tradepros?ticker=${stockId.toUpperCase()}&date_to_predict=${todaysDate()}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': `${VITE_AWS_API_KEY}`,
      },
    }
  ).then((response) => response.json());
}

export function fetchHistoricalData(
  stockId: string,
  resolution: string,
  startTimestampUnix: number,
  endTimestampUnix: number
) {
  const historicalDataURL = `${VITE_BASE_URL}/stock/candle?symbol=${stockId.toUpperCase()}&resolution=${resolution}&from=${startTimestampUnix}&to=${endTimestampUnix}&token=${VITE_FINNHUB_TOKEN}`;

  return fetch(historicalDataURL).then((response) => response.json());
}
