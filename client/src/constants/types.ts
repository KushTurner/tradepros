export type HistoricalData = {
  c: number[]; // Array of closing prices
  h: number[]; // Array of high prices
  l: number[]; // Array of low prices
  o: number[]; // Array of open prices
  s: string; // Status
  t: number[]; // Array of Unix timestamps
  v: number[]; // Array of volumes
};

export type DataObject = {
  [filter: string]: { value: number; date: string }[];
};
