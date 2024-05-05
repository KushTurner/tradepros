type ChartConfigType = {
  [key: string]: {
    days: number;
    weeks: number;
    months: number;
    years: number;
    resolution: string;
  };
};

export const chartConfig: ChartConfigType = {
  '1D': { days: 1, weeks: 0, months: 0, years: 0, resolution: '1' },
  '1W': { days: 0, weeks: 1, months: 0, years: 0, resolution: '15' },
  '1M': { days: 0, weeks: 0, months: 1, years: 0, resolution: 'H' },
  '1Y': { days: 0, weeks: 0, months: 0, years: 1, resolution: 'D' },
};

export const instructionsData = [
  {
    index: 1,
    header: 'Create An Account',
    description: 'Sign up with your email in under 2 minutes',
  },
  {
    index: 2,
    header: 'Analyse Companies',
    description: 'Determine which companies you want to invest into',
  },
  {
    index: 3,
    header: 'Start Investing Instantly',
    description:
      'Buy and Sell a variety of stocks with a $100,000 of virtual money',
  },
  {
    index: 4,
    header: 'Compete Against Your Friends',
    description: 'See who has profited the most using our leaderboard',
  },
];

export const errorMap: { [key: string]: string } = {
  'Firebase: Error (auth/invalid-email).': 'Invalid email address!',
  'Firebase: Error (auth/missing-password).': 'Password is required!',
  'Firebase: Error (auth/wrong-password).': 'Incorrect password!',
  'Firebase: Error (auth/user-not-found).': 'User not found!',
};

export type InvestmentObject = {
  name: string;
  shares: number;
  investment: number;
};

export type LeaderboardObject = {
  rank: string;
  username: string;
  balance: number;
};

export type StockObject = {
  companyName: string;
  ticker: string;
  logo: string;
  price: number;
  previousClose: number;
  marketCap: number;
  weekHigh: number;
  weekLow: number;
};

export type StockPredictionObject = {
  confidence: number;
  direction: string;
};

export type TradeHistoryObject = {
  name: string;
  action: string;
  amount: string;
  quantity: number;
  date: string;
};

export type PortfolioObject = {
  balance: number;
  username: string;
  created_at: string;
  balance_gain: number;
};

export type BuySellFormTypeParams = {
  type: string;
  logo: string;
  ticker: string;
  price: number;
  loading: boolean;
};

export interface ChartFilterProps {
  text: string;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}

export type InstructionsProps = {
  index: number;
  header: string;
  description: string;
};

export type SearchObject = {
  '1. symbol': string;
  '2. name': string;
};

export type WatchlistObject = {
  name: string;
  ticker: string;
  logo: string;
  finnhubIndustry: string;
};
