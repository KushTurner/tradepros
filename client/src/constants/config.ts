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
  '1M': { days: 0, weeks: 0, months: 1, years: 0, resolution: '60' },
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
