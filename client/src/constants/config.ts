type ChartConfigType = {
  [key: string]: {
    days: number;
    weeks: number;
    months: number;
    years: number;
    resolution: string;
  };
};

const chartConfig: ChartConfigType = {
  '1D': { days: 1, weeks: 0, months: 0, years: 0, resolution: '1' },
  '1W': { days: 0, weeks: 1, months: 0, years: 0, resolution: '15' },
  '1M': { days: 0, weeks: 0, months: 1, years: 0, resolution: '60' },
  '1Y': { days: 0, weeks: 0, months: 0, years: 1, resolution: 'D' },
};

export default chartConfig;
