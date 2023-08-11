import { ColumnDef } from '@tanstack/react-table';

export type CompanyData = {
  company_name: string;
  stock_price: string;
  pc_change: number;
  highest: string;
  lowest: string;
};

export type PortfolioData = {
  company_name: string;
  no_of_shares: number;
  total_investment: string;
  current_val: string;
  gain_loss: string;
};

export const companyColumns: ColumnDef<CompanyData>[] = [
  {
    header: 'Company Name',
    accessorKey: 'company_name',
  },
  {
    header: 'Stock Price',
    accessorKey: 'stock_price',
  },
  {
    header: '24%',
    accessorKey: 'pc_change',
  },
  {
    header: '24h High Price',
    accessorKey: 'highest',
  },
  {
    header: '24h Low Price',
    accessorKey: 'lowest',
  },
];

export const portfolioColumns: ColumnDef<PortfolioData>[] = [
  {
    header: 'Company Name',
    accessorKey: 'company_name',
  },
  {
    header: 'Number of Shares',
    accessorKey: 'no_of_shares',
  },
  {
    header: 'Total Investment',
    accessorKey: 'total_investment',
  },
  {
    header: 'Current Value',
    accessorKey: 'current_val',
  },
  {
    header: 'Gain / Loss',
    accessorKey: 'gain_loss',
  },
];