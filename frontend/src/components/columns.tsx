import { ColumnDef } from '@tanstack/react-table';

export type CompanyData = {
  company_name: string;
  stock_price: string;
  pc_change: number;
  highest: string;
  lowest: string;
};

export const columns: ColumnDef<CompanyData>[] = [
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
