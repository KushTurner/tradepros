import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getSortedRowModel,
} from '@tanstack/react-table';
import { useMediaQuery } from 'react-responsive';

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
}

function CompanyTable<TData, TValue>({
  columns,
  data,
}: DataTableProps<TData, TValue>) {
  const isLaptopAndDesktop = useMediaQuery({ query: '(min-width: 768px)' });
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    state: {
      columnVisibility: {
        no_of_shares: isLaptopAndDesktop,
        current_val: isLaptopAndDesktop,
      },
    },
  });

  return (
    <table className="text-white font-display w-full border-b-2 border-[#333438]">
      <thead className="border-y-2 border-[#333438]">
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th
                key={header.id}
                onClick={header.column.getToggleSortingHandler()}
                className="py-2 text-left text-xs text-[#6E707F] pr-5 lg:pl-16 md:pl-10 lg:text-base"
              >
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext()
                )}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody className="text-sm lg:text-lg">
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id} className="py-3 lg:pl-16 md:pl-10">
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default CompanyTable;
