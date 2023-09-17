import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
} from '@tanstack/react-table';
import { useState } from 'react';
import { AiFillCaretDown, AiFillCaretUp } from 'react-icons/ai';
import { useMediaQuery } from 'react-responsive';
import { useNavigate } from 'react-router-dom';

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
}

function CompanyTable<TData, TValue>({
  columns,
  data,
}: DataTableProps<TData, TValue>) {
  const isLaptopAndDesktop = useMediaQuery({ query: '(min-width: 768px)' });
  const [sorting, setSorting] = useState<SortingState>([]);
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    state: {
      columnVisibility: {
        highest: isLaptopAndDesktop,
        lowest: isLaptopAndDesktop,
      },
      sorting,
    },
    onSortingChange: setSorting,
  });
  const navigate = useNavigate();

  return (
    <>
      <table className="text-white font-display w-full border-b-2 border-border">
        <thead className="border-y-2 border-border">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className="py-2 text-left text-xs text-neutraldark pr-5 lg:pl-16 md:pl-10 lg:text-base hover:cursor-pointer"
                >
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext()
                  )}
                  {{
                    asc: (
                      <span className="inline-block ml-1 relative top-[3px] z-50 text-primarydark">
                        <AiFillCaretUp size={12} />
                      </span>
                    ),
                    desc: (
                      <span className="inline-block ml-1 relative top-[3px] text-primarydark">
                        <AiFillCaretDown size={12} />
                      </span>
                    ),
                  }[header.column.getIsSorted() as string] ?? null}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="text-sm lg:text-lg">
          {table.getRowModel().rows.map((row) => (
            <tr
              key={row.id}
              className="hover:bg-black hover:bg-opacity-30 hover:cursor-pointer overflow-hidden"
              onClick={() => navigate('/stock/aapl')}
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="py-3 lg:pl-16 md:pl-10">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex justify-end mt-2 text-sm">
        <button
          className="text-neutrak disabled:text-textdisabled mx-4"
          disabled={!table.getCanPreviousPage()}
          type="button"
          onClick={() => table.previousPage()}
        >
          {'< Previous'}
        </button>
        <p className="text-primarydark">
          {(table.options.state.pagination?.pageIndex ?? 0) + 1}
        </p>
        <button
          disabled={!table.getCanNextPage()}
          className="text-primary disabled:text-textdisabled mx-4"
          type="button"
          onClick={() => table.nextPage()}
        >
          {'Next >'}
        </button>
      </div>
    </>
  );
}

export default CompanyTable;
