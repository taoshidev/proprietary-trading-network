import { Table, Text } from "@mantine/core";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getSortedRowModel,
} from "@tanstack/react-table";
import { Source } from "../../types";

import { formatDate } from "../../utils";

import styles from "./sources.module.css";

interface ColumnData {
  source: string;
  start_ms: number;
  high: number;
  low: number;
  timespan_ms: number;
  lag_ms: number;
  open: number;
  close: number;
}

const columnHelper = createColumnHelper<ColumnData>();

const columns = [
  columnHelper.accessor("source", {
    header: "Source",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("start_ms", {
    header: "Start",
    cell: (info) => (
      <Text size="xs" ta="right">
        {formatDate(info.getValue())}
      </Text>
    ),
  }),
  columnHelper.accessor("high", {
    header: "High",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("low", {
    header: "Low",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("timespan_ms", {
    header: "Timespan (ms)",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("lag_ms", {
    header: "Lag (ms)",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("open", {
    header: "Open",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("close", {
    header: "Close",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
];


interface SourcesProps {
  sources: Source[];
}


export const Sources = ({ sources = [] }: SourcesProps) => {
  const table = useReactTable({
    data: sources,
    columns,
    getSortedRowModel: getSortedRowModel(),
    getCoreRowModel: getCoreRowModel(),
  });
  
  return (
    <Table.Tr className={styles.tr}>
      <Table.Td colSpan={5} className={styles.td}>
        <Table className={styles.table} verticalSpacing="md">
          <Table.Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Table.Tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Table.Th key={header.id} className={styles.th}>
                    <Text size="xs" fw={700} ta="right">
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                    </Text>
                  </Table.Th>
                ))}
              </Table.Tr>
            ))}
          </Table.Thead>
          <Table.Tbody>
            {table.getRowModel().rows.map((row) => (
              <Table.Tr key={row.id} className={styles.tr}>
                {row.getVisibleCells().map((cell) => (
                  <Table.Td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </Table.Td>
                ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.Td>
    </Table.Tr>
  );
};
