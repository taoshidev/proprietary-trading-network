import { Fragment } from "react";
import { Table, Text, Box, Button } from "@mantine/core";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getExpandedRowModel,
} from "@tanstack/react-table";

import { Order, Source } from "../../types";
import { formatCurrency, formatDate } from "../../utils";

import { Sources } from "../Sources";

import styles from "./orders.module.css";

interface ColumnData {
  leverage: number;
  order_type: string;
  processed_ms: number;
  price: number;
  price_sources: Source[];
}

const columnHelper = createColumnHelper<ColumnData>();

const columns = [
  columnHelper.accessor("leverage", {
    header: "Leverage",
    cell: (info) => (
      <Text size="xs" ta="right">
        {Math.abs(info.getValue())}
      </Text>
    ),
  }),
  columnHelper.accessor("order_type", {
    header: "Order Type",
    cell: (info) => (
      <Text size="xs" ta="right">
        {info.getValue()}
      </Text>
    ),
  }),
  columnHelper.accessor("processed_ms", {
    header: "Processed Time",
    cell: (info) => (
      <Text size="xs" ta="right">
        {formatDate(info.getValue())}
      </Text>
    ),
  }),
  columnHelper.accessor("price", {
    header: "Price",
    cell: (info) => (
      <Text size="xs" ta="right">
        {formatCurrency(info.getValue())}
      </Text>
    ),
  }),
  columnHelper.display({
    id: "expander",
    cell: ({ row }) => {
      if (row.original.price_sources.length === 0) {
        return null;
      }
      
      return (
        <Box ta="right">
          <Button
            variant="subtle"
            size="xs"
            onClick={() => row.toggleExpanded()}
          >
            View Sources
          </Button>
        </Box>
      );
    },
  }),
];

interface OrdersProps {
  orders: Order[];
}

export const Orders = ({ orders }: OrdersProps) => {
  const table = useReactTable({
    initialState: {
      pagination: {
        pageIndex: 0,
      },
    },
    data: orders,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
  });
  
  return (
    <Table.Tr className={styles.tr}>
      <Table.Td colSpan={9} className={styles.td}>
        <Table className={styles.table} verticalSpacing="xs">
          <Table.Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Table.Tr key={headerGroup.id} className={styles.tr}>
                {headerGroup.headers.map((header) => (
                  <Table.Th key={header.id}>
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
              <Fragment key={row.id}>
                <Table.Tr>
                  {row.getVisibleCells().map((cell) => (
                    <Table.Td key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </Table.Td>
                  ))}
                </Table.Tr>
                {row.getIsExpanded() && !!row.original.price_sources.length && (
                  <Sources sources={row.original.price_sources} />
                )}
              </Fragment>
            ))}
          </Table.Tbody>
        </Table>
      </Table.Td>
    </Table.Tr>
  );
};
