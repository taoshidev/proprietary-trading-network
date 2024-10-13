import { useState } from "react";
import { Box, Card, Title, Group, Text, Slider } from "@mantine/core";
import { Statistics, Checkpoint } from "../../types";

import { formatDate, toPercent } from "../../utils";

import {
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface OverviewGraphProps {
  statistics: Statistics;
}

export function OverviewGraph({ statistics }: OverviewGraphProps) {
  const [value, setValue] = useState(0.5);
  const [max, setMax] = useState(0.5);
  const { checkpoints } = statistics.data[0];
  
  const data = checkpoints.map((item: Checkpoint) => {
    return {
      ...item,
      mdd: 1 - item.mdd, // drawdown is negative, so we need to invert it
    };
  });
  
  return (
    <Box mb="lg">
      <Card withBorder>
        <Group justify="space-between" mb="lg" align="center">
          <Title order={3}>Overall Returns vs Drawdown</Title>
          <Box w={200}>
            <Text size="xs" mb="xs">
              Drawdown Scale
            </Text>
            <Slider
              radius={0}
              size="xs"
              min={0.01}
              max={0.5}
              step={0.001}
              defaultValue={0.5}
              value={value}
              onChange={setValue}
              onChangeEnd={setMax}
            />
          </Box>
        </Group>
        <ResponsiveContainer width="100%" height={500}>
          <ComposedChart
            data={data}
            margin={{
              top: 20,
              right: 20,
              bottom: 0,
              left: 10,
            }}
          >
            <XAxis
              dataKey="last_update_ms"
              scale="time"
              tickFormatter={(unixTime) => formatDate(unixTime, "MM/DD/YY")}
              minTickGap={40}
              tickMargin={10}
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
            />
            
            <YAxis
              allowDataOverflow
              tickMargin={20}
              label={{
                value: "returns",
                angle: -90,
                dx: -30,
                fontSize: "0.7rem",
              }}
              type="number"
              yAxisId="area"
              domain={["auto", "auto"]}
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
            />
            
            <YAxis
              allowDataOverflow
              orientation="right"
              label={{
                value: "drawdown",
                angle: 90,
                offset: -10,
                dx: 40,
                fontSize: "0.7rem",
              }}
              type="number"
              yAxisId="bar"
              domain={[0, max]}
              reversed
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
              tickMargin={20}
              tickFormatter={(value) => toPercent(value, 0)}
            />
            
            <Tooltip
              labelFormatter={(unixTime) => formatDate(unixTime, "MM/DD/YY")}
            />
            
            <Legend
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
              formatter={(value) => (
                <Text size="xs" component="span">
                  {value}
                </Text>
              )}
            />
            <CartesianGrid stroke="#f5f5f5" />
            <Bar yAxisId="bar" dataKey="mdd" name="drawdown" fill="#D4D4D4" />
            <Area
              type="monotone"
              dataKey="overall_returns"
              yAxisId="area"
              fill="#FAE0D6"
              stroke="#EB8861"
              strokeWidth={1}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>
    </Box>
  );
}
