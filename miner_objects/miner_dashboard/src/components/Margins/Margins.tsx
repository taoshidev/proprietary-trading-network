import {useState} from "react";
import {
  Box,
  Card,
  Title,
  Group,
  Checkbox,
  Tooltip,
  Text, ThemeIcon,
} from "@mantine/core";
import {IconHelp} from "@tabler/icons-react";
import {
  Area,
  AreaChart,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

import {formatDate} from "../../utils";

interface MarginsProps {
  miner: any;
}

const SCALES = {
  gl: {
    min: -8,
    max: 8,
  },
  delta: {
    min: -0.1,
    max: 0.1,
  },
};

export const Margins = ({miner}: MarginsProps) => {
  const {checkpoints} = miner;
  const [autoScale, setAutoScale] = useState({
    gl: false,
    delta: false,
  });

  const data = checkpoints.map((item: any) => ({
    margin: item.gain + item.loss,
    ...item,
  }));

  const handleAutoScale = (chartType: "gl" | "delta") => {
    setAutoScale((prevAutoScale) => ({
      ...prevAutoScale,
      [chartType]: !prevAutoScale[chartType],
    }));
  };

  const gradientOffset = () => {
    const dataMax = Math.max(...data.map((i: any) => i.margin));
    const dataMin = Math.min(...data.map((i: any) => i.margin));

    if (dataMax <= 0) {
      return 0;
    }
    if (dataMin >= 0) {
      return 1;
    }

    return dataMax / (dataMax - dataMin);
  };

  const off = gradientOffset();

  return (
    <Card mb="lg" withBorder>
      <Box w="100%" pos="relative" mb="xl">
        <Group justify="space-between" mb="lg">
          <Group align="center" justify="center">
            <Title order={3}>Captured Margins</Title>
            <Tooltip
              withArrow
              multiline
              position="right"
              label="Gains minus Losses - positive margin indicates that the miner was
            able to achieve more gains in the period than losses"
              styles={{
                tooltip: {
                  maxWidth: '20vw',
                  padding: '8px',
                  whiteSpace: 'normal', // Ensures text wraps
                },
              }}
            >
              <ThemeIcon variant="transparent" color="gray">
                <IconHelp size={16} stroke={1.4} />
              </ThemeIcon>
            </Tooltip>
          </Group>
          <Checkbox
            size="xs"
            label="Autoscale"
            checked={autoScale.delta}
            onChange={() => handleAutoScale("delta")}
          />
        </Group>
        <ResponsiveContainer height={300}>
          <AreaChart
            data={data}
            margin={{top: 20, right: 20, left: 10, bottom: 0}}
          >
            <RechartsTooltip
              labelFormatter={(unixTime) => formatDate(unixTime, "MM/DD/YY")}
            />

            <Legend
              wrapperStyle={{position: "relative"}}
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

            <XAxis
              dataKey="last_update_ms"
              scale="time"
              tickFormatter={(unixTime) => formatDate(unixTime, "MM/DD/YY")}
              minTickGap={20}
              tickMargin={20}
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
            />

            <YAxis
              tickMargin={20}
              type="number"
              style={{
                fontSize: "0.7rem",
                fontFamily: "inherit",
              }}
              label={{
                value: "returns",
                angle: -90,
                dx: -30,
                fontSize: "0.7rem",
              }}
              domain={[
                autoScale.delta ? "auto" : SCALES.delta.min,
                autoScale.delta ? "auto" : SCALES.delta.max,
              ]}
            />
            <CartesianGrid stroke="#f5f5f5"/>
            <Area
              type="monotone"
              dataKey="margin"
              fill="url(#splitColor)"
              stroke="#E35F25"
              strokeWidth={1}
            />

            <defs>
              <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                <stop offset={off} stopColor="#FAE0D6" stopOpacity={1}/>
                <stop offset={off} stopColor="#E35F25" stopOpacity={1}/>
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </Box>
    </Card>
  );
}
