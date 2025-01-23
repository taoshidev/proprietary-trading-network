import { Card, Group, Text, Tooltip, ThemeIcon, Box } from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";

import { Score } from "../../types";

type StatCardProps = {
  disabled?: boolean;
  title: string;
  item: Score | number; // Accept both Score and number
  isPercentage?: boolean;
  sigFigs?: number;
  tooltipText: string;
};

export const StatCard = ({
  disabled,
  title,
  item,
  isPercentage = false,
  sigFigs = 4,
  tooltipText,
}: StatCardProps) => {
  const { value, rank, percentile } = item as Score;

  return (
    <Card
      withBorder
      style={
        disabled
          ? { filter: "saturate(0)", opacity: 0.7, borderStyle: "dashed" }
          : {}
      }
    >
      <Group align="center" gap="xs" justify="space-between" mb="lg">
        <Group align="center" gap="0" justify="space-between">
          <Text fw="bold">{title}</Text>
          <Tooltip
            label={tooltipText}
            withArrow
            multiline
            styles={{
              tooltip: {
                maxWidth: "20vw",
                padding: "8px",
                whiteSpace: "normal", // Ensures text wraps
              },
            }}
          >
            <ThemeIcon variant="transparent" color="gray">
              <IconHelp size={16} stroke={1.4} />
            </ThemeIcon>
          </Tooltip>
        </Group>
        <Box>
          <Text fw="bold" c="orange">
            {value !== undefined && value !== null && !isNaN(value)
              ? isPercentage
                ? `${(value * 100).toFixed(sigFigs)}%`
                : value.toFixed(sigFigs)
              : "N/A"}
          </Text>
        </Box>
      </Group>

      {rank !== null && (
        <Group justify="space-between" align="center" mb="xs">
          <Text size="xs" c="gray">
            Rank
          </Text>
          <Text size="xs" style={{ textAlign: "right" }}>
            {rank}
          </Text>
        </Group>
      )}

      {percentile !== null && (
        <Group justify="space-between" align="center" mb="xs">
          <Text size="xs" c="gray">
            Percentile
          </Text>
          <Text size="xs" style={{ textAlign: "right" }}>
            {(percentile * 100).toFixed(0)}%
          </Text>
        </Group>
      )}
    </Card>
  );
};
