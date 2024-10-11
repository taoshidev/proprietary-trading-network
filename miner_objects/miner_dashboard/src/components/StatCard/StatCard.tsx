import { Card, Group, Text, Tooltip, ThemeIcon, Title, Box } from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";

import { Score } from "../../types";

type StatCardProps = {
  title: string;
  item: Score | number; // Accept both Score and number
  isPercentage?: boolean;
  sigFigs?: number;
  tooltipText: string;
};

export const StatCard = ({ title, item, isPercentage = false, sigFigs = 4, tooltipText }: StatCardProps) => {
  const value = typeof item === "number" ? item : item.value; // Get the number value
  const rank = typeof item === "number" ? null : item.rank;
  const percentile = typeof item === "number" ? null : item.percentile;
  
  return (
    <Card withBorder>
      <Group align="center" gap="xs" justify="space-between" mb="lg">
        <Title order={3}>
          {value !== undefined && value !== null && !isNaN(value)
            ? isPercentage
              ? `${(value * 100).toFixed(sigFigs)}%`
              : value.toFixed(sigFigs)
            : "N/A"}
        </Title>
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
      
      <Box mb="sm">
        <Text size="sm" fw="bold">{title}</Text>
      </Box>
      
      {rank !== null && (
        <Group justify="space-between" align="center" mb="xs">
          <Text size="xs" c="gray">Rank</Text>
          <Text size="xs" style={{ textAlign: "right" }}>{rank}</Text>
        </Group>
      )}
      
      {percentile !== null && (
        <Group justify="space-between" align="center" mb="xs">
          <Text size="xs" c="gray">Percentile</Text>
          <Text size="xs" style={{ textAlign: "right" }}>
            {(percentile * 100).toFixed(0)}%
          </Text>
        </Group>
      )}
    </Card>
  );
};