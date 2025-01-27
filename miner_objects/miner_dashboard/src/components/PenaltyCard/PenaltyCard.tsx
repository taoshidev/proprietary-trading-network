import { Card, Group, Text, ThemeIcon, Tooltip } from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";

import { toShortFloat } from "../../utils";

type PenaltyCardProps = {
  item: number;
  isPercentage?: boolean;
  sigFigs?: number;
  tooltipText: string;
};

export const PenaltyCard = ({ item, tooltipText }: PenaltyCardProps) => {
  const { drawdown_threshold, total, martingale } = item;

  return (
    <Card withBorder flex="1" h="100%">
      <Group justify="space-between" align="center">
        <Group align="center" gap="xs">
          <Text fw="bold">Penalties</Text>
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

        {drawdown_threshold !== null && (
          <Group justify="space-between" align="center">
            <Text size="xs" c="gray">
              Drawdown Threshold
            </Text>
            <Text size="xs" fw="bold" style={{ textAlign: "right" }}>
              {toShortFloat(drawdown_threshold)}
            </Text>
          </Group>
        )}

        {martingale !== null && (
          <Group justify="space-between" align="center">
            <Text size="xs" c="gray">
              Martingale
            </Text>
            <Text size="xs" fw="bold" style={{ textAlign: "right" }}>
              {toShortFloat(martingale)}
            </Text>
          </Group>
        )}

        {total !== null && (
          <Group justify="space-between" align="center">
            <Text size="xs" c="gray">
              Total
            </Text>
            <Text size="xs" fw="bold" style={{ textAlign: "right" }}>
              {total}
            </Text>
          </Group>
        )}
      </Group>
    </Card>
  );
};
