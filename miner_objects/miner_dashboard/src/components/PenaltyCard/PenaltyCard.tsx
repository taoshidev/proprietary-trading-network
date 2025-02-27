"use client";

import { Card, Group, Text, Tooltip, ThemeIcon, Box } from "@mantine/core";
import { IconQuestionMark } from "@tabler/icons-react";

import { toShortFloat } from "../../utils";

interface PenaltiesGroup {
  drawdown_threshold: number;
  martingale: number;
  total: number;
}

interface PenaltyCardProps {
  title: string;
  penalties: PenaltiesGroup;
  tooltipText: string;
}

export const PenaltyCard = ({
  title,
  penalties,
  tooltipText,
}: PenaltyCardProps) => {
  return (
    <Card withBorder h="100%">
      <Group justify="space-between" align="center" mb="sm">
        <Group gap="xs">
          <Text fw="bold">{title}</Text>
          <Tooltip label={tooltipText} withArrow multiline w={300}>
            <ThemeIcon variant="light" radius="xl" size="xs">
              <IconQuestionMark style={{ width: "70%", height: "70%" }} />
            </ThemeIcon>
          </Tooltip>
        </Group>
      </Group>

      <Box mb="xs">
        <Text size="xs">
          Drawdown Threshold:{" "}
          <Text size="xs" fw="bold" component="span">
            {toShortFloat(penalties.drawdown_threshold)}
          </Text>
        </Text>
      </Box>

      <Box mb="xs">
        <Text size="xs">
          Total:{" "}
          <Text size="xs" fw="bold" component="span">
            {toShortFloat(penalties.total)}
          </Text>
        </Text>
      </Box>

      <Box>
        <Text size="xs">
          Martingale:{" "}
          <Text size="xs" fw="bold" component="span">
            {toShortFloat(penalties.martingale)}
          </Text>
        </Text>
      </Box>
    </Card>
  );
};
