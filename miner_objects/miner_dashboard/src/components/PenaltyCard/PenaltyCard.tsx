import { Card, Group, Text, ThemeIcon, Tooltip } from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";
import chroma from "chroma-js";

import { orange } from "../../theme/colors";
import { isColorDark } from "../../utils";

type PenaltyCardProps = {
  title: string;
  item: number;
  isPercentage?: boolean;
  sigFigs?: number;
  tooltipText: string;
};

export const PenaltyCard = ({
  title,
  item,
  isPercentage = false,
  sigFigs = 4,
  tooltipText,
}: PenaltyCardProps) => {
  const value = item;
  const multiplier = 10000000;
  
  const getColors = () => {
    const baseColor = orange[6];
    const background = chroma(baseColor).brighten(value * multiplier);
    
    return {
      background: background.hex(),
      foreground: isColorDark(background.hex()) ? "#ffffff" : "#000000",
    };
  };
  
  return (
    <Card withBorder flex="1" h="100%" bg={getColors().background}>
      <Group justify="space-between" align="center">
        <Group align="center" gap="xs">
          <Text size="sm" fw="bold" c={getColors().foreground}>{title}</Text>
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
        
        <Text size="sm" c={getColors().foreground}>
          {value !== undefined && value !== null && !isNaN(value)
            ? isPercentage
              ? `${(value * 100).toFixed(sigFigs)}%`
              : value.toFixed(sigFigs)
            : "N/A"}
        </Text>
      </Group>
    </Card>
  );
};