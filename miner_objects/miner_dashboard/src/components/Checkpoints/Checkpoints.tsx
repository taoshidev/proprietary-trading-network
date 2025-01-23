import {
  Box,
  Group,
  ThemeIcon,
  Title,
  Text,
  Tooltip,
  SimpleGrid,
} from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";

import { Statistics } from "../../types";

import { StatCard } from "../StatCard";
import { PenaltyCard } from "../PenaltyCard";

interface CheckpointsProps {
  statistics: Statistics;
}

export const Checkpoints = ({ statistics }: CheckpointsProps) => {
  const { penalties, scores } = statistics.data[0];

  return (
    <Box>
      <Box mb="xl">
        <Title order={3} mb="sm">
          Scoring Metricss
        </Title>
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Calmar Ratio (%)"
            item={scores.calmar}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Risk-Adjusted Return measures the returns from all closed positions or open positions in loss over 90 days and normalizes by the drawdown used to capture these returns."
          />
          <StatCard
            title="Omega Ratio"
            item={scores.omega}
            sigFigs={2}
            tooltipText="The Omega Ratio is a ratio between the gains over the losses. The higher the ratio, the more gains dominated the historical behavior."
          />
          <StatCard
            title="Sharpe Ratio"
            item={scores.sharpe}
            sigFigs={2}
            tooltipText="The Sharpe Ratio assesses the return of an investment compared to the std. dev. of returns. A higher Sharpe ratio indicates higher returns at greater predictability."
          />
          <StatCard
            title="Short-Term Calmar Ratio (%)"
            item={scores["short-calmar"]}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Similar to the Risk-Adjusted Returns, Short-Term Risk-Adjusted Return differs in that it only considers positions closed within the past 5 days or open positions in loss."
          />
        </SimpleGrid>
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Statistical Confidence"
            item={scores.statistical_confidence}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Risk-Adjusted Return measures the returns from all closed positions or open positions in loss over 90 days and normalizes by the drawdown used to capture these returns."
          />
          <StatCard
            title="Sortino Ratio"
            item={scores.sortino}
            sigFigs={2}
            tooltipText="The Omega Ratio is a ratio between the gains over the losses. The higher the ratio, the more gains dominated the historical behavior."
          />
          <StatCard
            disabled
            title="Return (%)"
            item={scores.return}
            sigFigs={2}
            tooltipText="The Sharpe Ratio assesses the return of an investment compared to the std. dev. of returns. A higher Sharpe ratio indicates higher returns at greater predictability."
          />
          <StatCard
            disabled
            title="Short-Term Return (%)"
            item={scores.short_return}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Similar to the Risk-Adjusted Returns, Short-Term Risk-Adjusted Return differs in that it only considers positions closed within the past 5 days or open positions in loss."
          />
        </SimpleGrid>
      </Box>

      <Box mb="lg">
        <Group justify="flex-start" gap="xs" align="center" mb="sm">
          <Text size="sm">Penalty Multipliers</Text>
          <Tooltip
            label="The scores of the miners are multiplied by all of the penalties to produce the final penalized score. If any penalty is 0, the miner's score is set to 0."
            withArrow
            multiline
            styles={{
              tooltip: {
                maxWidth: "20vw",
                padding: "8px",
                whiteSpace: "normal",
              },
            }}
          >
            <ThemeIcon variant="transparent" color="gray">
              <IconHelp size={16} stroke={1.4} />
            </ThemeIcon>
          </Tooltip>
        </Group>

        <SimpleGrid cols={5}>
          <PenaltyCard
            item={penalties}
            isPercentage={false}
            sigFigs={5}
            tooltipText="A single two-week period should not account for more than 35% of total unrealized return. If it does, the multiplier will tend towards 0."
          />
        </SimpleGrid>
      </Box>
    </Box>
  );
};
