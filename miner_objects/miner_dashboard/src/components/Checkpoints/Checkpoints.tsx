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
          Scoring Metrics
        </Title>
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Calmar Ratio (%)"
            item={scores.calmar}
            isPercentage={true}
            sigFigs={2}
            tooltipText="The Calmar ratio looks at a ratio between the annualized 90 day return and the max drawdown."
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
            title="Statistical Confidence"
            item={scores.statistical_confidence}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Statistical Confidence (t-statistic) is the measure of statistical likelihood of the daily return distribution to come from a random distribution. The higher the value, the lower the probability that the returns are likely random."
          />
        </SimpleGrid>
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Sortino Ratio"
            item={scores.sortino}
            sigFigs={2}
            tooltipText="The Sortino Ratio measures the risk-adjusted return, similar to Sharpe, but only penalizes downside volatility rather than all volatility."
          />
          <StatCard
            disabled
            title="Return (%)"
            item={scores.return}
            sigFigs={2}
            tooltipText="Estimated Annualized Return, based on daily returns from the past 90 days."
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
              title="Penalties"
              penalties={penalties}
              tooltipText="Risk normalization multiplier, due to drawdown. Note that this term may be larger than one, indicating that the miner receives a boost in score due to minimal drawdown utilization."
            />
        </SimpleGrid>
      </Box>
    </Box>
  );
};
