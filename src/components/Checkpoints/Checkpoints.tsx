import { Box, Group, ThemeIcon, Title, Tooltip, SimpleGrid } from "@mantine/core";
import { IconHelp } from "@tabler/icons-react";

import { Statistics } from "../../types";

import { StatCard } from "../StatCard";
import { PenaltyCard } from "../PenaltyCard";

interface CheckpointsProps {
  statistics: Statistics;
}

export const Checkpoints = ({ statistics }: CheckpointsProps) => {
  const { penalties, scores, penalized_scores } = statistics.data[0];
  
  return (
    <Box>
      <Box mb="xl">
        <Title order={3} mb="sm">Scoring Metrics</Title>
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Risk-Adjusted Return"
            item={scores.risk_adjusted_return}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Risk-Adjusted Return measures the returns from all closed positions or open positions in loss over 90 days and normalizes by the drawdown used to capture these returns."
          />
          
          <StatCard
            title="Short-Term Risk-Adjusted Return"
            item={scores.short_risk_adjusted_return}
            isPercentage={true}
            sigFigs={2}
            tooltipText="Similar to the Risk-Adjusted Returns, Short-Term Risk-Adjusted Return differs in that it only considers positions closed within the past 5 days or open positions in loss."
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
        </SimpleGrid>
      </Box>
      
      
      <Box mb="lg">
        <Group justify="flex-start" gap="xs" align="center" mb="sm">
          <Title order={3}>Penalty Multipliers</Title>
          <Tooltip
            label="The scores of the miners are multiplied by all of the penalties to produce the final penalized score. If any penalty is 0, the miner's score is set to 0."
            withArrow
            multiline
            styles={{ tooltip: { maxWidth: "20vw", padding: "8px", whiteSpace: "normal" } }}
          >
            <ThemeIcon variant="transparent" color="gray">
              <IconHelp size={16} stroke={1.4} />
            </ThemeIcon>
          </Tooltip>
        </Group>
        
        <SimpleGrid cols={5}>
          <PenaltyCard
            title="Biweekly"
            item={penalties.biweekly}
            isPercentage={false}
            sigFigs={5}
            tooltipText="A single two-week period should not account for more than 35% of total unrealized return. If it does, the multiplier will tend towards 0."
          />
          
          <PenaltyCard
            title="Daily"
            item={penalties.daily}
            isPercentage={false}
            sigFigs={5}
            tooltipText="A single day should not account for more than 20% of total portfolio value increase. If it does, the multiplier will tend towards 0."
          />
          
          <PenaltyCard
            title="Drawdown"
            item={penalties.drawdown}
            isPercentage={false}
            sigFigs={5}
            tooltipText="Risk normalization multiplier, due to drawdown. Note that this term may be larger than one, indicating that the miner receives a boost in score due to minimal drawdown utilization."
          />
          
          <PenaltyCard
            title="Returns Ratio"
            item={penalties.returns_ratio}
            isPercentage={false}
            sigFigs={5}
            tooltipText="A single position should not represent more than 15% of total realized return. If it does, the multiplier will tend towards 0."
          />
          
          <PenaltyCard
            title="Time Consistency"
            item={penalties.time_consistency}
            isPercentage={false}
            sigFigs={5}
            tooltipText="A maximum of 30% of a miner's returns should come from positions closed within a single week. Beyond this, the multiplier will tend towards 0."
          />
        </SimpleGrid>
      </Box>
      
      <Box mb="xl">
        
        <SimpleGrid mb="lg" cols={4}>
          <StatCard
            title="Penalized Risk-Adjusted Return"
            item={penalized_scores.risk_adjusted_return}
            isPercentage={true}
            sigFigs={3}
            tooltipText="This is the final risk-adjusted return after applying all penalties."
          />
          
          <StatCard
            title="Penalized Short-Term Risk-Adjusted Return"
            item={penalized_scores.short_risk_adjusted_return}
            isPercentage={true}
            sigFigs={3}
            tooltipText="This is the final short-term risk-adjusted return after applying all penalties."
          />
          
          <StatCard
            title="Penalized Omega Ratio"
            item={penalized_scores.omega}
            sigFigs={3}
            tooltipText="The final Omega ratio after applying all penalties."
          />
          
          <StatCard
            title="Penalized Sharpe Ratio"
            item={penalized_scores.sharpe}
            sigFigs={3}
            tooltipText="The final Sharpe ratio after applying all penalties."
          />
        </SimpleGrid>
      </Box>
    </Box>
  );
};

