"use client";

import { Card, Box, Title, Progress } from "@mantine/core";

import { type Statistics } from "@/types";

interface StatisticsProps {
  statistics: Statistics;
}

export function Contribution({ statistics }: StatisticsProps) {
  const { scores } = statistics.data[0];
  const {
    omega,
    calmar,
    sharpe,
    sortino,
    statistical_confidence,
    "short-calmar": shortCalmar,
  } = scores;

  return (
    <Card withBorder h="100%" mb="xl">
      <Box mb="lg">
        <Title order={3}>Miner Contribution</Title>
      </Box>
      <Box>
        <Progress.Root size="xl" mb="md">
          <Progress.Section
            value={omega.overall_contribution * 100}
            color="#E35E24"
          >
            <Progress.Label>Omega</Progress.Label>
          </Progress.Section>
          <Progress.Section
            value={calmar.overall_contribution * 100}
            color="#A44C26"
          >
            <Progress.Label>Calmar</Progress.Label>
          </Progress.Section>
          <Progress.Section
            value={sharpe.overall_contribution * 100}
            color="#B05229"
          >
            <Progress.Label>Sharpe</Progress.Label>
          </Progress.Section>
          <Progress.Section
            value={shortCalmar.overall_contribution * 100}
            color="#C95E2F"
          >
            <Progress.Label>Short Ad. Return</Progress.Label>
          </Progress.Section>
          <Progress.Section
            value={sortino.overall_contribution * 100}
            color="#CF6638"
          >
            <Progress.Label>Sortino</Progress.Label>
          </Progress.Section>
          <Progress.Section
            value={statistical_confidence.overall_contribution * 100}
            color="#D26F44"
          >
            <Progress.Label>Statistical Confidence</Progress.Label>
          </Progress.Section>
        </Progress.Root>
      </Box>
    </Card>
  );
}
