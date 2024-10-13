import { Container } from "@mantine/core";

import { MinerData } from "../../types";
import { Challenges } from "../Challenges";

import { Checkpoints } from "../Checkpoints";
import { Statistics } from "../Statistics";
import { OverviewGraph } from "../OverviewGraph";
import { Positions } from "../Positions";

interface MainProps {
  data: MinerData;
}

export const Main = ({ data }: MainProps) => {
  const { statistics, positions } = data;
  const { hotkey } = statistics.data[0]
  
  return (
    <Container fluid pt="72">
      <Challenges statistics={statistics} />
      <Checkpoints statistics={statistics} />
      <Statistics statistics={statistics} positions={positions} />
      <Positions positions={positions[hotkey].positions} />
      <OverviewGraph statistics={statistics} />
    </Container>
  );
};