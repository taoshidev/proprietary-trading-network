import { Title } from "@mantine/core";
import { isNil } from "lodash";
import { Fragment } from "react";
import { Statistics as StatisticsType } from "../../types";
import { ScoreCard } from "../ScoreCard";

interface ChallengesProps {
  statistics: StatisticsType;
}

export const Challenges = ({ statistics }: ChallengesProps) => {
  const { challengeperiod } = statistics.data[0];
  const { status, scores } = challengeperiod;
  const { omega, overall, return_long, return_short, sharpe_ratio, sortino, statistical_confidence } = scores

  const scoreData = [
    { label: "Overall", score: overall },
    { label: "Omega", score: omega },
    { label: "Return Long", score: return_long },
    { label: "Return Short", score: return_short },
    { label: "Sharpe Ratio", score: sharpe_ratio },
    { label: "Sortino", score: sortino },
    { label: "Statistical Confidence", score: statistical_confidence },
  ];
  
  // if anything is in challenge period show element
  const isInChallenge = !isNil(scores)  && status === "testing";
  const passingThreshold = overall.target_percentile

  return (
    <Fragment>
      {isInChallenge && (
        <div className="mb-8">
          <Title order={3} mb="sm">
            Challenge Period
          </Title>
          <div className="flex gap-4">
            {scoreData.map(({ label, score }) => {
              if (!isNil(score)) {
                return (
                  <ScoreCard
                    key={label}
                    label={label}
                    value={score.percentile}
                    target={passingThreshold}
                  />
                );
              }
              return null;
            })}
          </div>
        </div>
      )}
    </Fragment>
  );
};
