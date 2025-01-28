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
    { label: "Overall", value: overall.percentile, target: overall.target_percentile},
    { label: "Omega", value: omega.value, target: omega.target_score},
    { label: "Return Long", value: return_long.value, target: return_long.target_score, isPercentage: true },
    { label: "Return Short", value: return_short.value, target: return_short.target_score, isPercentage: true },
    { label: "Sharpe Ratio", value: sharpe_ratio.value, target: sharpe_ratio.target_score },
    { label: "Sortino", value: sortino.value, target: sortino.target_score },
    { label: "Statistical Confidence", value: statistical_confidence.value, target: statistical_confidence.target_score },
  ];
  
  // if anything is in challenge period show element
  const isInChallenge = !isNil(scores)  && status === "testing";

  return (
    <Fragment>
      {isInChallenge && (
        <div className="mb-8">
          <Title order={3} mb="sm">
            Challenge Period
          </Title>
          <div className="flex gap-4">
            {scoreData.map(({ label, value, target, isPercentage }) => {
              if (!isNil(value)) {
                return (
                  <ScoreCard
                    key={label}
                    label={label}
                    value={value}
                    target={target}
                    isPercentage={isPercentage}
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
