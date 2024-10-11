import { Card, Title } from "@mantine/core";
import { isNil } from "lodash";
import { Fragment } from "react";
import { Statistics as StatisticsType } from "../../types";

import { ProgressBar, Badge } from "../Tremor";

interface ChallengesProps {
  statistics: StatisticsType;
}

export const Challenges = ({ statistics }: ChallengesProps) => {
  const { challengeperiod } = statistics.data[0];
  const { positions, "return": challengeReturn, return_ratio, unrealized_ratio } = challengeperiod;
  
  // if anything is in challenge period show element
  const isInChallenge = !isNil(positions) || !isNil(challengeReturn) || !isNil(return_ratio) || !isNil(unrealized_ratio);
  
  return (
    <Fragment>
      {isInChallenge && (
        <div className="mb-8">
          <Title order={3} mb="sm">Challenge Period</Title>
          <div className="flex gap-4">
            {!isNil(positions) && (
              <Card withBorder className="flex-1">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <p className="font-medium text-gray-900">Positions</p>
                    <span
                      className="font-medium text-gray-900 dark:text-gray-50">{positions.value}<span
                      className="font-normal text-gray-500">/{positions.target.toFixed(2).toString()}</span></span>
                  </div>
                  <ProgressBar
                    showAnimation
                    className="mb-4"
                    variant={positions.passing ? "success" : "default"}
                    value={positions.value}
                    max={positions.target}
                  />
                  <div className="flex justify-between text-sm">
                    {positions.passing ? (
                      <Badge variant="success">
                        Passing
                      </Badge>
                    ) : (
                      <Badge>
                        Not Passing
                      </Badge>
                    )}
                  </div>
                </div>
              </Card>
            )}
            
            {!isNil(challengeReturn) && (
              <Card withBorder className="flex-1">
                <div>
                  <p className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-900">Return</span>
                    <span
                      className="font-medium text-gray-900 dark:text-gray-50">{challengeReturn.value.toFixed(2)}<span
                      className="font-normal text-gray-500">/{challengeReturn.target.toFixed(2).toString()+'%'}</span></span>
                  </p>
                  <ProgressBar
                    showAnimation
                    className="mb-4"
                    variant={challengeReturn.passing ? "success" : "default"}
                    value={challengeReturn.value}
                    max={challengeReturn.target}
                  />
                  
                  <div>
                    {challengeReturn.passing ? (
                      <Badge variant="success">
                        Passing
                      </Badge>
                    ) : (
                      <Badge>
                        Not Passing
                      </Badge>
                    )}
                  </div>
                </div>
              </Card>
            )}
            
            {!isNil(return_ratio) && (
              <Card withBorder className="flex-1">
                <div>
                  <p className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-900">Return Ratio</span>
                    <span
                      className="font-medium text-gray-900 dark:text-gray-50">{return_ratio.value}<span
                      className="font-normal text-gray-500">/{return_ratio.target.toFixed(2).toString()}</span></span>
                  </p>
                  <ProgressBar
                    showAnimation
                    className="mb-4"
                    variant={return_ratio.passing ? "success" : "default"}
                    value={return_ratio.value}
                    max={return_ratio.target}
                  />
                  
                  <div>
                    {return_ratio.passing ? (
                      <Badge variant="success">
                        Passing
                      </Badge>
                    ) : (
                      <Badge>
                        Not Passing
                      </Badge>
                    )}
                  </div>
                </div>
              </Card>
            )}
            
            {!isNil(unrealized_ratio) && (
              <Card withBorder className="flex-1">
                <div>
                  <p className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-900">Unrealized Ratio</span>
                    <span
                      className="font-medium text-gray-900 dark:text-gray-50">{unrealized_ratio.value.toFixed(2)}<span
                      className="font-normal text-gray-500">/{unrealized_ratio.target.toFixed(2).toString()}</span></span>
                  </p>
                  <ProgressBar
                    showAnimation
                    className="mb-4"
                    variant={unrealized_ratio.passing ? "success" : "default"}
                    value={unrealized_ratio.value}
                    max={unrealized_ratio.target}
                  />
                  
                  <div>
                    {unrealized_ratio.passing ? (
                      <Badge variant="success">
                        Passing
                      </Badge>
                    ) : (
                      <Badge>
                        Not Passing
                      </Badge>
                    )}
                  </div>
                </div>
              </Card>
            )}
          
          </div>
        </div>
      )}
    </Fragment>
  
  );
};
