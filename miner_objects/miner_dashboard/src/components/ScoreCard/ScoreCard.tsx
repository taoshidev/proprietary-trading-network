import { Card, Badge } from "@mantine/core";
import { ProgressBar } from "../Tremor";

interface ScoreCardProps {
  label: string;
  value: number;
  target: number;
  isPercentage?: boolean;
}

export const ScoreCard = ({ label, value, target, isPercentage = false }: ScoreCardProps) => {
  return (
    <Card withBorder className="flex-1">
      <div>
        <div className="flex justify-between text-sm mb-2">
          <p className="font-medium text-gray-900">{label}</p>
          <span className="font-medium text-gray-900 dark:text-gray-50">
            {value.toFixed(2)}
            <span className="font-normal text-gray-500">
              /{target.toFixed(2)}{isPercentage ? "%" : ""}
            </span>
          </span>
        </div>
        <ProgressBar
          showAnimation
          className="mb-4"
          variant={value >= target ? "success" : "default"}
          value={value}
          max={target}
        />
        <div className="flex justify-between text-sm">
          {value >= target ? (
            <Badge variant="success">Passing</Badge>
          ) : (
            <Badge>Not Passing</Badge>
          )}
        </div>
      </div>
    </Card>
  );
};
