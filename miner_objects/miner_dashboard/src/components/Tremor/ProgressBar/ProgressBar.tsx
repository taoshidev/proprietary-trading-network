// Tremor ProgressBar [v0.0.2]

import React from "react";
import { tv, type VariantProps } from "tailwind-variants";
import Decimal from "decimal.js";

import { cx } from "../../../utils";

import { Tooltip } from "../Tooltip";

const progressBarVariants = tv({
  slots: {
    background: "",
    bar: "",
    tick: "",
  },
  variants: {
    variant: {
      default: {
        background: "bg-gray-200 dark:bg-gray-500/30",
        bar: "bg-gray-500 dark:bg-gray-500",
        tick: "bg-gray-500 dark:bg-gray-500",
      },
      neutral: {
        background: "bg-gray-200 dark:bg-gray-500/40",
        bar: "bg-gray-500 dark:bg-gray-500",
        tick: "bg-gray-500 dark:bg-gray-500",
      },
      warning: {
        background: "bg-yellow-200 dark:bg-yellow-500/30",
        bar: "bg-yellow-500 dark:bg-yellow-500",
        tick: "bg-yellow-500 dark:bg-yellow-500",
      },
      error: {
        background: "bg-red-200 dark:bg-red-500/30",
        bar: "bg-red-500 dark:bg-red-500",
        tick: "bg-red-500 dark:bg-red-500",
      },
      success: {
        background: "bg-orange-200 dark:bg-orange-500/30",
        bar: "bg-orange-500 dark:bg-orange-500",
        tick: "bg-orange-500 dark:bg-orange-500",
      },
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

interface ProgressBarProps
  extends React.HTMLProps<HTMLDivElement>,
    VariantProps<typeof progressBarVariants> {
  value?: number;
  max?: number;
  min?: number;
  showAnimation?: boolean;
  label?: string;
}

const ProgressBar = React.forwardRef<HTMLDivElement, ProgressBarProps>(
  (
    {
      value = 0,
      max = 100,
      label,
      showAnimation = false,
      variant,
      className,
      ...props
    }: ProgressBarProps,
    forwardedRef,
  ) => {
    const { background, bar, tick } = progressBarVariants({ variant });
    let progressWidth: Decimal;
    
    const decimalValue = new Decimal(value); // Convert value to Decimal
    const decimalMax = new Decimal(max); // Convert max (target) to Decimal
    
    if (decimalValue.greaterThan(decimalMax)) {
      // Case 1: When the current value needs to decrease to hit the target (max)
      progressWidth = decimalValue.minus(decimalMax).div(decimalValue).times(100); // Inverse progress calculation
    } else {
      // Case 2: When the current value needs to increase to hit the target (max)
      progressWidth = decimalValue.div(decimalMax).times(100); // Normal progress calculation
    }
    
    // Ensure progressWidth is capped at 100% to avoid overflow
    progressWidth = Decimal.min(progressWidth, 100);
    
    // Format the progress width with 2 decimal places
    const formattedProgressWidth = progressWidth.toFixed(2);
    
    return (
      <div
        ref={forwardedRef}
        className={cx("flex w-full items-center", className)}
        tremor-id="tremor-raw"
        {...props}
      >
        <div
          className={cx(
            "relative flex h-2 w-full items-center rounded-full",
            background(),
          )}
          aria-label="progress bar"
          aria-valuenow={value}
          aria-valuemax={max}
        >
          <div
            className={cx(
              "h-full flex-col rounded-full",
              bar(),
              showAnimation &&
              "transform-gpu transition-all duration-300 ease-in-out",
            )}
            style={{ width: `${formattedProgressWidth}%` }}
          />
          
          
          <div
            className={cx(
              "absolute w-2 -translate-x-1/2",
              "transform-gpu transition-all duration-300 ease-in-out",
            )}
            style={{
              left: `${formattedProgressWidth}%`,
            }}
          >
            <Tooltip triggerAsChild content={value}>
              <div
                aria-hidden="true"
                className={cx(
                  "relative mx-auto h-4 w-1 rounded-full ring-2",
                  "ring-white dark:ring-gray-950",
                  tick(),
                )}
              >
                <div
                  aria-hidden
                  className="absolute size-7 -translate-x-[45%] -translate-y-[15%]"
                ></div>
              </div>
            </Tooltip>
          </div>
        
        
        </div>
        {label ? (
          <span
            className={cx(
              // base
              "ml-2 whitespace-nowrap text-sm font-medium leading-none",
              // text color
              "text-gray-900 dark:text-gray-50",
            )}
          >
            {label}
          </span>
        ) : null}
      </div>
    );
  },
);

ProgressBar.displayName = "ProgressBar";

export { ProgressBar, progressBarVariants, type ProgressBarProps };