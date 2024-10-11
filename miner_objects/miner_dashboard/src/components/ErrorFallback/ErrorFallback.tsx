import React from "react";
import { Center } from "@mantine/core";

interface ErrorFallbackProps {
  error: Error;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error }) => {
  return (
    <Center>
      <div className="text-center">
        <div>{error.message}</div>
      </div>
    </Center>
  );
};