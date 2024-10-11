import { CopyButton, Tooltip, ActionIcon, rem } from "@mantine/core";
import { IconCheck, IconCopy } from "@tabler/icons-react";

interface CopyProps {
  value: string;
}

export const Copy = ({ value }: CopyProps) => {
  return (
    <CopyButton value={value} timeout={2000}>
      {({ copied, copy }) => (
        <Tooltip label={copied ? "Copied" : "Copy"} withArrow position="left">
          <ActionIcon variant="subtle" onClick={copy}>
            {copied ? (
              <IconCheck style={{ width: rem(12) }} />
            ) : (
              <IconCopy style={{ width: rem(12) }} />
            )}
          </ActionIcon>
        </Tooltip>
      )}
    </CopyButton>
  );
};
