import {
  Anchor,
  Container,
  Group,
  Burger,
  Title,
  Stack,
  Badge,
} from "@mantine/core";
import Avatar from "boring-avatars";

import { isInChallengePeriod, shortenAddress } from "../../utils";

import { Copy } from "../Copy";

const url = (uid: string) => `https://x.taostats.io/account/${uid}`;

export const AppHeader = ({
  opened,
  toggle,
  data,
}: {
  opened: boolean,
  toggle: () => void,
  data: any,
}) => {
  const { challengeperiod, hotkey } = data.data[0];
  const isChallengePeriod = isInChallengePeriod(challengeperiod);
  
  return (
    <Container fluid h="100%">
      <Group h="100%" py="xs" justify="space-between">
        <Stack>
          <Title order={2} c="orange">taoshi</Title>
        </Stack>
        
        <Group gap="xl">
          <Stack justify="flex-end" align="flex-end" gap="6">
            <Group gap="xs">
              <Anchor
                c="orange"
                size="sm"
                href={url(hotkey)}
              >
                {shortenAddress(hotkey, 10)}
              </Anchor>
              <Copy value={hotkey} />
            </Group>
            
            {isChallengePeriod && (
              <Group justify="center" align="center">
                <Badge radius="0" variant="light" color="orange" size="xs">
                  In Challenge Period
                </Badge>
              </Group>
            )}
          </Stack>
          
          <Avatar
            size={32}
            name={hotkey.toString()}
            variant="pixel"
            colors={["#E46933", "#282828", "#FFF7F2", "#EDE9D0"]}
          />
        </Group>
        
        
        <Burger
          opened={opened}
          onClick={toggle}
          hiddenFrom="sm"
          size="sm"
        />
      </Group>
    </Container>
  );
};