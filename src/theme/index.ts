import { createTheme } from "@mantine/core";

import { orange } from "./colors";

export const theme = createTheme({
  cursorType: "pointer",
  defaultRadius: "0",
  black: "#000",
  primaryColor: "orange",
  colors: {
    orange,
  },
  headings: { fontFamily: 'ADLaM Display' },
  fontFamily: 'DM Sans Variable',
});