import { Stack } from "expo-router";
import "../global.css";
//what is stack? https://docs.expo.dev/tutorial/add-navigation/#what-is-a-stack

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen name="index" options={{ title: "Home" }} />
      <Stack.Screen name="about" options={{ title: "About" }} />
    </Stack>
  );
}
