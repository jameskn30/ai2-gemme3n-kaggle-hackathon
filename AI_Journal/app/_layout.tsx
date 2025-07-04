import { Stack } from 'expo-router';
import '../global.css';
import { StatusBar } from 'expo-status-bar';
//what is stack? https://docs.expo.dev/tutorial/add-navigation/#what-is-a-stack

export default function RootLayout() {
  return (
    <>
      <Stack>
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="(screens)/about" options={{ headerShown: false }}/>
        <Stack.Screen name="(screens)/settings" options={{ headerShown: false, presentation: 'modal' }}/>
        <Stack.Screen name="(screens)/statistics" options={{ headerShown: false, presentation: 'modal' }}/>
      </Stack>
      <StatusBar style="dark" />
    </>
  );
}
