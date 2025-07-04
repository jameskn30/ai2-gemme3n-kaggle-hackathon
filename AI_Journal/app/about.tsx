import '../global.css';
import { View, Text, StyleSheet } from "react-native";

export default function AboutScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>About</Text>
      <Text style={styles.description}>
        This is the About screen. Here you can add information about your app or team.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#25292e",
    padding: 24,
  },
  title: {
    fontSize: 36,
    fontWeight: "bold",
    color: "red",
    marginBottom: 16,
  },
  description: {
    fontSize: 18,
    color: "#fff",
    textAlign: "center",
  },
});
