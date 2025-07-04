import { Text, View, StyleSheet, Alert, FlatList } from 'react-native';
import FAB from './components/FAB';
import CardView from './components/CardView';
import { LinearGradient } from 'expo-linear-gradient';
import Header from './components/Header';

export default function Index() {
  return (
    <LinearGradient
      colors={['#F7F0F0', '#fff2f2']}
      style={styles.container}
    >
      {/* Scrollable list view with long text */}
      <View style={{ flex: 1, width: '100%' }}>
        <Header />
        <Text
          style={{
            fontWeight: 'bold',
            fontSize: 18,
            marginBottom: 10,
            marginLeft: 16,
            alignSelf: 'flex-start',
          }}
        >
          June
        </Text>
        <View style={{ flex: 1 }}>
          <FlatList
            data={Array.from({ length: 10 }, (_, i) => i)}
            keyExtractor={(item) => item.toString()}
            renderItem={() => (
              <CardView
                title={'Very Long Scrollable Text'}
                content={
                  'This is a very long line of text for the scrollable list view. You can scroll down to see more content. This sentence is intentionally extended to ensure it is at least five hundred characters long. The purpose of this text is to provide enough content so that the FlatList can demonstrate its scrolling capabilities. By adding more and more words, we ensure that the user will need to scroll to read everything. This is useful for testing layouts, overflow, and user experience. Sometimes, developers need to see how their UI behaves with lots of text, so this example is intentionally verbose. Keep scrolling to see how the CardView component handles large amounts of content. This should be enough to reach the five hundred character requirement for this prompt. If not, we can always add more filler text to ensure the length is sufficient for the test case.'
                }
              />
            )}
          />
        </View>
      </View>
      <FAB
        onPress={() => {
          Alert.alert('FAB pressed');
        }}
      />
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 25,
  },
  text: {
    fontWeight: 'bold',
    fontSize: 50,
    color: 'red',
  },
});
