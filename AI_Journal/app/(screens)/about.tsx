import '@/global.css';
import { View, Text, StyleSheet, Alert } from 'react-native';
import ImageViewer from '@/app/components/ImageVIewer';
import Button from '@/app/components/Button';

export default function AboutScreen() {
  const PlaceHolderImage = require('@/assets/images/therock.webp');
  return (
    <View style={styles.container}>
      <Text style={styles.title}>About</Text>
      <View style={styles.imageContainer}>
        <ImageViewer imageSource={PlaceHolderImage} width={320} height={440} />
      </View>
      <Text style={styles.description}>
        This is the About screen. Here you can add information about your app or
        team.
      </Text>
      <Button
        title="Choose a photo"
        onPress={() => {
          Alert.alert(
            'Choose a photo',
            'Do you want to choose a photo?',
            [
              { text: 'No', style: 'cancel' },
              {
                text: 'Yes',
                onPress: () => {
                  console.log('Yes');
                },
              },
            ],
            { cancelable: true }
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 24,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: 'red',
    marginBottom: 16,
  },
  imageContainer: {
    flex: 0,
  },
  image: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
  description: {
    fontSize: 18,
    color: 'black',
    textAlign: 'center',
  },
});
