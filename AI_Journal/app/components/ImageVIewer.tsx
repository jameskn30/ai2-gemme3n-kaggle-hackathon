import React from 'react';
import { View, StyleSheet, ImageSourcePropType } from 'react-native';
import { Image } from 'expo-image';

type ImageViewerProps = {
  imageSource: string | ImageSourcePropType;
  width: number;
  height: number;
};

function ImageViewer({ imageSource, width, height }: ImageViewerProps) {
  return (
    <View style={[styles.container, { width, height }]}>
      <Image
        source={imageSource}
        style={{ width: width, height: height, borderRadius: 18 }}
        contentFit="cover"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default ImageViewer;
