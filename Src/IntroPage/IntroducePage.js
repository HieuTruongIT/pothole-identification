import React, { useState } from 'react';
import { View, Text, Image, Dimensions, TouchableOpacity, Modal } from 'react-native';
import Swiper from 'react-native-swiper';
import { introstyle as styles } from './introstyle';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const images = [
  require('../../assets/pothole_detection_image/pothole_image1.png'),
  require('../../assets/pothole_detection_image/pothole_image2.png'),
  require('../../assets/pothole_detection_image/pothole_image3.png'),
  require('../../assets/pothole_detection_image/pothole_image4.png'),
  require('../../assets/pothole_detection_image/pothole_image5.png'),
  require('../../assets/pothole_detection_image/pothole_image6.png'),
];

const IntroducePage = ({ navigation }) => {
  const [modalVisible, setModalVisible] = useState(false);

  return (
    <View style={styles.container}>
      {/* Swiper for images */}
      <Swiper
        style={styles.wrapper}
        showsButtons={false}
        loop={true}
        autoplay={true}
        autoplayTimeout={5}
        showsPagination={true}
      >
        {images.map((image, index) => (
          <View style={styles.slide} key={index}>
            <Image source={image} style={styles.image} resizeMode="cover" />
          </View>
        ))}
      </Swiper>

      {/* Header with Menu and Title */}
      <View style={styles.header}>
        {/* Hamburger Menu Icon */}
        <TouchableOpacity onPress={() => setModalVisible(true)} style={styles.menuIconContainer}>
          <Text style={styles.menuIcon}>☰</Text>
        </TouchableOpacity>

        {/* Title Container */}
        <View style={styles.titleContainer}>
          <Text style={styles.largeTitle}>Pothole Detection</Text>
          <Text style={styles.smallTitle}>Safe for Every Road</Text>
        </View>
      </View>

      {/* Modal Menu */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalView}>
          <TouchableOpacity onPress={() => { setModalVisible(false); navigation.navigate('Home'); }}>
            <Text style={styles.modalText}>Home</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => { setModalVisible(false); navigation.navigate('About'); }}>
            <Text style={styles.modalText}>About</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => { setModalVisible(false); navigation.navigate('Contact'); }}>
            <Text style={styles.modalText}>Contact</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => { setModalVisible(false); navigation.navigate('More'); }}>
            <Text style={styles.modalText}>More</Text>
          </TouchableOpacity>
        </View>
      </Modal>

      {/* Get Started Button */}
      <TouchableOpacity style={styles.getStartedButton} onPress={() => navigation.navigate('Home')}>
        <Text style={styles.getStartedText}>Get Started</Text>
      </TouchableOpacity>
    </View>
  );
};

export default IntroducePage;
