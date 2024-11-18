import React from 'react';
import { View, Text, Image, TouchableOpacity, Linking, ScrollView } from 'react-native';
import styles from './contactstyle'; // Import custom styles

function ContactPage() {
  const openLink = (url) => {
    Linking.openURL(url);
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* Background image */}
      <Image 
        source={require('../../assets/pothole_detection_image/Contact.png')} 
        style={styles.backgroundImage} 
      />

      {/* Header */}
      <View style={styles.headerContainer}>
        <Text style={styles.headerText}>CONTACT US</Text>
      </View>

      {/* Description */}
      <View style={styles.descriptionContainer}>
        <Text style={styles.description}>
          "For inquiries, please contact us through one of the platforms below. 
          We are dedicated to providing you with the best service possible."
        </Text>
      </View>

      {/* Footer with contact links */}
      <View style={styles.footer}>
        <Text style={styles.footerTitle}>Developers:</Text>
        <Text style={styles.developers}>
          Trương Trọng Hiếu, Nguyễn Minh Lộc, Nguyễn Tiến Sĩ, Nguyễn Thành Hiển
        </Text>

        <View style={styles.socialLinks}>
          {/* GitHub */}
          <TouchableOpacity onPress={() => openLink('https://github.com')}>
            <Image 
              source={require('../../assets/pothole_detection_image/github.png')} 
              style={styles.icon}
            />
          </TouchableOpacity>

          {/* LinkedIn */}
          <TouchableOpacity onPress={() => openLink('https://linkedin.com')}>
            <Image 
              source={require('../../assets/pothole_detection_image/linked.png')} 
              style={styles.icon}
            />
          </TouchableOpacity>

          {/* Twitter */}
          <TouchableOpacity onPress={() => openLink('https://twitter.com')}>
            <Image 
              source={require('../../assets/pothole_detection_image/twitter.png')} 
              style={styles.icon}
            />
          </TouchableOpacity>

          {/* Facebook */}
          <TouchableOpacity onPress={() => openLink('https://facebook.com')}>
            <Image 
              source={require('../../assets/pothole_detection_image/facebook.png')} 
              style={styles.icon}
            />
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
}

export default ContactPage;
