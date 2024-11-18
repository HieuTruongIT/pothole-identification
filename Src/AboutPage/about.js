import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TextInput, Button, Alert } from 'react-native';
import styles from './aboutstyle'; 

function AboutPage() {
  const [feedback, setFeedback] = useState('');

  const handleFeedbackSubmit = () => {
    // Xử lý phản hồi (có thể gửi đến server hoặc lưu vào trạng thái)
    Alert.alert("Feedback Submitted", "Thank you for your feedback!");
    setFeedback(''); // Xóa ô nhập liệu sau khi gửi
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.imageContainer}>
        <Image 
          source={require('../../assets/pothole_detection_image/About.png')} 
          style={styles.image} // Đảm bảo style được áp dụng cho ảnh
        />
        
        {/* Câu chào nằm bên trên phần "ABOUT US" */}
        <Text style={styles.greetingText}>Hello! Hope you are having a great day!</Text>

        <View style={styles.header}>
          <Text style={styles.headerText}>ABOUT US</Text>
        </View>
        
        <View style={styles.descriptionContainer}>
          <Text style={styles.description}>
            Introducing our innovative pothole detection app that harnesses the power of AI 
            {'\n'}to swiftly identify road defects.
          </Text>
        </View>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          "Our pothole detection application is proudly developed by a dedicated team of four students: Hieu, Hien, Si and Loc of the University of Transport. Each member brings skills to building application and distinctive expertise, ensuring a comprehensive approach to solving road safety issues. We are passionate about using technology to make our roads safer and more reliable for everyone. Together, we believe in creating innovative solutions that positively impact communities and enhance the overall driving experience."
        </Text>
      </View>

      {/* Phản hồi từ người dùng */}
      <View style={styles.feedbackContainer}>
        <Text style={styles.feedbackText}>You can send us feedback:</Text>
        <TextInput 
          style={styles.feedbackInput} 
          placeholder="Type your feedback here..."
          value={feedback}
          onChangeText={setFeedback}
        />
        <Button 
          title="Submit Feedback" 
          onPress={handleFeedbackSubmit} 
        />
      </View>
    </ScrollView>
  );
}

export default AboutPage;
