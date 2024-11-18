import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff', // Màu nền của trang
  },
  imageContainer: {
    width: '100%',
    height: 630, // Chiều cao có thể điều chỉnh theo nhu cầu
    justifyContent: 'flex-start', // Căn ảnh về phía trên
    alignItems: 'center',
    position: 'relative', // Để cho phép các thành phần con được căn chỉnh tương đối
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain', // Đảm bảo ảnh không bị cắt xén
  },
  greetingText: {
    position: 'absolute', // Đặt câu chào ở vị trí tuyệt đối
    top: 65, // Khoảng cách từ trên cùng
    alignItems: 'center',
    width: '100%', // Đảm bảo câu chào chiếm toàn bộ chiều rộng
    textAlign: 'center', // Căn giữa
    fontSize: 13, // Kích thước chữ
    color: '#fff', // Màu chữ (có thể thay đổi theo thiết kế)
  },
  header: {
    position: 'absolute', // Đặt tiêu đề ở vị trí tuyệt đối
    top: 100, // Khoảng cách từ trên cùng
    alignItems: 'center',
    width: '100%', // Đảm bảo tiêu đề chiếm toàn bộ chiều rộng
  },
  headerText: {
    fontSize: 45,
    fontWeight: 'bold',
    color: '#ffa07a', // Màu chữ (có thể thay đổi theo thiết kế)
  },
  descriptionContainer: {
    position: 'absolute', // Đặt mô tả ở vị trí tuyệt đối
    bottom: 20, // Khoảng cách từ dưới lên
    alignItems: 'center', // Căn giữa theo chiều ngang
    width: '100%', // Đảm bảo mô tả chiếm toàn bộ chiều rộng
  },
  description: {
    fontSize: 16,
    textAlign: 'center', // Căn giữa chữ
    paddingHorizontal: 20, // Khoảng cách bên trái và bên phải
    color: '#ffefd5', // Màu chữ (có thể thay đổi theo thiết kế)
  },
  footer: {
    padding: 20,
    backgroundColor: '#f8f8f8', // Màu nền của footer
    alignItems: 'center', // Căn giữa theo chiều ngang
    justifyContent: 'center', // Căn giữa theo chiều dọc
    // Thêm hiệu ứng bóng đổ
    shadowColor: '#000', // Màu bóng
    shadowOffset: {
      width: 0, // Độ lệch theo chiều ngang
      height: 4, // Tăng độ lệch chiều dọc để bóng dài hơn
    },
    shadowOpacity: 0.5, // Tăng độ mờ của bóng
    shadowRadius: 6, // Tăng độ mềm mại của bóng
    elevation: 8, // Tăng độ cao cho Android
    borderRadius: 10, // Bo góc footer nếu cần
    margin: 10, // Giảm khoảng cách nếu cần
  },
  footerText: {
    fontSize: 14,
    textAlign: 'justify', // Căn đều văn bản
    color: '#333', // Màu chữ
  },
  feedbackContainer: {
    padding: 20,
    alignItems: 'center', // Căn giữa theo chiều ngang
  },
  feedbackText: {
    fontSize: 16,
    marginBottom: 10,
    color: '#333', // Màu chữ
  },
  feedbackInput: {
    width: '100%',
    borderColor: '#ccc', // Màu viền
    borderWidth: 1,
    borderRadius: 5,
    padding: 10,
    marginBottom: 10,
    height: 100, // Chiều cao ô nhập
    textAlignVertical: 'top', // Đặt con trỏ ở đầu ô nhập
  },
});

export default styles;

