import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

export const introstyle = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000', // Nền đen để làm nổi bật chữ trắng
  },
  header: {
    position: 'absolute',  // Giữ header cố định
    top: 10,               // Vị trí cách đầu màn hình 15px
    left: 10,              // Cách cạnh trái màn hình 15px
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 1,             // Đảm bảo header luôn ở trên các nội dung khác
  },
  menuIcon: {
    top: 20,               // Vị trí cách đầu màn hình 15px
    left: 20,   
    fontSize: 24,
    color: 'white',  // Màu chữ trắng
    marginRight: 10,
  },
  titleContainer: {
    top: 80,               // Vị trí cách đầu màn hình 15px  
    left: 35,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column', // Đặt tiêu đề theo cột
  },
  largeTitle: {
    fontSize: 24,          // Kích thước chữ lớn cho "Pothole Detection"
    fontWeight: 'bold',    // In đậm
    color: 'white',        // Màu chữ trắng
  },
  smallTitle: {
    fontSize: 16,          // Kích thước chữ nhỏ hơn cho "Safe for Every Road"
    color: 'white',        // Màu chữ trắng
  },
  wrapper: {
    height: height,
  },
  slide: {
    justifyContent: 'center',
    alignItems: 'center',
    height: height,
    width: width,
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  modalView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',  // Màu nền mờ cho modal
  },
  modalText: {
    color: 'white',  // Màu chữ trắng
    fontSize: 18,
    margin: 10,      
    fontWeight: 'bold', // In đậm
  },
  getStartedButton: {
    position: 'absolute',
    bottom: 100, // Cách bottom 50px
    left: '27.5%', // Cách bên trái 27.5%
    right: '27.5%', // Cách bên phải 27.5%
    backgroundColor: '#00ffff', // Màu nền xanh dương
    borderRadius: 25,
    paddingVertical: 15,
    alignItems: 'center',
    justifyContent: 'center',
    width: '45%', // Chiều rộng 50% của màn hình
  },
  getStartedText: {
    color: 'white', // Màu chữ trắng
    fontWeight: 'bold', // Chữ in đậm
    fontSize: 18,
  },
});
