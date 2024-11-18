import React from 'react';
import { useNavigate } from 'react-router-dom'; 
import './HomePage.css'; 
import Logo from '../../assets/AI.gif';

function HomeScreen() {

  const navigate = useNavigate();


  const handleRegisterClick = () => {
    navigate('/register'); 
  };

  const handleSignInClick = () => {
    navigate('/signin'); 
  };

  return (
    <div className="container">

      <header className="header">
        <div className="logo">
          <img src={Logo} alt="Logo" />
        </div>
        <nav className="navigation-bar">
          <ul>
            <li>Home</li>
            <li>About</li>
            <li>Contact</li>
            <li>More</li>
          </ul>
        </nav>
        <div className="router-links">

          <a onClick={handleRegisterClick} style={{ cursor: 'pointer' }}>Tạo Tài Khoản |</a>
          <a onClick={handleSignInClick} style={{ cursor: 'pointer' }}>Đăng Nhập</a>
        </div>
      </header>

      <main className="body">
        <div className="image-section">
          <img src={Logo} alt="Logo" />
        </div>
        <div className="content-section">
          <p>
            Ứng dụng nhận diện khuôn mặt là một hệ thống sử dụng công nghệ trí tuệ nhân tạo để phân tích và nhận diện đặc điểm khuôn mặt của người dùng từ hình ảnh hoặc video. Ứng dụng này có thể tự động xác định danh tính, hỗ trợ đăng nhập an toàn mà không cần mật khẩu, giúp bảo mật và tối ưu hoá trải nghiệm người dùng. Với tính năng này, hệ thống có thể ứng dụng trong nhiều lĩnh vực như bảo mật, thanh toán trực tuyến, kiểm soát ra vào, và quản lý nhân sự.
          </p>
        </div>
      </main>

      <footer className="footer">
        <div className="footer-section">
          <h4>Quick Links</h4>
          <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#">Services</a></li>
            <li><a href="#">Contact</a></li>
          </ul>
        </div>
        <div className="footer-section">
          <h4>Contact Us</h4>
          <p>Email: contact@example.com</p>
          <p>Phone: +123 456 7890</p>
        </div>
        <div className="footer-section">
          <h4>Follow Us</h4>
          <div className="social-icons">
            <a href="#"><i className="fab fa-facebook-f"></i></a>
            <a href="#"><i className="fab fa-twitter"></i></a>
            <a href="#"><i className="fab fa-instagram"></i></a>
            <a href="#"><i className="fab fa-linkedin-in"></i></a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default HomeScreen;
