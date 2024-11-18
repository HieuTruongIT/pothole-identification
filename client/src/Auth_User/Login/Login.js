import React, { useState } from 'react';
import './LoginPage.css'
import Logo from '../../assets/AI.gif';

function HomeScreen() {
    const [message, setMessage] = useState("");

    const fetchMessage = async () => {
        const response = await fetch('http://localhost:5000/api/message');
        const data = await response.json();
        setMessage(data.message);
    };

    return (
        <div className='Container'>
            <div className='Header'>
                <div className='Logo'>
                    <img src="/logo.jpg" alt="Logo" />
                </div>
                <div className='Bar'>
                    <ul>
                        <li>Home</li>
                        <li>About</li>
                        <li>Contact</li>
                        <li>More</li>
                    </ul>
                </div>
                <div className='Router'>
                    <a href='../Component/Register/Register'>Tạo Tài Khoản |</a>
                    <a href='./Signin'>Đăng Nhập</a>
                </div>
            </div>
            <div className='Body'>
                <div className='Image'>
                    <img src={Logo} alt="AI" />
                </div>
                <div className='Content'>
                    <span>
                        Ứng dụng nhận diện khuôn mặt là một hệ thống sử dụng công nghệ trí tuệ nhân tạo để phân tích và nhận diện đặc điểm khuôn mặt của người dùng từ hình ảnh hoặc video. 
                        {message && <p>{message}</p>}
                    </span>
                    <button onClick={fetchMessage}>Get Message from Backend</button>
                </div>
            </div>
            <div className='Footer'>
                <div className='Personal'> </div>
                <div className='Product'> </div>
                <div className='Content'> </div>
            </div>
        </div>
    );
}

export default HomeScreen;
