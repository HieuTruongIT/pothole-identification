import React from 'react';
import { Link } from 'react-router-dom';
import styles from './RegisterPage.css'; 

function RegisterScreen() {
    return (
        <div className={styles.container}>
            <div className={styles.logo}>
                <img src="/Image/Logo/logo.jpg" alt="Logo" className={styles.logoImage} />
            </div>
            <div className={styles.form}>
                <div className={styles.title}>
                    <img src="/Image/Logo/Face_ID_logo.jpg" alt="Face_ID_LOGO" className={styles.titleImage} />
                    <span className={styles.titleText}>Face ID</span>
                </div>
                <div className={styles.enter}>
                    <div className={styles.inputGroup}>
                        <i className="fa-solid fa-user"></i>
                        <input type="text" placeholder="Nhập Tên người dùng..." className={styles.input} />
                    </div>
                    <div className={styles.inputGroup}>
                        <i className="fa-solid fa-envelope"></i>
                        <input type="email" placeholder="Nhập email..." className={styles.input} />
                    </div>
                    <div className={styles.inputGroup}>
                        <i className="fa-solid fa-lock"></i>
                        <input type="password" placeholder="Nhập mật khẩu..." className={styles.input} />
                    </div>
                </div>
                <div className={styles.login}>
                    <button className={styles.submitButton}>
                        <Link to="/Client/Log_In/Log_in" className={styles.link}>Đăng Ký</Link>
                    </button>
                </div>
                <div className={styles.forget}>
                    <Link to="#" className={styles.link}>Quên Mật Khẩu</Link>
                </div>
                <div className={styles.create}>
                    Nếu bạn đã có tài khoản, <Link to="/Client/Log_In/Log_in" className={styles.link}>Đăng Nhập</Link>
                </div>
            </div>
        </div>
    );
}

export default RegisterScreen;
