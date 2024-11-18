import React, { useState, useEffect, useRef } from 'react';
import './NoctisAIgen.css';

function NoctisAIScreen() {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [mediaType, setMediaType] = useState('image');
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.2);
    const [iouThreshold, setIouThreshold] = useState(0.7);
    const [isDragging, setIsDragging] = useState(false);
    const [processedImagePath, setProcessedImagePath] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [processedVideoPath, setProcessedVideoPath] = useState(null);

    const previewVideoRef = useRef(null);
    const processedVideoRef = useRef(null);

    useEffect(() => {
        if (selectedFile) {
            const url = URL.createObjectURL(selectedFile);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        }
    }, [selectedFile]);

    const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

    const validateFile = (file) => {
        const isValidImage = mediaType === 'image' && ['image/png', 'image/jpg', 'image/jpeg', 'image/gif', 'image/webp'].includes(file.type);
        const isValidVideo = mediaType === 'video' && file.type === 'video/mp4';
        return isValidImage || isValidVideo;
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file && validateFile(file)) {
            setSelectedFile(file);
            setIsLoading(true);
            sendImageNameToAPI(file.name);
            fetchProcessedMedia();
        } else {
            alert('Invalid file format');
        }
    };

    const sendImageNameToAPI = async (imageName) => {
        const apiUrl = 'http://127.0.0.1:8000/save-image-name/';
        try {
            await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({ image_name: imageName }),
            });
            console.log('Image name sent successfully');
        } catch (error) {
            console.error('Network error:', error);
        }
    };

    const fetchProcessedMedia = async () => {
        try {
            const endpoint = mediaType === 'image' ? '/get-output-image/' : '/get-output-video/';
            const response = await fetch(`http://127.0.0.1:8000${endpoint}`);

            if (response.ok) {
                const blob = await response.blob();
                const objectUrl = URL.createObjectURL(blob);

                if (mediaType === 'image') {
                    setProcessedImagePath(objectUrl);
                } else {
                    setProcessedVideoPath(objectUrl);
                }
            } else {
                console.error('Failed to fetch processed media');
            }
        } catch (error) {
            console.error('Error fetching processed media:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDragEnter = () => setIsDragging(true);
    const handleDragLeave = () => setIsDragging(false);
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && validateFile(file)) {
            setSelectedFile(file);
            setIsLoading(true);
            sendImageNameToAPI(file.name);
            fetchProcessedMedia();
        } else {
            alert('Invalid file format');
        }
    };

    const callWebcamAPI = async () => {
        const apiUrl = 'http://127.0.0.1:8000/execute-webcam-script/';
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            if (response.ok) {
                console.log('Webcam script executed successfully');
            } else {
                console.error('Failed to execute webcam script');
            }
        } catch (error) {
            console.error('Network error while calling webcam API:', error);
        }
    };

    const toggleWebcam = async () => {
        await callWebcamAPI();
    };

    const handleVideoEnd = () => {
        if (previewVideoRef.current && processedVideoRef.current) {
            previewVideoRef.current.pause();
            processedVideoRef.current.pause();
        }
    };

    return (
        <div className="noctisAIScreen">
            {isLoading && (
                <div className="loading-overlay">
                    <div className="loading-spinner"></div>
                </div>
            )}
            {isSidebarOpen && (
                <div className="sidebar">
                    <button className="close-btn" onClick={toggleSidebar}>×</button>
                    <h2 className="sidebar-title">NoctisAI ☠️</h2>
                    <h3>Image/Video</h3>
                    <div className="radio-group">
                        <label>
                            <input
                                type="radio"
                                name="media"
                                value="image"
                                checked={mediaType === 'image'}
                                onChange={() => setMediaType('image')}
                            /> Image
                        </label>
                        <label>
                            <input
                                type="radio"
                                name="media"
                                value="video"
                                checked={mediaType === 'video'}
                                onChange={() => setMediaType('video')}
                            /> Video
                        </label>
                    </div>

                    <div className="file-upload">
                        <label>Input {mediaType === 'image' ? 'Image' : 'Video'} File</label>
                        <div
                            className="file-input"
                            onDragOver={(e) => e.preventDefault()}
                            onDragEnter={handleDragEnter}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            style={{
                                border: isDragging ? '2px dashed red' : '2px solid #ccc'
                            }}
                        >
                            {selectedFile ? (
                                <div className="file-details">
                                    <span>{selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
                                    <button onClick={() => setSelectedFile(null)}>×</button>
                                </div>
                            ) : (
                                <div>
                                    <p style={{ textAlign: 'left' }}>Drag and Drop File Here</p>
                                    <input
                                        id="file-input"
                                        type="file"
                                        onChange={handleFileChange}
                                        style={{ display: 'none' }}
                                    />
                                    <p style={{ textAlign: 'left', fontSize: '15px' }} >
                                        Limit File Type - {mediaType === 'image' ? 'JPG, PNG' : 'MP4'}
                                    </p>
                                    <button className="browse-btn" onClick={() => document.getElementById('file-input').click()}>Browse File</button>
                                </div>
                            )}
                        </div>
                    </div>

                    <button className="webcam-btn" onClick={toggleWebcam}>Webcam</button>
                    <div className="slider">
                        <label>Confidence Threshold: <span className="slider-value">{confidenceThreshold}</span></label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={confidenceThreshold}
                            onChange={(e) => setConfidenceThreshold(e.target.value)}
                        />
                    </div>
                    <div className="slider">
                        <label>IOU Threshold: <span className="slider-value">{iouThreshold}</span></label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={iouThreshold}
                            onChange={(e) => setIouThreshold(e.target.value)}
                        />
                    </div>
                </div>
            )}
            <div className="main-content">
                <div className="header-buttons">
                    <button className="toggle-btn" onClick={toggleSidebar}>
                        {isSidebarOpen ? '👈 Close Sidebar' : 'Open Sidebar 👉'}
                    </button>
                    <div className="button-container">
                        <button className="deploy-btn">
                            <span className='deploy-text'>Deploy</span>
                        </button>
                        <button className="ellipsis-btn">
                            <span className="ellipsis">⋮</span>
                        </button>
                    </div>
                </div>
                <div className="header-container">
                    <h1>Real Time Image Detection 🚦🚧 </h1>
                    <div className="vehicles">
                        <div className="vehicle bike">🚲</div>
                        <div className="vehicle motorbike">🛵</div>
                        <div className="vehicle train">🚗</div>
                    </div>
                </div>
                <h3>Comprehensive Object Recognition System for Real-Time Detection</h3>
                <div className="image-container">
                    <div className="image">
                        <h3>Uploaded Image</h3>
                    </div>
                    <div className="image">
                        <h3>Image Detection</h3>
                    </div>
                </div>
                <div className="image-detection-container">
                    <div className="preview-container">
                        {previewUrl && mediaType === 'image' && (
                            <img src={previewUrl} alt="Preview" className="preview-img" />
                        )}
                        {previewUrl && mediaType === 'video' && (
                            <video ref={previewVideoRef} autoPlay loop onEnded={handleVideoEnd} className="preview-video">
                                <source src={previewUrl} type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        )}
                    </div>
                    <div className="output-container">
                        {processedImagePath && mediaType === 'image' && (
                            <img src={processedImagePath} alt="Processed" className="output-img" />
                        )}
                        {processedVideoPath && mediaType === 'video' && (
                            <video ref={processedVideoRef} autoPlay loop onEnded={handleVideoEnd} controls>
                                <source src={processedVideoPath} type="video/webm" />
                                <source src={processedVideoPath.replace('.webm', '.mp4')} type="video/mp4" />
                                Your browser does not support the video tag.
                            </video>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default NoctisAIScreen;
