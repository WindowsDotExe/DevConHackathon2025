import React, { useRef, useEffect, useState } from 'react';
import '../Styles/mainPage.css';

function MainPage() {
  const videoRef = useRef(null);
  const [logs, setLogs] = useState(["System initialized...", "Waiting for events..."]);

  useEffect(() => {
    async function startVideo() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setLogs((prevLogs) => [...prevLogs, "Camera feed started."]);
        }
      } catch (err) {
        console.error('Error accessing camera: ', err);
        setLogs((prevLogs) => [...prevLogs, "Error accessing camera."]);
      }
    }

    startVideo();
  }, []);

  return (
    <div className="container">
      {/* Left Section - Webcam Feed */}
      <div className="left-section">
        <h1>Live Feed</h1>
        <video ref={videoRef} className="video-feed" />
      </div>

      {/* Right Section - Logs & Info Box */}
      <div className="right-section">
        <LogComponent logs={logs} />
        <InfoBox setLogs={setLogs} />
      </div>
    </div>
  );
}

// Log Component (Top Right)
function LogComponent({ logs }) {
  return (
    <div className="log-box">
      <h2>Logs</h2>
      <div className="log-content">
        {logs.map((log, index) => (
          <p key={index}>{log}</p>
        ))}
      </div>
    </div>
  );
}

// Info Box Component (Bottom Right)
function InfoBox({ setLogs }) {
  const handleAction = (action) => {
    setLogs((prevLogs) => [...prevLogs, `User selected: ${action}`]);
  };

  return (
    <div className="info-box">
      <h2>Info</h2>
      <div className="info-text">Medium</div> {/* Hardcoded info level */}
      <button className="info-btn accept" onClick={() => handleAction("Accept")}>Accept</button>
      <button className="info-btn ignore" onClick={() => handleAction("Ignore")}>Ignore</button>
    </div>
  );
}

export default MainPage;
