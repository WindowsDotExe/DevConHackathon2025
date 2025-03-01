import React, { useRef, useEffect } from 'react';

function MainPage() {
  const videoRef = useRef(null);

  useEffect(() => {
    // Request access to the webcam
    async function startVideo() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          // Set the video element's srcObject to the webcam stream
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (err) {
        console.error('Error accessing camera: ', err);
      }
    }

    startVideo();
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',  // horizontally center
        justifyContent: 'center',  // vertically center 
        height: '100vh', // Make the container fill the view height
      }}
    >
      <h1>Hello, this is the main page!</h1>
      
      {/* Webcam Feed */}
      <video
        ref={videoRef}
        style={{
          width: '640px',  
          height: '480px',
          backgroundColor: 'black', 
          marginTop: '20px',
        }}
      />
    </div>
  );
}

export default MainPage;