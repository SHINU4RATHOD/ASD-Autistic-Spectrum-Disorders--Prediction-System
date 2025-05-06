import React, { useRef, useState, useEffect } from 'react';

function Webcam({ setPrediction }) {
  const videoRef = useRef(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream;
    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (err) {
        setError('Failed to access webcam');
      }
    };
    startWebcam();

    const interval = setInterval(async () => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
        const frameData = canvas.toDataURL('image/jpeg').split(',')[1];

        try {
          const response = await fetch('http://localhost:5000/predict/webcam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: [frameData] })
          });
          const data = await response.json();
          if (data.error) {
            setError(data.error);
          } else {
            setResult(data);
            setPrediction(data);
          }
        } catch (err) {
          setError('Failed to process webcam feed');
        }
      }
    }, 1000); // Process every second

    return () => {
      clearInterval(interval);
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [setPrediction]);

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold mb-4">Real-Time ASD Detection</h2>
      {error && <p className="text-red-500">{error}</p>}
      <video ref={videoRef} autoPlay className="w-full max-w-md mx-auto mb-4" />
      {result && (
        <div className="text-center">
          <p className="text-lg">Prediction: <span className="font-bold">{result.label}</span></p>
          <p className="text-lg">ASD Probability: <span className="font-bold">{(result.probability * 100).toFixed(2)}%</span></p>
        </div>
      )}
    </div>
  );
}

export default Webcam;