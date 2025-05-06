import React, { useState } from 'react';

function Upload({ setPrediction }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/predict/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
        setPrediction(data);
      }
    } catch (err) {
      setError('Failed to process video');
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold mb-4">Upload Video for ASD Detection</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="file"
          accept=".mp4,.avi"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
        >
          Analyze Video
        </button>
      </form>
      {error && <p className="text-red-500 mt-4">{error}</p>}
      {result && (
        <div className="mt-4 text-center">
          <p className="text-lg">Prediction: <span className="font-bold">{result.label}</span></p>
          <p className="text-lg">ASD Probability: <span className="font-bold">{(result.probability * 100).toFixed(2)}%</span></p>
          {result.visualization && (
            <video controls src={`http://localhost:5000${result.visualization}`} className="w-full max-w-md mx-auto mt-4" />
          )}
        </div>
      )}
    </div>
  );
}

export default Upload;