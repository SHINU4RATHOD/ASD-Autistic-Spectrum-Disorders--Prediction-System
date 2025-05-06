import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Webcam from './components/Webcam';
import Upload from './components/Upload';
import Analyzer from './components/Analyzer';
import Chatbot from './components/Chatbot';
import './tailwind.css';

function App() {
  const [prediction, setPrediction] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Navbar */}
        <nav className="bg-blue-600 p-4">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-white text-2xl font-bold">ASD Prediction System</h1>
            <div className="space-x-4">
              <Link to="/" className="text-white hover:text-blue-200">Home</Link>
              <Link to="/webcam" className="text-white hover:text-blue-200">Real-Time</Link>
              <Link to="/upload" className="text-white hover:text-blue-200">Upload Video</Link>
              <Link to="/analyzer" className="text-white hover:text-blue-200">AI Analyzer</Link>
              <Link to="/chatbot" className="text-white hover:text-blue-200">Chatbot</Link>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <div className="container mx-auto p-4">
          <Routes>
            <Route path="/" element={
              <div className="text-center">
                <h2 className="text-3xl font-semibold mb-4">Welcome to ASD Detection</h2>
                <p className="text-lg">Use real-time webcam or upload videos to detect ASD behaviors.</p>
              </div>
            } />
            <Route path="/webcam" element={<Webcam setPrediction={setPrediction} />} />
            <Route path="/upload" element={<Upload setPrediction={setPrediction} />} />
            <Route path="/analyzer" element={<Analyzer prediction={prediction} />} />
            <Route path="/chatbot" element={<Chatbot />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;