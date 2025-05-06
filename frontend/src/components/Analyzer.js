import React, { useState, useEffect } from 'react';

function Analyzer({ prediction }) {
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (prediction && prediction.features) {
      const fetchAnalysis = async () => {
        try {
          const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: prediction.features })
          });
          const data = await response.json();
          if (data.error) {
            setError(data.error);
          } else {
            setAnalysis(data);
          }
        } catch (err) {
          setError('Failed to analyze features');
        }
      };
      fetchAnalysis();
    }
  }, [prediction]);

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold mb-4">AI Behavior Analyzer</h2>
      {error && <p className="text-red-500">{error}</p>}
      {!prediction && <p className="text-gray-500">No prediction data available. Please run a detection first.</p>}
      {analysis && (
        <div className="space-y-4">
          {Object.entries(analysis).map(([behavior, details]) => (
            <div key={behavior} className="border p-4 rounded">
              <h3 className="text-xl font-medium capitalize">{behavior.replace('_', ' ')}</h3>
              <p className="text-gray-600">{details.description}</p>
              <p className="text-sm">Detected: <span className={details.detected ? 'text-green-500' : 'text-red-500'}>{details.detected ? 'Yes' : 'No'}</span></p>
              {Object.entries(details).map(([key, value]) => (
                key !== 'detected' && key !== 'description' && (
                  <p key={key} className="text-sm">{key.replace('_', ' ')}: {value.toFixed(2)}</p>
                )
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Analyzer;