import React, { useState } from 'react';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setMessages([...messages, userMessage, { sender: 'bot', text: data.response }]);
      }
    } catch (err) {
      setError('Failed to communicate with chatbot');
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold mb-4">ASD Assistant Chatbot</h2>
      {error && <p className="text-red-500">{error}</p>}
      <div className="h-96 overflow-y-auto border p-4 mb-4 rounded">
        {messages.map((msg, index) => (
          <div key={index} className={`mb-2 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block p-2 rounded ${msg.sender === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 border p-2 rounded-l"
          placeholder="Ask about ASD or system usage..."
        />
        <button type="submit" className="bg-blue-600 text-white p-2 rounded-r hover:bg-blue-700">
          Send
        </button>
      </form>
    </div>
  );
}

export default Chatbot;