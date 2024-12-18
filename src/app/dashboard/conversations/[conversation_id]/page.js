'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3000';

export default function ConversationPage() {
  const [conversationDetails, setConversationDetails] = useState({
    threads: [],
    files: [],
  });
  const [chatInput, setChatInput] = useState('');
  const [selectedPdfUrl, setSelectedPdfUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const {conversation_id: conversationId} = useParams();
  const chatSectionRef = useRef(null);
  const websocketRef = useRef(null);

  // Helper to scroll chat section to the bottom
  const scrollToBottom = () => {
    if (chatSectionRef.current) {
      chatSectionRef.current.scrollTop = chatSectionRef.current.scrollHeight;
    }
  };

  // Fetch conversation details
  const fetchConversationDetails = async () => {
    setIsLoading(true);
    setErrorMessage('');
    try {
      const [conversationResponse, filesResponse] = await Promise.all([
        axios.get(`${BASE_URL}/api/conversations/${conversationId}`),
        axios.get(`${BASE_URL}/api/get_uploaded_files/${conversationId}`),
      ]);

      setConversationDetails({
        ...conversationResponse.data,
        files: filesResponse.data.files || [],
      });
    } catch (error) {
      setErrorMessage('Failed to fetch conversation details. Please try again.');
      console.log('Error fetching conversation details:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (conversationId) fetchConversationDetails();
  }, [conversationId]);

  // Set up WebSocket
  useEffect(() => {
    // Open WebSocket connection
    websocketRef.current = new WebSocket(`${BASE_URL}/api/llm_chat/${conversationId}`);

    websocketRef.current.onopen = () => {
      console.log("WebSocket connection established");
    };

    websocketRef.current.onmessage = (event) => {
      const aiReply = { role: 'ai', message: event.data };
      setConversationDetails((prev) => ({
        ...prev,
        threads: [...(prev.threads || []), aiReply],
      }));
      scrollToBottom();
    };

    websocketRef.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    websocketRef.current.onclose = () => {
      console.log("WebSocket connection closed");
    };

    return () => {
      // Clean up WebSocket on component unmount
      websocketRef.current?.close();
    };
  }, [conversationId]);

  useEffect(scrollToBottom, [conversationDetails.threads]);

  const handleFileSelection = (event) => {
    const selectedFiles = Array.from(event.target.files).map((file) => ({ name: file.name, file }));
    setConversationDetails((prev) => ({
      ...prev,
      files: [...(prev.files || []), ...selectedFiles],
    }));
  };

  const handleFileUpload = async () => {
    const filesToUpload = (conversationDetails.files || []).filter((file) => file.file);
    if (filesToUpload.length === 0) {
      setErrorMessage('No files selected for upload.');
      return;
    }

    const formData = new FormData();
    filesToUpload.forEach((file) => formData.append('files', file.file));

    setIsUploading(true);
    setErrorMessage('');
    try {
      await axios.post(`${BASE_URL}/api/upload/${conversationId}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      await fetchConversationDetails();
    } catch (error) {
      setErrorMessage('File upload failed. Please try again.');
      console.log('Error uploading files:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileClick = (file) => {
    if (file && file.file_name) {
      setSelectedPdfUrl(`${BASE_URL}/api/view_pdf/${conversationId}/${file.file_name}`);
    } else {
      setSelectedPdfUrl(null);
    }
  };

  const handleChatInput = () => {
    if (!chatInput.trim()) {
      setErrorMessage('Message cannot be empty.');
      return;
    }

    const userMessage = { role: 'user', message: chatInput };
    setConversationDetails((prev) => ({
      ...prev,
      threads: [...(prev.threads || []), userMessage],
    }));

    websocketRef.current?.send(chatInput);
    setChatInput('');
  };

  const { threads = [], files = [] } = conversationDetails;

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      <div className="flex-grow flex flex-col md:flex-row gap-6 px-6 py-8">
        {/* Left Column: File Upload */}
        <div className="w-full md:w-1/5 bg-gray-800 p-6 rounded-lg shadow-lg space-y-4">
          <h2 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300">
            Upload PDFs
          </h2>
          <input
            type="file"
            multiple
            accept="application/pdf"
            onChange={handleFileSelection}
            className="block w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:bg-cyan-500 file:text-black hover:file:bg-teal-400"
          />
          <button
            onClick={handleFileUpload}
            disabled={isUploading}
            className={`w-full py-2 rounded-lg ${
              isUploading ? 'bg-gray-500' : 'bg-gradient-to-r from-cyan-500 to-teal-400 hover:shadow-xl'
            }`}
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </button>
          <h3 className="text-lg text-cyan-300">Uploaded Files</h3>
          <ul className="space-y-2">
            {files.map((file, index) => (
              <li
                key={index}
                onClick={() => handleFileClick(file)}
                className="cursor-pointer hover:text-cyan-400"
              >
                {file.file_name || file.name}
              </li>
            ))}
          </ul>
        </div>

        {/* Middle Column: Selected File */}
        <div className="w-full md:w-1/2 p-6">
          <h2 className="text-xl font-semibold text-cyan-300">PDF Viewer</h2>
          {selectedPdfUrl ? (
            <iframe
              src={selectedPdfUrl}
              className="w-full h-[600px]"
              title="PDF Viewer"
              frameBorder="0"
            ></iframe>
          ) : (
            <div className="w-full h-[600px] flex items-center justify-center bg-gray-700 text-gray-400 rounded-lg">
              <p>Select a file to view its content.</p>
            </div>
          )}
        </div>

        {/* Right Column: Conversation Threads */}
        <div className="w-full md:w-1/3 bg-gray-800 p-6 rounded-lg shadow-lg space-y-4">
          <h2 className="text-xl font-semibold text-cyan-300">Chat with AI</h2>
          <div
            ref={chatSectionRef}
            className="h-64 bg-gray-700 rounded-lg overflow-y-auto p-4"
          >
            {threads.map((thread, index) => (
              <div
                key={index}
                className={`p-2 rounded-md mb-2 ${
                  thread.role === 'user' ? 'bg-cyan-500 text-black' : 'bg-gray-600'
                }`}
              >
                <p>{thread.message}</p>
              </div>
            ))}
          </div>
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Ask something..."
            className="w-full p-2 bg-gray-700 rounded-lg text-white"
          />
          <button
            onClick={handleChatInput}
            className="w-full py-2 mt-2 bg-gradient-to-r from-cyan-500 to-teal-400 rounded-lg"
          >
            Send
          </button>
          {errorMessage && <p className="text-red-500 mt-2">{errorMessage}</p>}
        </div>
      </div>
      {isLoading && <div className="text-center text-cyan-300">Loading...</div>}
    </div>
  );
}
