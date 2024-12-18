'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation'; // Import useRouter for App Router

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3000';

export default function Dashboard() {
  const [projects, setProjects] = useState([]);
  const [newProject, setNewProject] = useState({ conversation_name: '', conversation_description: '' });
  const [loading, setLoading] = useState(false); // For loading state
  const [error, setError] = useState(''); // For error messages
  const router = useRouter(); // Initialize useRouter

  // Fetch all conversations
  const fetchConversations = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${BASE_URL}/api/fetch_conversations`);
      if (response.status === 200) {
        setProjects(response.data.conversations || []); // Ensure conversations exist
      }
    } catch (error) {
      console.error('Error fetching conversations:', error);
      setError('Failed to load conversations.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch conversations on component mount
  useEffect(() => {
    fetchConversations();
  }, []);

  // Add a new project (conversation)
  const addConversation = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await axios.post(`${BASE_URL}/api/add_conversation`, newProject, {
        headers: { 'Content-Type': 'application/json' },
      });
      console.log('Conversation added:', response.data);
      setNewProject({ conversation_name: '', conversation_description: '' }); // Clear the form
      fetchConversations(); // Refresh the project list
    } catch (error) {
      console.error('Error adding conversation:', error);
      setError('Failed to add conversation.');
    } finally {
      setLoading(false);
    }
  };

  // Navigate to conversation view
  const viewConversation = async (conversationId) => {
    try {
      setLoading(true);
      const response = await axios.get(`${BASE_URL}/api/conversations/${conversationId}`);
      if (response.status === 200) {
        router.push(`/dashboard/conversations/${conversationId}`); // Navigate to dynamic route
      }
    } catch (error) {
      console.error('Error fetching conversation:', error);
      setError('Failed to load conversation.');
    } finally {
      setLoading(false);
    }
  };

  // Format date without seconds
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return ''; // Handle invalid date
    return date.toLocaleString([], {
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      {/* Page Title */}
      <h1 className="text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 py-6 text-center">
        Projects
      </h1>

      {/* Error Display */}
      {error && <p className="text-center text-red-500">{error}</p>}

      {/* Form Section */}
      <div className="h-[30%] flex items-center justify-center px-6">
        <div className="w-full max-w-4xl bg-gray-800 p-6 rounded-lg shadow-lg hover:shadow-xl transition duration-300">
          <h2 className="text-2xl text-white mb-6 text-center">Add a New Project</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              addConversation();
            }}
            className="flex items-center space-x-4"
          >
            <input
              type="text"
              placeholder="Project Name"
              value={newProject.conversation_name}
              onChange={(e) =>
                setNewProject({ ...newProject, conversation_name: e.target.value })
              }
              className="flex-grow p-3 bg-gray-700 text-white rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-cyan-500"
              required
            />
            <input
              type="text"
              placeholder="Description"
              value={newProject.conversation_description}
              onChange={(e) =>
                setNewProject({ ...newProject, conversation_description: e.target.value })
              }
              className="flex-grow p-3 bg-gray-700 text-white rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-cyan-500"
              required
            />
            <button
              type="submit"
              className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-teal-400 text-black font-medium rounded-lg shadow-lg hover:shadow-xl transition duration-300"
              disabled={loading}
            >
              {loading ? 'Submitting...' : 'Submit'}
            </button>
          </form>
        </div>
      </div>

      {/* Projects Section */}
      <div className="h-[70%] overflow-y-auto px-6 py-8">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl text-white mb-6 text-center">Your Projects</h2>
          {loading && <p className="text-center text-gray-400">Loading...</p>}
          {projects.length === 0 ? (
            <p className="text-center text-gray-400">No projects available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full bg-gray-800 text-white">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="px-4 py-2 text-left">Name</th>
                    <th className="px-4 py-2 text-left">Topic</th>
                    <th className="px-4 py-2 text-left">Date</th>
                    <th className="px-4 py-2 text-left">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {projects.map((project, index) => (
                    <tr key={index} className="border-b border-gray-600">
                      <td className="px-4 py-2">{project.name}</td>
                      <td className="px-4 py-2">{project.topic}</td>
                      <td className="px-4 py-2">{formatDate(project.created_at)}</td>
                      <td className="px-4 py-2">
                        <button
                          type="button"
                          className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-teal-400 text-black rounded-lg"
                          onClick={() => viewConversation(project.id)}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
