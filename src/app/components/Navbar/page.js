'use client';

import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_VERCEL_URL || 'http://localhost:3000';

export default function Navbar() {
  const handleLogout = async () => {
    try {
      const response = await axios.post(
        `${BASE_URL}/api/logout`,
        {}, // No data payload required
        { withCredentials: true } // Ensure cookies are sent along with the request
      );
  
      if (response.status === 200) { 
        window.location.href = '/auth/login'; // or router.push('/auth/login') if using Next.js
      }
    } catch (error) {
      console.log('Logout failed:', error.response?.data || error.message);
    }
  };
  

  return (
    <nav className="bg-gradient-to-r from-gray-900 via-gray-800 to-black shadow-lg">
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">
        {/* Logout Button */}
        <div>
          <button
            onClick={handleLogout}
            className="bg-cyan-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105"
          >
            Logout
          </button>
        </div>

        {/* Navigation Buttons */}
        <div className="flex items-center space-x-6">
          <button className="text-white text-lg font-medium hover:text-cyan-400 transition-all duration-300">
            Projects
          </button>
          <div className="relative group">
            <button className="text-white text-lg font-medium hover:text-cyan-400 transition-all duration-300">
              Tools
            </button>
            {/* Dropdown */}
            <div className="absolute hidden group-hover:block bg-gray-800 shadow-lg rounded-lg mt-2 w-40">
              <ul className="text-sm text-white py-2">
                <li className="px-4 py-2 hover:bg-gray-700 rounded-lg cursor-pointer transition-all duration-300">
                  Tool 1
                </li>
                <li className="px-4 py-2 hover:bg-gray-700 rounded-lg cursor-pointer transition-all duration-300">
                  Tool 2
                </li>
              </ul>
            </div>
          </div>
          <button className="text-white text-lg font-medium hover:text-cyan-400 transition-all duration-300">
            Random 1
          </button>
          <button className="text-white text-lg font-medium hover:text-cyan-400 transition-all duration-300">
            Random 2
          </button>
        </div>
      </div>
    </nav>
  );
}
