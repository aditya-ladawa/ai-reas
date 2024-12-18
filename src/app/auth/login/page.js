'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_VERCEL_URL || 'http://localhost:3000';

const InputField = ({ label, type, name, value, onChange, error, placeholder }) => (
  <div className="mb-4">
    <label className="block text-white text-lg mb-2" htmlFor={name}>
      {label}
    </label>
    <input
      type={type}
      name={name}
      id={name}
      value={value}
      onChange={onChange}
      className={`w-full px-4 py-2 text-lg text-white bg-gray-600 rounded-lg border ${
        error ? 'border-red-500' : 'border-gray-500'
      } focus:outline-none focus:ring-2 ${error ? 'focus:ring-red-500' : 'focus:ring-cyan-500'}`}
      placeholder={placeholder}
    />
    {error && <p className="text-red-500 text-sm">{error}</p>}
  </div>
);

export default function LogIn() {
  const router = useRouter();
  const [formData, setFormData] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    if (errors[name]) {
      setErrors({ ...errors, [name]: null }); // Clear the error for the field
    }
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Invalid email format';
    }
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;
  
    setLoading(true);
    setApiError('');
  
    try {
      const response = await axios.post(
        `${BASE_URL}/api/login`,
        formData,
        {
          withCredentials: true, // Ensure cookies are sent and received
        }
      );
  
      router.push('/dashboard'); // Redirect to the dashboard after successful login
    } catch (error) {
      if (error.response) {
        console.log('API Error:', error.response.data); // Log API error response
        setApiError(error.response.data.detail || 'An error occurred. Please try again.');
      } else {
        console.log('Axios Error:', error.message); // Log actual Axios error
        setApiError('Something went wrong. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  
  const handleSignUpClick = () => router.push('/auth/signup');

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      <section className="flex-1 py-12 px-6 bg-gradient-to-r from-gray-800 via-black to-gray-900">
        <div className="max-w-lg mx-auto bg-gray-700 p-8 rounded-xl shadow-lg">
          <h2 className="text-3xl sm:text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 text-center mb-8">
            Welcome Back
          </h2>

          <form onSubmit={handleSubmit}>
            <InputField
              label="Email Address"
              type="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              error={errors.email}
              placeholder="Enter your email address"
            />
            <InputField
              label="Password"
              type="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              error={errors.password}
              placeholder="Enter your password"
            />
            {apiError && <p className="text-red-500 text-sm text-center mb-4">{apiError}</p>}
            <button
              type="submit"
              disabled={loading}
              className={`w-full px-8 py-4 text-lg sm:text-xl font-medium text-black bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full shadow-lg hover:shadow-2xl focus:outline-none transition duration-300 ${
                loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? 'Logging In...' : 'Log In'}
            </button>
          </form>
        </div>
      </section>

      <footer className="bg-gradient-to-r from-gray-800 via-black to-gray-900 py-8 text-center text-white">
        <p className="text-lg">
          Don't have an account yet?{' '}
          <button
            onClick={handleSignUpClick}
            className="text-cyan-300 hover:text-cyan-400 font-medium underline"
          >
            Sign Up
          </button>
        </p>
      </footer>
    </div>
  );
}
