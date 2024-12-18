'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

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
      } focus:outline-none focus:ring-2 ${
        error ? 'focus:ring-red-500' : 'focus:ring-cyan-500'
      }`}
      placeholder={placeholder}
    />
    {error && <p className="text-red-500 text-sm">{error}</p>}
  </div>
);

export default function SignUp() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
  });
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

    if (!formData.name.trim()) {
      newErrors.name = 'Full name is required';
    } else if (formData.name.trim().length < 3) {
      newErrors.name = 'Full name must be at least 3 characters long';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Invalid email format';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters long';
    } else if (!/[A-Z]/.test(formData.password)) {
      newErrors.password = 'Password must contain at least one uppercase letter';
    } else if (!/[0-9]/.test(formData.password)) {
      newErrors.password = 'Password must contain at least one number';
    } else if (!/[!@#$%^&*(),.?":{}|<>]/.test(formData.password)) {
      newErrors.password = 'Password must contain at least one special character';
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
      const response = await fetch(`${BASE_URL}/api/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data.message);
        router.push('/dashboard'); // Redirect to the login page
      } else {
        const errorData = await response.json();
        setApiError(errorData.detail || 'An error occurred. Please try again.');
      }
    } catch (error) {
      setApiError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogInClick = () => router.push('/auth/login');

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      {/* Sign-Up Form Section */}
      <section className="flex-1 py-12 px-6 bg-gradient-to-r from-gray-800 via-black to-gray-900">
        <div className="max-w-lg mx-auto bg-gray-700 p-8 rounded-xl shadow-lg">
          <h2 className="text-3xl sm:text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 text-center mb-8">
            Create Your Account
          </h2>

          <form onSubmit={handleSubmit}>
            <InputField
              label="Full Name"
              type="text"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              error={errors.name}
              placeholder="Enter your full name"
            />
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
            {apiError && (
              <p className="text-red-500 text-sm text-center mb-4">
                {typeof apiError === 'string' ? apiError : apiError.msg || 'An error occurred.'}
              </p>
            )}
            <button
              type="submit"
              disabled={loading}
              className={`w-full px-8 py-4 text-lg sm:text-xl font-medium text-black bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full shadow-lg hover:shadow-2xl focus:outline-none transition duration-300 ${
                loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? 'Creating Account...' : 'Sign Up'}
            </button>
          </form>
        </div>
      </section>

      {/* Footer Section */}
      <footer className="bg-gradient-to-r from-gray-800 via-black to-gray-900 py-8 text-center text-white">
        <p className="text-lg">
          Already have an account?{' '}
          <button
            onClick={handleLogInClick}
            className="text-cyan-300 hover:text-cyan-400 font-medium underline"
          >
            Log In
          </button>
        </p>
      </footer>
    </div>
  );
}
