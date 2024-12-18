'use client'
import { useRouter } from 'next/navigation';
import Image from 'next/image';


export default function Home() {
  const router = useRouter(); // Initialize the router

  const handleGetStartedClick = () => {
    router.push('/auth/signup'); // Navigate to the signup page
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <section className="relative w-full h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black flex items-center justify-center text-center px-4 py-8 border-b border-gray-700">
        <div className="absolute top-0 left-0 w-full h-full">
          <Image
            src="/aireas_hero.jpg"
            alt="AIREAS Logo"
            layout="fill"
            objectFit="cover"
            className="opacity-50"
          />
        </div>
        <div className="relative z-10 w-full text-center px-4">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 mb-6">
            AIREAS
          </h1>
          <p className="text-lg sm:text-xl text-white max-w-3xl mx-auto mb-8 leading-relaxed">
            A cutting-edge AI research assistant designed to revolutionize the way we work with data, research, and technology. Unlock the future with AI-powered research capabilities.
          </p>
          <button
            className="px-8 py-4 text-lg sm:text-xl font-medium text-black bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full shadow-lg hover:shadow-2xl focus:outline-none transition duration-300"
            onClick={handleGetStartedClick} // Add onClick event handler
          >
            Get Started
          </button>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-gradient-to-r from-gray-800 via-black to-gray-900 border-b border-gray-700">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 mb-12">
            Features
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-12">
            <div className="bg-gray-700 p-8 rounded-xl shadow-lg hover:shadow-xl transition duration-300">
              <h3 className="text-2xl text-white mb-4">AI-Driven Insights</h3>
              <p className="text-gray-300">
                Harness the power of artificial intelligence to gain advanced insights and research summaries.
              </p>
            </div>
            <div className="bg-gray-700 p-8 rounded-xl shadow-lg hover:shadow-xl transition duration-300">
              <h3 className="text-2xl text-white mb-4">Advanced Search</h3>
              <p className="text-gray-300">
                Quickly search through academic papers, articles, and documents to find the information you need.
              </p>
            </div>
            <div className="bg-gray-700 p-8 rounded-xl shadow-lg hover:shadow-xl transition duration-300">
              <h3 className="text-2xl text-white mb-4">Collaborative Tools</h3>
              <p className="text-gray-300">
                Collaborate effortlessly with team members by sharing research, notes, and summaries in real-time.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20 px-6 bg-gradient-to-r from-gray-800 via-black to-gray-900 border-b border-gray-700">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 mb-10">
            About AIREAS
          </h2>
          <p className="text-lg sm:text-xl text-white mb-8 leading-relaxed">
            AIREAS is an advanced AI-powered research assistant designed to simplify the complexities of academic and scientific research. 
            It leverages the latest in AI technology to bring intelligent solutions to your fingertips.
          </p>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-20 px-6 bg-gradient-to-r from-gray-800 via-black to-gray-900">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-light text-transparent bg-clip-text bg-gradient-to-r from-white to-cyan-300 mb-10">
            Contact Us
          </h2>
          <p className="text-lg sm:text-xl text-white mb-8 leading-relaxed">
            Have any questions or need support? Reach out to us, and we’ll be happy to assist you.
          </p>
          <button className="px-8 py-4 text-lg sm:text-xl font-medium text-black bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full shadow-lg hover:shadow-2xl focus:outline-none transition duration-300">
            Contact Support
          </button>
        </div>
      </section>
    </div>
  );
}
