import Navbar from '../../../components/Navbar/page';

export default function DashboardLayout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900"> {/* Ensure full height and consistent background */}
      <Navbar />
      <main className="flex-1">{children}</main> {/* Remove unnecessary padding */}
    </div>
  );
}
