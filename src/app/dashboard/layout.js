export default function DashboardLayout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900">
      <main className="flex-1">{children}</main>
    </div>
  );
}
