import React from 'react';
import { Link } from 'react-router-dom';

function MainPage({ onLogout }) {
  return (
    <div className="flex flex-col h-screen">
      <header className="flex justify-between items-center p-4 bg-green-500 text-white">
        <h1 className="text-2xl font-bold">새싹비빔밥</h1>
        <button onClick={onLogout} className="px-4 py-2 bg-white text-green-500 rounded">Logout</button>
      </header>
      <main className="flex-grow flex items-center justify-center">
        <div className="space-y-4">
          <Link to="/chat" className="block w-64 px-4 py-2 bg-green-500 text-white text-center rounded">Chat</Link>
          <Link to="/report" className="block w-64 px-4 py-2 bg-green-500 text-white text-center rounded">Report</Link>
          <button className="w-64 px-4 py-2 bg-green-500 text-white rounded">TextLog</button>
        </div>
      </main>
    </div>
  );
}

export default MainPage;