import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="flex flex-col h-screen">
      <header className="flex justify-between items-center p-4 bg-green-500 text-white">
        <h1 className="text-2xl font-bold">새싹비빔밥</h1>
        <div>
          <Link to="/login" className="mr-4 px-4 py-2 bg-white text-green-500 rounded">Log in</Link>
          <Link to="/signUp" className="px-4 py-2 bg-white text-green-500 rounded">Sign up</Link>
        </div>
      </header>
      <main className="flex-grow flex items-center justify-center">
        <h2 className="text-4xl font-bold text-green-600">Welcome to 새싹비빔밥</h2>
      </main>
    </div>
  );
}

export default Home;