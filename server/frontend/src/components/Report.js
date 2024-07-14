import React from 'react';
import { Link } from 'react-router-dom';

function Report({ onLogout }) {
  return (
    <div className="report-container">
      <nav>
        <Link to="/chat" className="nav-link">Back to Chat</Link>
        <button onClick={onLogout} className="logout-button">Logout</button>
      </nav>
      <h1>Analysis Report</h1>
      <p>This is where the analysis report will be displayed.</p>
      {/* 여기에 리포트 내용을 추가하세요 */}
    </div>
  );
}

export default Report;