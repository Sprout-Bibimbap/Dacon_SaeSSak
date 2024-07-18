import React from 'react';

const AttachmentRelationship = ({ title, humanText, aiText }) => {
  return (
    <div className="relative border-2 border-gray-300 rounded-lg p-5 pt-6 mb-6">
      <h2 className="absolute top-[1px] -translate-y-[calc(50%+25px)] bg-white px-2 text-2xl font-bold">
        {title}
      </h2>
      <div className="space-y-6 mt-4">
        {/* 사람 말풍선 */}
        <div className="flex items-start">
          <svg className="w-10 h-10 mr-3 flex-shrink-0" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="#4A5568" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="#4A5568" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <div className="bg-blue-100 p-3 rounded-lg rounded-tl-none flex-grow">
            <p>{humanText}</p>
          </div>
        </div>
        
        {/* AI 말풍선 */}
        <div className="flex items-start justify-end">
          <div className="bg-green-100 p-3 rounded-lg rounded-tr-none flex-grow">
            <p>{aiText}</p>
          </div>
          <svg className="w-10 h-10 ml-3 flex-shrink-0" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="2" width="20" height="20" rx="2" stroke="#4A5568" strokeWidth="2"/>
            <circle cx="8" cy="8" r="2" fill="#4A5568"/>
            <circle cx="16" cy="8" r="2" fill="#4A5568"/>
            <path d="M7 16h10" stroke="#4A5568" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 12v4" stroke="#4A5568" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
      </div>
    </div>
  );
};

export default AttachmentRelationship;