import React from 'react';
import yellowSiren from '../img/yellow.svg';
import redSiren from '../img/red.svg';
import greenSiren from '../img/green.svg';



const SirenStatus = ({ title, siren, aiText }) => {

  const sirenImages = {
    '위험': redSiren,
    '주의': yellowSiren,
    '안전': greenSiren
  };

  const sirenColorClasses = {
    '위험': 'red',
    '주의': 'yellow',
    '안전': 'green'
  };

  return (
    <div className="component-container">
      <h2 className="component-title">{title}</h2>
      
      <div className="flex justify-between">
        <div className="flex-grow flex justify-center items-center">
          <img 
            src={sirenImages[siren]} 
            alt={`${siren} 상태`} 
            className="w-50 h-50"
          />
        </div>
        
        <div className="flex flex-col items-start space-y-2">
          {Object.entries(sirenColorClasses).map(([key, color]) => (
            <div key={key} className="flex items-center">
              <div className={`w-4 h-4 rounded-full bg-${color}-500 mr-2`}></div>
              <span>{key}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="text-center p-2 bg-gray-100 rounded mt-4">
        <p>{aiText || '데이터를 불러오는 중입니다...'}</p>
      </div>
    </div>
  );
};

export default SirenStatus;