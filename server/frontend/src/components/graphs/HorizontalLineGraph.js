import React from 'react';

const HorizontalLineGraph = ({ title, data }) => {
  const getPositionForValue = (value) => {
    switch (value) {
      case '하': return 100;
      case '중': return 300;
      case '상': return 500;
      default: return 300;
    }
  };

  // 오른쪽으로 이동할 거리를 정의합니다 (예: 50px)
  const rightShift = 50;

  return (
    <div className="relative border-2 border-gray-300 rounded-lg p-2 pt-6 mb-4">
      <h2 className="absolute top-[1px] -translate-y-[calc(50%+25px)] bg-white px-2 text-2xl font-bold">
        {title}
      </h2>
      <svg viewBox="0 0 650 150" preserveAspectRatio="xMidYMid meet" className="w-full h-auto mt-2">
        {['하', '중', '상'].map((label, index) => (
          <g key={label}>
            <text x={100 + rightShift + index * 200} y="20" textAnchor="middle" fontSize="14" fill="#666">{label}</text>
            <line
              x1={100 + rightShift + index * 200}
              y1="30"
              x2={100 + rightShift + index * 200}
              y2={30 + data.length * 40}
              stroke="#ccc"
              strokeWidth="1"
              strokeDasharray="5,5"
            />
          </g>
        ))}

        {data.map((item, index) => (
          <g key={item.name}>
            <line
              x1={100 + rightShift}
              y1={40 + index * 40}
              x2={500 + rightShift}
              y2={40 + index * 40}
              stroke="#e0e0e0"
              strokeWidth="2"
            />
            <circle
              cx={getPositionForValue(item.value) + rightShift}
              cy={40 + index * 40}
              r="6"
              fill="green"
            />
            <text
              x={80 + rightShift}
              y={45 + index * 40}
              textAnchor="end"
              fontSize="16"
              fontWeight="bold"
              fill="#333"
            >
              {item.name}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default HorizontalLineGraph;