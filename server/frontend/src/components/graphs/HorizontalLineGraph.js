import React from 'react';
import './HorizontalLineGraph.css';

const HorizontalLineGraph = ({ data }) => {
  const getBarColor = (value) => {
    switch(value.toLowerCase()) {
      case '상':
        return '#4CAF50';
      case '중':
        return '#FFC107';
      case '하':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };

  return (
    <div className="horizontal-line-graph">
      {data.map((item, index) => (
        <div key={index} className="graph-item">
          <div className="item-label">{item.name}</div>
          <div className="bar-container">
            <div 
              className="bar" 
              style={{
                width: item.value === '상' ? '100%' : (item.value === '중' ? '66%' : '33%'),
                backgroundColor: getBarColor(item.value)
              }}
            ></div>
          </div>
          <div className="item-value">{item.value}</div>
        </div>
      ))}
    </div>
  );
};

export default HorizontalLineGraph;