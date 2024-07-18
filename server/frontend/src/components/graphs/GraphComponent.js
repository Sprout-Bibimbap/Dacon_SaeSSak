import React from 'react';
import "./GraphComponent.css";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const GraphComponent = ({ data, color }) => {
  const maxValue = Math.max(...data.map(item => item.value));
  const yAxisMax = Math.max(4, Math.ceil(maxValue));

  return (
    <div className="graph-container">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis domain={[0, yAxisMax]} />
          <Tooltip />
          <Line type="linear" dataKey="value" stroke={color} strokeWidth={2} dot={{ r: 6 }} activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default GraphComponent;