import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const GraphComponent = ({ title, data, color }) => {
  const maxValue = Math.max(...data.map(item => item.value));
  const yAxisMax = Math.max(4, Math.ceil(maxValue));

  return (
    <div className="relative border-2 border-gray-300 rounded-lg p-4 pt-8 mb-4">
      <h2 className="absolute top-[1px] -translate-y-[calc(50%+25px)] bg-white px-2 text-2xl font-bold">
        {title}
      </h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
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