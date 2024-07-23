import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Title from './graphs/Title';
import BrainMap from './graphs/BrainMap';
import GraphComponent from './graphs/GraphComponent';
import HorizontalLineGraph from './graphs/HorizontalLineGraph';
import AttachmentRelationship from './graphs/AttachmentRelationship';
import Emergency from './graphs/Emergency';
import "../index.css";
import "./Report.css";

function Report({ userName = "새싹" }) {
  const navigate = useNavigate();
  // 말하기 성장표(API)
  const graphData = [
    { name: '23.10.08', value: 1.6 },
    { name: '23.11.12', value: 1.7 },
    { name: '23.12.10', value: 1.8 },
    { name: '24.01.12', value: 2.0 },
    { name: '24.03.20', value: 2.2 },
    { name: '24.04.15', value: 2.3 },
    { name: '24.05.14', value: 2.7 },
    { name: '24.06.11', value: 2.5 },
    { name: '24.08.20', value: 2.6 },
    { name: '24.10.12', value: 2.8 },
    { name: '24.11.15', value: 3.2 },
  ];

  // 창의·잠재력지수(API)
  const horizontalGraphData1 = [
    { name: '집중력', value: '상' },
    { name: '어휘력', value: '상' },
    { name: '논리력', value: '하' },
  ];

  // 마음성장지수(API)
  const horizontalGraphData2 = [
    { name: '공감능력', value: '중' },
    { name: '자기이해능력', value: '상' },
  ];

  // 현재 날짜
  const currentDate = new Date();
  const year = currentDate.getFullYear();
  const month = currentDate.getMonth() + 1;  
  const day = currentDate.getDate();

  // 애착 관계(API)
  const [humanText, setHumanText] = useState('');
  const [aiText, setAiText] = useState('');

  // 관심 주제(API) 
  const brainAreas = [
    { id: 1, x: 30, y: 18 }, 
    { id: 2, x: 55, y: 12 }, 
    { id: 3, x: 25, y: 30 },  
    { id: 4, x: 46, y: 30 }, 
    { id: 5, x: 72, y: 33 }, 
    { id: 6, x: 48, y: 45 }, 
    { id: 7, x: 67, y: 50 }, 
  ];
  const brainFonts = ['14px', '12px', '16px', '20px', '15px', '12px', '14px']
  const userBrainAreas = brainAreas;
  const otherUsersBrainAreas = brainAreas;
  const [userBrainData, setUserBrainData] = useState([]);
  const [otherUsersBrainData, setOtherUsersBrainData] = useState([]);

  // 발달지연 경고(API)
  const [emergencyData, setEmergencyData] = useState({
    title: "",
    siren: "",
    aiText: ""
  });

  useEffect(() => {
    const fetchAttachmentData = async () => {
      const humanTextResponse = new Promise((resolve) => {
        setTimeout(() => {
          resolve("엄마 아빠 좋아. 아빠랑 동생이랑 놀이터에서 같이 놀고 싶어. 엄마 아빠가 바빠. 새싹이랑 더 많이 놀아줬으면 좋겠어.");
        }, 1000);
      });

      const aiTextResponse = new Promise((resolve) => {
        setTimeout(() => {
          resolve("새싹이는 부모와의 애착 수준이 높지만 더 높은 수준의 사랑을 원하고 있습니다.");
        }, 1000);
      });

      const [humanTextData, aiTextData] = await Promise.all([humanTextResponse, aiTextResponse]);
      setHumanText(humanTextData);
      setAiText(aiTextData);
    };

    const fetchBrainData = async () => {
      const userResponse = await new Promise((resolve) => {
        setTimeout(() => {
          resolve(["사랑", "아빠", "엄마", "소방차", "밥", "타요", "도티"]);
        }, 1000);
      });
      setUserBrainData(userResponse);

      const otherUsersResponse = await new Promise((resolve) => {
        setTimeout(() => {
          resolve(["가족", "놀이", "친구", "음식", "동물", "자동차", "책"]);
        }, 1000);
      });
      setOtherUsersBrainData(otherUsersResponse);
    };

    const fetchEmergencyData = async () => {
      const response = await new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            title: "발달지연 경고",
            siren: "위험", 
            aiText: "현재 발달 지연 위험이 감지되었습니다. 가까운 병원 방문이나 전문가 상담을 고려해보세요."
          });
        }, 1000);
      });
      setEmergencyData(response);
    };

    fetchAttachmentData();
    fetchBrainData();
    fetchEmergencyData();
  }, []);

  const handleBackToMain = () => {
    navigate('/'); 
  };

  return (
    <div className="report-container max-w-4xl mx-auto p-4">
      <nav className="report-nav">
        <button onClick={handleBackToMain} className="back-button">Back</button>
        <Link to="/chat" className="nav-link">Back to Chat</Link>
      </nav>

      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">
          {year}년 {month}월 {day}일 {userName}이의 성장 보고서
        </h1>
        <div className="flex space-x-4">
          <button className="p-2 hover:bg-gray-100 rounded-full" title="공유하기">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 48 48">
              <path stroke="#1E1E1E" strokeLinecap="round" strokeLinejoin="round" strokeWidth="4" d="m17.18 27.02 13.66 7.96m-.02-21.96-13.64 7.96M42 10a6 6 0 1 1-12 0 6 6 0 0 1 12 0ZM18 24a6 6 0 1 1-12 0 6 6 0 0 1 12 0Zm24 14a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z"/>
            </svg>
          </button>
          <button className="p-2 hover:bg-gray-100 rounded-full" title="PDF 내보내기">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 48 48">
              <path stroke="#1E1E1E" strokeLinecap="round" strokeLinejoin="round" strokeWidth="4" d="M42 30v8a4 4 0 0 1-4 4H10a4 4 0 0 1-4-4v-8m8-10 10 10m0 0 10-10M24 30V6"/>
            </svg>
          </button>
        </div>
      </div>


      
      <div className="space-y-8">
        <div>
            <Title>말하기 성장표</Title>
            <GraphComponent 
              data={graphData} 
              color="#4CAF50"
            />
        </div>

        <div>
          <Title>창의·잠재력지수</Title>
          <HorizontalLineGraph 
            data={horizontalGraphData1}
          />
        </div>

        <div>
          <Title>마음성장지수</Title>
          <HorizontalLineGraph 
            data={horizontalGraphData2}
          />
        </div>

        <div>
          <Title>애착 관계</Title>
          <AttachmentRelationship
            humanText={humanText}
            aiText={aiText}
          />
        </div>

        <div>
          <Title>관심 주제</Title>
          <BrainMap
            userBrainAreas={userBrainAreas}
            otherUsersBrainAreas={otherUsersBrainAreas}
            userBrainData={userBrainData}
            otherUsersBrainData={otherUsersBrainData}
            fontSizeAdjusts={brainFonts}
            userName={userName}
          />
        </div>

        <div>
          <Title>발달지연 경고</Title>
          <Emergency
            siren={emergencyData.siren}
            aiText={emergencyData.aiText}
          />
        </div>
      </div>
    </div>
  );
}

export default Report;