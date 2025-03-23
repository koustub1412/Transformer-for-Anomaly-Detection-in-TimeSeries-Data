// import React, { useState } from 'react';
// import axios from 'axios';
// import { Line } from 'react-chartjs-2';
// import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js';

// ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement);

// const FileUpload = () => {
//   const [uploadedFileName, setUploadedFileName] = useState('');
//   const [anomalies, setAnomalies] = useState([]);
//   const [dataset, setDataset] = useState([]);
//   const [error, setError] = useState(null);
//   const [highlightedAnomaly, setHighlightedAnomaly] = useState(null);
//   const [losses, setLosses] = useState([]);
//   const [combinedScores, setCombinedScores] = useState([]);
//   const [threshold, setThreshold] = useState(0);
//   const anomaliesPerPage = 40;
//   const [currentPage, setCurrentPage] = useState(1);

//   // Handle file upload
//   const handleFileUpload = async (event) => {
//     const file = event.target.files ? event.target.files[0] : null;
  
//     if (!file) {
//       setError('Please select a file before uploading.');
//       return;
//     }
  
//     const formData = new FormData();
//     formData.append('file', file);
  
//     try {
//       const response = await axios.post('http://localhost:5000/api/upload', formData, {
//         headers: { 'Content-Type': 'multipart/form-data' },
//       });
  
//       console.log('Upload response:', response.data); 
  
//       if (response.data.anomalies) {
//         const dataset = response.data.losses;
//         setDataset(Array.isArray(dataset) ? dataset : []);  // Ensure dataset is an array
//         setAnomalies(response.data.anomalies);
//         setLosses(Array.isArray(response.data.losses) ? response.data.losses : []); // Ensure losses is an array
//         setCombinedScores(response.data.combined_scores);
//         setThreshold(response.data.threshold);
//         setUploadedFileName(file.name);
//         setError(null);
//       } else {
//         throw new Error('Invalid response from server. Anomalies data is missing.');
//       }
//     } catch (uploadError) {
//       console.error('Error uploading file:', uploadError);
//       setError('Error uploading file. Please try again.');
//     }
//   };

//   // Get the anomalies for the current page
//   const indexOfLastAnomaly = currentPage * anomaliesPerPage;
//   const indexOfFirstAnomaly = indexOfLastAnomaly - anomaliesPerPage;
//   const currentAnomalies = anomalies.slice(indexOfFirstAnomaly, indexOfLastAnomaly);

//   // Prepare chart data
//   const chartData = {
//     labels: Array.isArray(dataset) && dataset.length > 0 ? dataset.map((_, index) => index) : [],
//     datasets: [
//       {
//         label: 'Dataset',
//         data: dataset,
//         fill: false,
//         borderColor: 'rgba(75,192,192,1)',
//         pointBackgroundColor: 'rgba(75,192,192,1)',
//         pointRadius: 3,
//         borderWidth: 1,
//       },
//       {
//         label: 'Anomalies',
//         data: Array.isArray(dataset) ? dataset.map((data, index) =>
//           anomalies.includes(index) ? data : NaN
//         ) : [],
//         backgroundColor: 'red',
//         borderColor: 'red',
//         pointRadius: 6,
//         borderWidth: 2,
//         fill: false,
//         tension: 0,
//       },
//       {
//         label: 'Highlighted Anomaly',
//         data: highlightedAnomaly !== null ? [
//           { x: highlightedAnomaly, y: dataset[highlightedAnomaly] }
//         ] : [],
//         pointBackgroundColor: 'purple',
//         pointRadius: 10,
//         borderColor: 'purple',
//         fill: false,
//         tension: 0,
//       },
//     ],
//   };

//   const handleAnomalyClick = (index) => {
//     setHighlightedAnomaly(index);
//   };

//   const chartOptions = {
//     responsive: true,
//     maintainAspectRatio: false,
//     scales: {
//       x: {
//         type: 'linear',
//         ticks: {
//           maxRotation: 0,
//           autoSkip: true,
//         },
//       },
//       y: {
//         ticks: {
//           beginAtZero: false,
//         },
//       },
//     },
//   };

//   const handlePageChange = (newPage) => {
//     if (newPage > 0 && newPage <= Math.ceil(anomalies.length / anomaliesPerPage)) {
//       setCurrentPage(newPage);
//     }
//   };

//   return (
//     <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
//       <h2>Anomaly Detection</h2>

//       <div style={{ marginBottom: '15px' }}>
//         <input
//           type="file"
//           onChange={handleFileUpload}
//           style={{ marginRight: '10px' }}
//         />
//         <button onClick={handleFileUpload}>Upload</button>
//       </div>

//       {uploadedFileName && <p><strong>Uploaded File:</strong> {uploadedFileName}</p>}
//       {Array.isArray(losses) && losses.length > 0 && <p><strong>Losses:</strong> {losses.join(', ')}</p>}
//       {/* {threshold && <p><strong>Threshold:</strong> {threshold}</p>}
//       {Array.isArray(combinedScores) && combinedScores.length > 0 && <p><strong>Combined Scores:</strong> {combinedScores.join(', ')}</p>} */}

//       {error && <p style={{ color: 'red' }}>{error}</p>}

//       {Array.isArray(dataset) && dataset.length > 0 && (
//         <div style={{ width: '100%', height: '600px' }}>
//           <Line data={chartData} options={chartOptions} />
//         </div>
//       )}

//       {anomalies.length > 0 && (
//         <div>
//           <h3>Anomalies Detected</h3>

//           <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
//             {currentAnomalies.map((anomaly, index) => (
//               <button
//                 key={index}
//                 onClick={() => handleAnomalyClick(anomaly)}
//                 style={{
//                   textAlign: 'center',
//                   padding: '5px',
//                   border: '1px solid #ccc',
//                   backgroundColor: highlightedAnomaly === anomaly ? 'yellow' : 'white',
//                   cursor: 'pointer',
//                   fontWeight: highlightedAnomaly === anomaly ? 'bold' : 'normal',
//                 }}
//               >
//                 Index: {anomaly}
//               </button>
//             ))}
//           </div>

//           <div>
//             <button
//               onClick={() => handlePageChange(currentPage - 1)}
//               disabled={currentPage === 1}
//             >
//               Previous
//             </button>
//             <span>{` Page ${currentPage} of ${Math.ceil(anomalies.length / anomaliesPerPage)} `}</span>
//             <button
//               onClick={() => handlePageChange(currentPage + 1)}
//               disabled={currentPage === Math.ceil(anomalies.length / anomaliesPerPage)}
//             >
//               Next
//             </button>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default FileUpload;
import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js';

// Register chart.js components
ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement);

const FileUpload = () => {
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [anomalies, setAnomalies] = useState([]);
  const [dataset, setDataset] = useState([]);
  const [error, setError] = useState(null);
  const [highlightedAnomaly, setHighlightedAnomaly] = useState(null);
  const anomaliesPerPage = 40;
  const [currentPage, setCurrentPage] = useState(1);

  // Handle file upload
  const handleFileUpload = async (file) => {
    if (!file) {
      setError('Please select a file before uploading.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('Upload response:', response.data);

      if (response.data.anomalies) {
        const dataset = response.data.combined_scores; // Assuming combined_scores is the dataset
        setDataset(Array.isArray(dataset) ? dataset : []); // Ensure dataset is an array
        setAnomalies(response.data.anomalies);
        setUploadedFileName(file.name);
        setError(null);
      } else {
        throw new Error('Invalid response from server. Anomalies data is missing.');
      }
    } catch (uploadError) {
      console.error('Error uploading file:', uploadError);
      setError('Error uploading file. Please try again.');
    }
  };

  // Get the anomalies for the current page
  const indexOfLastAnomaly = currentPage * anomaliesPerPage;
  const indexOfFirstAnomaly = indexOfLastAnomaly - anomaliesPerPage;
  const currentAnomalies = anomalies.slice(indexOfFirstAnomaly, indexOfLastAnomaly);

  // Prepare chart data
  const chartData = {
    labels: dataset.length > 0 ? dataset.map((_, index) => index) : [],
    datasets: [
      {
        label: 'Dataset',
        data: dataset,
        fill: false,
        borderColor: 'rgba(75,192,192,1)',
        pointBackgroundColor: 'rgba(75,192,192,1)',
        pointRadius: 3,
        borderWidth: 1,
      },
      {
        label: 'Anomalies',
        data: dataset.map((data, index) =>
          anomalies.includes(index) ? data : NaN
        ),
        backgroundColor: 'red',
        borderColor: 'red',
        pointRadius: 6,
        borderWidth: 2,
        fill: false,
        tension: 0,
      },
      {
        label: 'Highlighted Anomaly',
        data: highlightedAnomaly !== null ? [
          { x: highlightedAnomaly, y: dataset[highlightedAnomaly] }
        ] : [],
        pointBackgroundColor: 'purple',
        pointRadius: 10,
        borderColor: 'purple',
        fill: false,
        tension: 0,
      },
    ],
  };

  const handleAnomalyClick = (index) => {
    setHighlightedAnomaly(index);
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        min: 0, // Set to the minimum index (can be Math.min if needed for non-zero start)
        max: dataset.length +5, // Set to the maximum index dynamically
        ticks: {
          stepSize: Math.ceil(dataset.length / 10), // Dynamically calculate step size for the x-axis
          autoSkip: true, // Skip labels if there are too many
          maxRotation: 0, // Keep labels horizontal
        },
      },
      y: {
        min: Math.min(...dataset)-0.1 , // Dynamically set just below the minimum value
        max: Math.max(...dataset) + 0.5, // Add some padding above the maximum value
        ticks: {
        stepSize: 1, // Set step size to 1 to avoid decimal points
        autoSkip: true, // Skip labels if there are too many
        maxRotation: 0, // Keep labels horizontal
      },
      },
    },
  };
  

  // Handle page change
  const handlePageChange = (newPage) => {
    if (newPage > 0 && newPage <= Math.ceil(anomalies.length / anomaliesPerPage)) {
      setCurrentPage(newPage);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Anomaly Detection</h2>

      <div style={{ marginBottom: '15px' }}>
        <input
          type="file"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          style={{ marginRight: '10px' }}
        />
        <button onClick={() => handleFileUpload(document.querySelector('input[type="file"]').files[0])}>
          Upload
        </button>
      </div>

      {uploadedFileName && <p><strong>Uploaded File:</strong> {uploadedFileName}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Display dataset in a chart */}
      {dataset.length > 0 && (
        <div style={{ width: '100%', height: '600px' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      {/* Anomalies and pagination */}
      {anomalies.length > 0 && (
        <div>
          <h3>Anomalies Detected</h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
            {currentAnomalies.map((anomaly, index) => (
              <button
                key={index}
                onClick={() => handleAnomalyClick(anomaly)}
                style={{
                  textAlign: 'center',
                  padding: '5px',
                  border: '1px solid #ccc',
                  backgroundColor: highlightedAnomaly === anomaly ? 'yellow' : 'white',
                  cursor: 'pointer',
                  fontWeight: highlightedAnomaly === anomaly ? 'bold' : 'normal',
                }}
              >
                Index: {anomaly}
              </button>
            ))}
          </div>

          <div>
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            <span>{` Page ${currentPage} of ${Math.ceil(anomalies.length / anomaliesPerPage)} `}</span>
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === Math.ceil(anomalies.length / anomaliesPerPage)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
