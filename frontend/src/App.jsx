import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { UploadCloud, Activity, Database, Cpu, ChevronRight, Loader2, History } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState('LSTM');
  const [targetCol, setTargetCol] = useState('Close');
  const [dateCol, setDateCol] = useState('Date');
  const [featureCols, setFeatureCols] = useState('Open,High,Low,Volume');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleRunExperiment = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Mohon upload file CSV terlebih dahulu.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('target_col', targetCol);
    formData.append('date_col', dateCol);
    
    if (modelType === 'MLR') {
      formData.append('feature_cols', featureCols);
    }

    try {
      // Mengirim data ke Mesin Python (FastAPI) di Port 8000
      const response = await axios.post('http://127.0.0.1:8000/api/experiment', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const resData = response.data;
      
      // Formatting Data untuk Grafik Recharts
      const chartData = resData.chart_data.dates.map((date, index) => ({
        date: date,
        Actual: resData.chart_data.actual[index],
        Predicted: resData.chart_data.predicted[index],
      }));
      
      const resultObj = {
        id: Date.now(),
        fileName: file.name,
        model: resData.model_type,
        mape: resData.metrics.MAPE,
        rmse: resData.metrics.RMSE,
        mae: resData.metrics.MAE,
        chartData: chartData
      };
      
      setCurrentResult(resultObj);
      setHistory(prev => [resultObj, ...prev]);
      
    } catch (err) {
      setError(err.response?.data?.error || "Terjadi kesalahan sistem saat melatih model.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Forecasting Experiment Platform</h1>
        <p>Advanced Machine Learning Web Interface</p>
      </header>

      {error && (
        <div style={{background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444', padding: '1rem', marginBottom: '2rem', borderRadius: '4px', color: '#fca5a5'}}>
          <strong>Error: </strong> {error}
        </div>
      )}

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem'}}>
        {/* PANEL KIRI: KONFIGURASI */}
        <div>
          <div className="glass-card">
            <h2 className="card-title">
              <Database size={24} color="#3b82f6" /> 1. Upload Dataset
            </h2>
            <div className="file-upload-wrapper">
              <input type="file" accept=".csv" onChange={handleFileChange} />
              <div className="file-upload-btn">
                <UploadCloud size={24} />
                {file ? file.name : 'Pilih File CSV Dataset Anda'}
              </div>
            </div>
          </div>

          <div className="glass-card">
            <h2 className="card-title">
              <Cpu size={24} color="#a78bfa" /> 2. Parameter Model
            </h2>
            <form onSubmit={handleRunExperiment} className="form-group">
              
              <div className="form-group" style={{marginBottom: '1rem'}}>
                <label>Algoritma Machine Learning</label>
                <select className="form-control" value={modelType} onChange={e => setModelType(e.target.value)}>
                  <option value="LSTM">Deep Learning (LSTM)</option>
                  <option value="ARIMA">Statistik (ARIMA)</option>
                  <option value="Holt-Winters">Pemulusan (Holt-Winters)</option>
                  <option value="Prophet">Additive (Meta Prophet)</option>
                  <option value="MLR">Multivariate (Regresi Linear Berganda)</option>
                </select>
              </div>

              <div className="form-grid" style={{marginBottom: '1rem'}}>
                <div className="form-group">
                  <label>Kolom Target (Y)</label>
                  <input type="text" className="form-control" value={targetCol} onChange={e => setTargetCol(e.target.value)} />
                </div>
                <div className="form-group">
                  <label>Kolom Tanggal</label>
                  <input type="text" className="form-control" value={dateCol} onChange={e => setDateCol(e.target.value)} />
                </div>
              </div>

              {modelType === 'MLR' && (
                <div className="form-group" style={{marginBottom: '1.5rem'}}>
                  <label>Fitur Pendukung Multivariate (Pisahkan dgn koma)</label>
                  <input type="text" className="form-control" value={featureCols} onChange={e => setFeatureCols(e.target.value)} placeholder="Contoh: Open,High,Low,Volume" />
                </div>
              )}

              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? <Loader2 className="loader" size={20} /> : <Activity size={20} />}
                {loading ? 'AI Sedang Melatih Model...' : 'Jalankan Eksperimen'}
              </button>
            </form>
          </div>
        </div>

        {/* PANEL KANAN: HASIL VISUALISASI */}
        <div>
          {currentResult ? (
            <div className="glass-card" style={{height: '100%'}}>
              <h2 className="card-title">
                <Activity size={24} color="#10b981" /> Hasil Evaluasi: {currentResult.model}
              </h2>
              
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Tingkat Kesalahan (MAPE)</div>
                  <div className="metric-value" style={{color: currentResult.mape < 5 ? '#10b981' : (currentResult.mape < 25 ? '#f59e0b' : '#ef4444')}}>
                    {currentResult.mape}%
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Selisih Harga (MAE)</div>
                  <div className="metric-value">{currentResult.mae}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">RMSE</div>
                  <div className="metric-value">{currentResult.rmse}</div>
                </div>
              </div>

              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={currentResult.chartData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="date" stroke="#94a3b8" tick={{fill: '#94a3b8'}} />
                    <YAxis stroke="#94a3b8" tick={{fill: '#94a3b8'}} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px'}} />
                    <Legend />
                    <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={2} dot={false} name="Data Aktual" />
                    <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} name={`Prediksi ${currentResult.model}`} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="glass-card" style={{height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)'}}>
              <Activity size={64} style={{opacity: 0.2, marginBottom: '1rem'}} />
              <p>Belum ada eksperimen yang dijalankan.</p>
              <p style={{fontSize: '0.875rem'}}>Pilih dataset dan algoritma di sebelah kiri untuk memulai.</p>
            </div>
          )}
        </div>
      </div>

      {/* RIWAYAT EKSPERIMEN */}
      {history.length > 0 && (
        <div className="glass-card" style={{marginTop: '2rem'}}>
          <h2 className="card-title">
            <History size={24} color="#f59e0b" /> Riwayat & Perbandingan Eksperimen
          </h2>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Waktu</th>
                  <th>Dataset</th>
                  <th>Algoritma Model</th>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>MAPE</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td>{new Date(item.id).toLocaleTimeString()}</td>
                    <td>{item.fileName}</td>
                    <td><strong style={{color: '#60a5fa'}}>{item.model}</strong></td>
                    <td>{item.rmse}</td>
                    <td>{item.mae}</td>
                    <td>
                      <span style={{
                        background: item.mape < 5 ? 'rgba(16, 185, 129, 0.2)' : (item.mape < 25 ? 'rgba(245, 158, 11, 0.2)' : 'rgba(239, 68, 68, 0.2)'),
                        color: item.mape < 5 ? '#34d399' : (item.mape < 25 ? '#fbbf24' : '#fca5a5'),
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontWeight: '600'
                      }}>
                        {item.mape}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
