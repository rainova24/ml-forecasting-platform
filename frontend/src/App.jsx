import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { UploadCloud, Activity, Database, Cpu, Settings2, Loader2, History, Box } from 'lucide-react';

function App() {
  // State Navigasi
  const [activeTab, setActiveTab] = useState('auto'); // 'auto' | 'pretrained'

  // ==========================================
  // STATE: TAB 1 (Auto-Training)
  // ==========================================
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState('LSTM');
  const [targetCol, setTargetCol] = useState('Close');
  const [dateCol, setDateCol] = useState('Date');
  const [featureCols, setFeatureCols] = useState('Open,High,Low,Volume');
  const [timeResample, setTimeResample] = useState('none');
  const [missingValues, setMissingValues] = useState('drop');
  const [oneHotEncodeCols, setOneHotEncodeCols] = useState('');
  
  // ==========================================
  // STATE: TAB 2 (Pre-Trained Models)
  // ==========================================
  const [pretrainedFile, setPretrainedFile] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [ptTargetCol, setPtTargetCol] = useState('Close');
  const [ptDateCol, setPtDateCol] = useState('Date');
  const [testSize, setTestSize] = useState(0.2);
  const [ptJsonInput, setPtJsonInput] = useState('');
  const [ptJsonOutput, setPtJsonOutput] = useState(null);

  // ==========================================
  // GLOBAL STATE
  // ==========================================
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Fetch daftar model saat tab Pre-trained dibuka
  useEffect(() => {
    if (activeTab === 'pretrained') {
      axios.get('http://127.0.0.1:8000/api/pretrained/models')
        .then(res => {
          setAvailableModels(res.data.models || []);
          if (res.data.models && res.data.models.length > 0) {
            setSelectedModel(res.data.models[0].name);
          }
        })
        .catch(err => console.error("Gagal menarik daftar model:", err));
    }
  }, [activeTab]);

  const handleFileChange = (e, isPretrained = false) => {
    if (e.target.files && e.target.files.length > 0) {
      if (isPretrained) setPretrainedFile(e.target.files[0]);
      else setFile(e.target.files[0]);
    }
  };

  // EKSEKUSI TAB 1 (AUTO-TRAINING)
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
    formData.append('time_resample', timeResample);
    formData.append('missing_values', missingValues);
    formData.append('one_hot_encode_cols', oneHotEncodeCols);
    if (modelType === 'MLR') {
      formData.append('feature_cols', featureCols);
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/experiment', formData);
      const resData = response.data;
      
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
      setError(err.response?.data?.error || "Terjadi kesalahan sistem.");
    } finally {
      setLoading(false);
    }
  };

  // EKSEKUSI TAB 2 (PRE-TRAINED EVALUATE)
  const handleRunPretrained = async (e) => {
    e.preventDefault();
    if (!pretrainedFile) {
      setError("Mohon upload file CSV pengujian terlebih dahulu.");
      return;
    }
    if (!selectedModel) {
      setError("Pilih model yang tersedia terlebih dahulu.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', pretrainedFile);
    formData.append('model_name', selectedModel);
    formData.append('target_col', ptTargetCol);
    formData.append('date_col', ptDateCol);
    formData.append('test_size', testSize);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/pretrained/evaluate', formData);
      const resData = response.data;
      
      const chartData = resData.chart_data.dates.map((date, index) => ({
        date: date,
        Actual: resData.chart_data.actual[index],
        Predicted: resData.chart_data.predicted[index],
      }));
      
      const resultObj = {
        id: Date.now(),
        fileName: pretrainedFile.name,
        model: resData.model_name + " (Pre-Trained)",
        mape: resData.metrics.MAPE,
        rmse: resData.metrics.RMSE,
        mae: resData.metrics.MAE,
        chartData: chartData
      };
      
      setCurrentResult(resultObj);
      setHistory(prev => [resultObj, ...prev]);
    } catch (err) {
      setError(err.response?.data?.error || "Gagal menguji Pre-Trained model.");
    } finally {
      setLoading(false);
    }
  };

  // EKSEKUSI TAB 2 (PRE-TRAINED PREDICT JSON)
  const handleRunJsonPrediction = async (e) => {
    e.preventDefault();
    if (!selectedModel) {
      setError("Pilih model terlebih dahulu.");
      return;
    }
    try {
      let parsedInput;
      try {
        parsedInput = JSON.parse(ptJsonInput);
      } catch (e) {
        setError("Format JSON tidak valid. Pastikan penulisan menggunakan kutip ganda (\"\").");
        return;
      }

      setLoading(true);
      setError(null);
      setPtJsonOutput(null);
      
      const response = await axios.post('http://127.0.0.1:8000/api/pretrained/predict', {
        model_name: selectedModel,
        input: parsedInput
      });
      setPtJsonOutput(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Gagal menjalankan prediksi JSON.");
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

      {/* TAB NAVIGATION */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', borderBottom: '1px solid #334155', paddingBottom: '1rem' }}>
        <button 
          onClick={() => {setActiveTab('auto'); setError(null); setCurrentResult(null);}} 
          style={{ background: activeTab === 'auto' ? '#3b82f6' : 'transparent', color: activeTab === 'auto' ? '#fff' : '#94a3b8', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Activity size={20} /> Auto-Training (Eksperimen Bebas)
        </button>
        <button 
          onClick={() => {setActiveTab('pretrained'); setError(null); setCurrentResult(null);}} 
          style={{ background: activeTab === 'pretrained' ? '#10b981' : 'transparent', color: activeTab === 'pretrained' ? '#fff' : '#94a3b8', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Box size={20} /> Model Kelompok (Pre-Trained)
        </button>
      </div>

      {error && (
        <div style={{background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444', padding: '1rem', marginBottom: '2rem', borderRadius: '4px', color: '#fca5a5'}}>
          <strong>Error: </strong> {error}
        </div>
      )}

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem'}}>
        {/* PANEL KIRI: KONFIGURASI SESUAI TAB AKTIF */}
        <div style={{minWidth: 0}}>
          
          {/* ============== TAB 1: AUTO TRAINING ============== */}
          {activeTab === 'auto' && (
            <>
              <div className="glass-card">
                <h2 className="card-title">
                  <Database size={24} color="#3b82f6" /> 1. Upload Dataset
                </h2>
                <div className="file-upload-wrapper">
                  <input type="file" accept=".csv" onChange={e => handleFileChange(e, false)} />
                  <div className="file-upload-btn">
                    <UploadCloud size={24} />
                    {file ? file.name : 'Pilih File CSV Dataset Anda'}
                  </div>
                </div>
              </div>

              <div className="glass-card">
                <h2 className="card-title">
                  <Settings2 size={24} color="#f59e0b" /> 2. Teknik Preprocessing
                </h2>
                <div className="form-group" style={{marginBottom: '1rem'}}>
                  <label>Time Resampling</label>
                  <select className="form-control" value={timeResample} onChange={e => setTimeResample(e.target.value)}>
                    <option value="none">Tanpa Resampling</option>
                    <option value="daily">Harian (Daily)</option>
                    <option value="weekly">Mingguan (Weekly)</option>
                    <option value="monthly">Bulanan (Monthly)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Data Kosong (Missing Values)</label>
                  <select className="form-control" value={missingValues} onChange={e => setMissingValues(e.target.value)}>
                    <option value="drop">Hapus Baris (Drop NA)</option>
                    <option value="mean">Isi Rata-rata (Mean Imputation)</option>
                  </select>
                </div>
              </div>

              <div className="glass-card">
                <h2 className="card-title">
                  <Cpu size={24} color="#a78bfa" /> 3. Parameter Model
                </h2>
                <form onSubmit={handleRunExperiment} className="form-group">
                  <div className="form-group" style={{marginBottom: '1rem'}}>
                    <label>Algoritma Machine Learning</label>
                    <select className="form-control" value={modelType} onChange={e => setModelType(e.target.value)}>
                      <option value="LSTM">Deep Learning (LSTM)</option>
                      <option value="ARIMA">Statistik (ARIMA)</option>
                      <option value="Holt-Winters">Pemulusan (Holt-Winters)</option>
                      <option value="Prophet">Additive (Meta Prophet)</option>
                      <option value="MLR">Multivariate (Regresi Berganda)</option>
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
                  {(modelType === 'Prophet-Add-Regressor' || modelType === 'MLR') && (
                    <div className="form-group" style={{marginBottom: '1rem'}}>
                      <label>Kolom Kategori (One-Hot Encoding)</label>
                      <input type="text" className="form-control" placeholder="Contoh: Kondisi_Mental, Cuaca" value={oneHotEncodeCols} onChange={e => setOneHotEncodeCols(e.target.value)} />
                      <small style={{color: '#64748b'}}>*Wajib diisi untuk Prophet-Add-Regressor buatan Firman</small>
                    </div>
                  )}
                  {modelType === 'MLR' && (
                    <div className="form-group" style={{marginBottom: '1.5rem'}}>
                      <label>Fitur Pendukung Numerik</label>
                      <input type="text" className="form-control" placeholder="Contoh: Open,High,Low" value={featureCols} onChange={e => setFeatureCols(e.target.value)} />
                    </div>
                  )}
                  <button type="submit" className="btn-primary" disabled={loading}>
                    {loading ? <Loader2 className="loader" size={20} /> : <Activity size={20} />}
                    {loading ? 'AI Sedang Melatih Model...' : 'Jalankan Pelatihan Baru'}
                  </button>
                </form>
              </div>
            </>
          )}

          {/* ============== TAB 2: PRE-TRAINED MODELS ============== */}
          {activeTab === 'pretrained' && (
            <>
              <div className="glass-card">
                <h2 className="card-title">
                  <Box size={24} color="#10b981" /> 1. Pilih Model Kelompok
                </h2>
                {availableModels.length === 0 ? (
                  <p style={{color: '#fca5a5'}}>Belum ada model yang diupload ke folder <code>pretrained_models/</code></p>
                ) : (
                  <div className="form-group">
                    <label>Model Tersedia</label>
                    <select className="form-control" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                      {availableModels.map(m => (
                        <option key={m.name} value={m.name}>{m.name} ({m.model_type})</option>
                      ))}
                    </select>
                    {availableModels.find(m => m.name === selectedModel)?.description && (
                      <p style={{marginTop: '0.5rem', fontSize: '0.875rem', color: '#94a3b8'}}>
                        <i>{availableModels.find(m => m.name === selectedModel).description}</i>
                      </p>
                    )}
                  </div>
                )}
              </div>

              <div className="glass-card">
                <h2 className="card-title">
                  <Database size={24} color="#3b82f6" /> 2. Upload Data Uji (Test Data)
                </h2>
                <div className="file-upload-wrapper" style={{marginBottom: '1rem'}}>
                  <input type="file" accept=".csv" onChange={e => handleFileChange(e, true)} />
                  <div className="file-upload-btn">
                    <UploadCloud size={24} />
                    {pretrainedFile ? pretrainedFile.name : 'Pilih CSV Pengujian'}
                  </div>
                </div>
                
                <form onSubmit={handleRunPretrained}>
                  <div className="form-grid" style={{marginBottom: '1rem'}}>
                    <div className="form-group">
                      <label>Kolom Target (Y)</label>
                      <input type="text" className="form-control" value={ptTargetCol} onChange={e => setPtTargetCol(e.target.value)} />
                    </div>
                    <div className="form-group">
                      <label>Kolom Tanggal</label>
                      <input type="text" className="form-control" value={ptDateCol} onChange={e => setPtDateCol(e.target.value)} />
                    </div>
                  </div>
                  <div className="form-group" style={{marginBottom: '1.5rem'}}>
                    <label>Porsi Data Uji (0.1 - 0.5)</label>
                    <input type="number" step="0.1" min="0.1" max="0.5" className="form-control" value={testSize} onChange={e => setTestSize(e.target.value)} />
                    <small style={{color: '#64748b', marginTop: '0.25rem', display: 'block'}}>Secara default, 20% data terakhir (0.2) akan digunakan untuk mengevaluasi akurasi model.</small>
                  </div>
                  
                  <button type="submit" className="btn-primary" disabled={loading || availableModels.length === 0} style={{background: '#10b981'}}>
                    {loading ? <Loader2 className="loader" size={20} /> : <Activity size={20} />}
                    {loading ? 'Menguji Model...' : 'Uji Akurasi Model Ini'}
                  </button>
                </form>
              </div>

              <div className="glass-card" style={{marginTop: '2rem'}}>
                <h2 className="card-title">
                  <Cpu size={24} color="#a78bfa" /> 3. Prediksi Satuan (JSON Input)
                </h2>
                <div className="form-group" style={{marginBottom: '1rem'}}>
                  <label>Skema Input yang Dibutuhkan</label>
                  <div style={{background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', fontSize: '0.875rem', fontFamily: 'monospace', color: '#a78bfa', whiteSpace: 'pre-wrap', wordBreak: 'break-word'}}>
                    {availableModels.find(m => m.name === selectedModel)?.input_schema ? JSON.stringify(availableModels.find(m => m.name === selectedModel).input_schema) : "Belum ada skema metadata.json."}
                  </div>
                </div>
                <form onSubmit={handleRunJsonPrediction}>
                  <div className="form-group" style={{marginBottom: '1rem'}}>
                    <label>Input Data (JSON)</label>
                    <textarea className="form-control" rows="6" value={ptJsonInput} onChange={e => setPtJsonInput(e.target.value)} placeholder='Contoh: {"ds": "2026-06-01", "cat_Learning": 1}' style={{fontFamily: 'monospace'}}></textarea>
                  </div>
                  <button type="submit" className="btn-primary" disabled={loading || availableModels.length === 0} style={{background: '#a78bfa'}}>
                    {loading ? <Loader2 className="loader" size={20} /> : <Activity size={20} />}
                    {loading ? 'Memproses JSON...' : 'Jalankan Prediksi JSON'}
                  </button>
                </form>
                {ptJsonOutput && (
                  <div style={{marginTop: '1.5rem'}}>
                    <label style={{color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem'}}><Activity size={16}/> Inferensi Berhasil</label>
                    <pre style={{background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px', overflowX: 'auto', color: '#e2e8f0', fontFamily: 'monospace', fontSize: '0.875rem', border: '1px solid #334155', whiteSpace: 'pre-wrap', wordBreak: 'break-word'}}>
                      {JSON.stringify(ptJsonOutput, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </>
          )}

        </div>

        {/* PANEL KANAN: HASIL VISUALISASI (Global untuk kedua tab) */}
        <div style={{minWidth: 0}}>
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
                    <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} name={`Prediksi Model`} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="glass-card" style={{height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)'}}>
              <Activity size={64} style={{opacity: 0.2, marginBottom: '1rem'}} />
              <p>Belum ada eksperimen yang dijalankan.</p>
              <p style={{fontSize: '0.875rem'}}>Lakukan pengujian di sebelah kiri untuk melihat hasil.</p>
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
                  <th>Algoritma / Nama Model</th>
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
