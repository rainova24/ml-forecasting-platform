import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { UploadCloud, Activity, Database, Cpu, Settings2, Loader2, History, FlaskConical, ChevronRight, ServerCrash, CheckCircle2, BarChart2 } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState('LSTM');
  const [targetCol, setTargetCol] = useState('Close');
  const [dateCol, setDateCol] = useState('Date');
  const [featureCols, setFeatureCols] = useState('Open,High,Low,Volume');
  const [oneHotEncodeCols, setOneHotEncodeCols] = useState('');
  const [windowSize, setWindowSize] = useState(0);

  // State Baru untuk Preprocessing
  const [timeResample, setTimeResample] = useState('none');
  const [missingValues, setMissingValues] = useState('drop');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [currentResult, setCurrentResult] = useState(null);
  const [history, setHistory] = useState([]);

  // ── Pre-trained model tester ──────────────────────────────────────────────
  const [pretrainedModels, setPretrainedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [jsonInput, setJsonInput] = useState('');
  const [ptLoading, setPtLoading] = useState(false);
  const [ptError, setPtError] = useState(null);
  const [ptResult, setPtResult] = useState(null);
  const outputRef = useRef(null);

  // ── Evaluation state ─────────────────────────────────────────────────────
  const [evalFile, setEvalFile] = useState(null);
  const [evalModelName, setEvalModelName] = useState('');
  const [evalTestSize, setEvalTestSize] = useState(20);
  const [evalTargetCol, setEvalTargetCol] = useState('Close');
  const [evalDateCol, setEvalDateCol] = useState('Date');
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const evalResultRef = useRef(null);

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/api/pretrained/models')
      .then(res => {
        setPretrainedModels(res.data.models || []);
        if (res.data.models?.length > 0) {
          setSelectedModel(res.data.models[0].name);
          setEvalModelName(res.data.models[0].name);
        }
      })
      .catch(() => {}); // server may not be running yet
  }, []);

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

    // Parameter Preprocessing Baru
    formData.append('time_resample', timeResample);
    formData.append('missing_values', missingValues);

    if (modelType === 'MLR') {
      formData.append('feature_cols', featureCols);
      formData.append('window_size', windowSize);
    }

    formData.append('one_hot_encode_cols', oneHotEncodeCols);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/experiment', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

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
      setError(err.response?.data?.error || "Terjadi kesalahan sistem saat melatih model.");
    } finally {
      setLoading(false);
    }
  };

  // ── Pre-trained tester handler ────────────────────────────────────────────
  const handlePretrainedPredict = async (e) => {
    e.preventDefault();
    setPtError(null);
    setPtResult(null);

    // Validate JSON
    let parsed;
    try {
      parsed = JSON.parse(jsonInput);
    } catch {
      setPtError('Input bukan JSON yang valid. Periksa format kembali.');
      return;
    }

    if (!selectedModel) {
      setPtError('Pilih model terlebih dahulu.');
      return;
    }

    setPtLoading(true);
    try {
      const res = await axios.post('http://127.0.0.1:8000/api/pretrained/predict', {
        model_name: selectedModel,
        input: parsed,
      });
      setPtResult(res.data);
      setTimeout(() => outputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
    } catch (err) {
      setPtError(err.response?.data?.error || 'Terjadi kesalahan saat memanggil model.');
    } finally {
      setPtLoading(false);
    }
  };

  // ── Evaluation handler ──────────────────────────────────────────────────
  const handleEvaluatePreTrained = async (e) => {
    e.preventDefault();
    setEvalError(null);
    setEvalResult(null);

    if (!evalFile) {
      setEvalError('Mohon upload file CSV terlebih dahulu.');
      return;
    }
    if (!evalModelName) {
      setEvalError('Pilih model terlebih dahulu.');
      return;
    }

    setEvalLoading(true);
    const formData = new FormData();
    formData.append('file', evalFile);
    formData.append('model_name', evalModelName);
    formData.append('target_col', evalTargetCol);
    formData.append('date_col', evalDateCol);
    formData.append('test_size', evalTestSize / 100);

    try {
      const res = await axios.post('http://127.0.0.1:8000/api/pretrained/evaluate', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const data = res.data;
      const chartData = data.chart_data.dates.map((date, i) => ({
        date,
        Actual: data.chart_data.actual[i],
        Predicted: data.chart_data.predicted[i],
      }));
      setEvalResult({ ...data, chartData });
      setTimeout(() => evalResultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
    } catch (err) {
      setEvalError(err.response?.data?.error || 'Terjadi kesalahan saat mengevaluasi model.');
    } finally {
      setEvalLoading(false);
    }
  };

  const selectedModelMeta = pretrainedModels.find(m => m.name === selectedModel);

  return (
    <div className="container">
      <header className="header">
        <h1>Forecasting Experiment Platform</h1>
        <p>Advanced Machine Learning Web Interface</p>
      </header>

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444', padding: '1rem', marginBottom: '2rem', borderRadius: '4px', color: '#fca5a5' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
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

          {/* KARTU BARU: TEKNIK PREPROCESSING */}
          <div className="glass-card">
            <h2 className="card-title">
              <Settings2 size={24} color="#f59e0b" /> 2. Teknik Preprocessing
            </h2>
            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Time Resampling (Rekapitulasi Waktu)</label>
              <select className="form-control" value={timeResample} onChange={e => setTimeResample(e.target.value)}>
                <option value="none">Tanpa Resampling (Bawaan Dataset)</option>
                <option value="daily">Rata-rata Harian (Daily)</option>
                <option value="weekly">Rata-rata Mingguan (Weekly)</option>
                <option value="monthly">Rata-rata Bulanan (Monthly)</option>
              </select>
            </div>
            <div className="form-group">
              <label>Penanganan Data Kosong (Missing Values)</label>
              <select className="form-control" value={missingValues} onChange={e => setMissingValues(e.target.value)}>
                <option value="drop">Hapus Baris yang Kosong (Drop NA)</option>
                <option value="mean">Isi dengan Rata-rata (Mean Imputation)</option>
              </select>
            </div>
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', fontSize: '0.875rem', color: '#94a3b8' }}>
              <span style={{ color: '#60a5fa' }}>Info:</span> AI akan otomatis mendeteksi dan menerjemahkan penulisan nama bulan Indonesia (cth: "Juni") menjadi format waktu internasional.
            </div>
            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label>One Hot Encoding (Pisahkan dgn koma)</label>
              <input type="text" className="form-control" value={oneHotEncodeCols} onChange={e => setOneHotEncodeCols(e.target.value)} />
            </div>
            {modelType === 'MLR' && (
              // Window Size untuk MLR
              <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                <label>Window Size (Jumlah Periode ke Belakang yang Dipakai)</label>
                <input type="text" className="form-control" value={windowSize} onChange={e => setWindowSize(e.target.value)} />
                <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: 'rgba(250, 204, 21, 0.1)', borderRadius: '4px', fontSize: '0.8rem', color: '#fca5a5' }}>
                  <strong>Contoh:</strong> Jika input 3, model akan memakai data 3 periode terakhir untuk prediksi.
                </div>
              </div>
            )}
          </div>

          <div className="glass-card">
            <h2 className="card-title">
              <Cpu size={24} color="#a78bfa" /> 3. Parameter Model
            </h2>
            <form onSubmit={handleRunExperiment} className="form-group">

              <div className="form-group" style={{ marginBottom: '1rem' }}>
                <label>Algoritma Machine Learning</label>
                <select className="form-control" value={modelType} onChange={e => setModelType(e.target.value)}>
                  <option value="LSTM">Deep Learning (LSTM)</option>
                  <option value="ARIMA">Statistik (ARIMA)</option>
                  <option value="Holt-Winters">Pemulusan (Holt-Winters)</option>
                  <option value="Prophet">Additive (Meta Prophet)</option>
                  <option value="Prophet-Add-Regressor">Additive (Meta Prophet) + Regressor</option>
                  <option value="MLR">Multivariate (Regresi Linear Berganda)</option>
                </select>
              </div>

              <div className="form-grid" style={{ marginBottom: '1rem' }}>
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
                <div className="form-group" style={{ marginBottom: '1.5rem' }}>
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
            <div className="glass-card" style={{ height: '100%' }}>
              <h2 className="card-title">
                <Activity size={24} color="#10b981" /> Hasil Evaluasi: {currentResult.model}
              </h2>

              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Tingkat Kesalahan (MAPE)</div>
                  <div className="metric-value" style={{ color: currentResult.mape < 5 ? '#10b981' : (currentResult.mape < 25 ? '#f59e0b' : '#ef4444') }}>
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
                    <XAxis dataKey="date" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={2} dot={false} name="Data Aktual" />
                    <Line type="monotone" dataKey="Predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} name={`Prediksi ${currentResult.model}`} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="glass-card" style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
              <Activity size={64} style={{ opacity: 0.2, marginBottom: '1rem' }} />
              <p>Belum ada eksperimen yang dijalankan.</p>
              <p style={{ fontSize: '0.875rem' }}>Pilih dataset dan algoritma di sebelah kiri untuk memulai.</p>
            </div>
          )}
        </div>
      </div>

      {/* RIWAYAT EKSPERIMEN */}
      {history.length > 0 && (
        <div className="glass-card" style={{ marginTop: '2rem' }}>
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
                    <td><strong style={{ color: '#60a5fa' }}>{item.model}</strong></td>
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

      {/* ═══════════════════════════════════════════════════════
          PRE-TRAINED MODEL TESTER
      ═══════════════════════════════════════════════════════ */}
      <div className="glass-card" style={{ marginTop: '2rem' }}>
        <h2 className="card-title">
          <FlaskConical size={24} color="#a78bfa" /> Uji Model Pre-Trained (Server)
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>

          {/* LEFT: Form */}
          <form onSubmit={handlePretrainedPredict} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            {/* Model Selector */}
            <div className="form-group">
              <label>Pilih Model Pre-Trained</label>
              {pretrainedModels.length === 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1rem', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '8px', color: '#fbbf24', fontSize: '0.875rem' }}>
                  <ServerCrash size={16} />
                  Belum ada model tersedia. Tambahkan artefak ke <code style={{ fontFamily: 'monospace', background: 'rgba(255,255,255,0.05)', padding: '1px 5px', borderRadius: '4px' }}>backend/pretrained_models/</code>
                </div>
              ) : (
                <select
                  className="form-control"
                  value={selectedModel}
                  onChange={e => { setSelectedModel(e.target.value); setPtResult(null); setPtError(null); }}
                >
                  {pretrainedModels.map(m => (
                    <option key={m.name} value={m.name}>{m.name} ({m.model_type})</option>
                  ))}
                </select>
              )}
            </div>

            {/* Model Description */}
            {selectedModelMeta && (
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.2)', borderRadius: '8px', fontSize: '0.85rem', color: '#c4b5fd' }}>
                <strong style={{ color: '#a78bfa' }}>Deskripsi: </strong>{selectedModelMeta.description || '—'}
                {selectedModelMeta.input_schema && Object.keys(selectedModelMeta.input_schema).length > 0 && (
                  <div style={{ marginTop: '0.5rem', color: '#94a3b8' }}>
                    <strong style={{ color: '#7dd3fc' }}>Skema Input: </strong>
                    <code style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{JSON.stringify(selectedModelMeta.input_schema, null, 2)}</code>
                  </div>
                )}
              </div>
            )}

            {/* JSON Input */}
            <div className="form-group">
              <label>Input Data (JSON)</label>
              <textarea
                className="form-control"
                value={jsonInput}
                onChange={e => setJsonInput(e.target.value)}
                placeholder={'// Contoh flat array:\n[150.5, 148.2, 152.0]\n\n// Contoh 2-D (multi-row):\n[[150.5, 148.2, 152.0], [149.0, 147.5, 153.1]]'}
                rows={8}
                style={{
                  fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
                  fontSize: '0.85rem',
                  resize: 'vertical',
                  lineHeight: '1.6',
                }}
              />
            </div>

            {/* Error */}
            {ptError && (
              <div style={{ background: 'rgba(239,68,68,0.1)', borderLeft: '4px solid #ef4444', padding: '0.75rem 1rem', borderRadius: '4px', color: '#fca5a5', fontSize: '0.875rem' }}>
                <strong>Error: </strong>{ptError}
              </div>
            )}

            <button
              type="submit"
              className="btn-primary"
              disabled={ptLoading || pretrainedModels.length === 0}
              style={{ background: 'linear-gradient(135deg, #7c3aed, #a78bfa)', boxShadow: '0 4px 14px rgba(139,92,246,0.4)' }}
            >
              {ptLoading
                ? <><Loader2 className="loader" size={20} /> Memproses Inferensi...</>
                : <><ChevronRight size={20} /> Jalankan Prediksi</>
              }
            </button>
          </form>

          {/* RIGHT: JSON Output */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', fontWeight: '500' }}>Output JSON</div>

            {ptResult ? (
              <div ref={outputRef} style={{ position: 'relative' }}>
                {/* Status badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: '#34d399', fontSize: '0.875rem', fontWeight: '600' }}>
                  <CheckCircle2 size={16} /> Inferensi Berhasil &mdash; <span style={{ color: '#94a3b8', fontWeight: '400' }}>{ptResult.model_name} ({ptResult.model_type})</span>
                </div>
                <pre style={{
                  background: 'rgba(15,23,42,0.8)',
                  border: '1px solid rgba(167,139,250,0.3)',
                  borderRadius: '10px',
                  padding: '1.25rem',
                  color: '#e2e8f0',
                  fontSize: '0.85rem',
                  fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
                  lineHeight: '1.65',
                  overflowX: 'auto',
                  overflowY: 'auto',
                  maxHeight: '380px',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}>
                  {JSON.stringify(ptResult, null, 2)}
                </pre>
              </div>
            ) : (
              <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '240px',
                background: 'rgba(15,23,42,0.4)',
                border: '1px dashed rgba(255,255,255,0.1)',
                borderRadius: '10px',
                color: 'var(--text-muted)',
                gap: '0.75rem',
                fontSize: '0.875rem',
              }}>
                <FlaskConical size={40} style={{ opacity: 0.2 }} />
                <span>Output akan muncul di sini setelah prediksi dijalankan.</span>
              </div>
            )}
          </div>

        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════
          EVALUASI MODEL PRE-TRAINED DENGAN DATASET CSV
      ═══════════════════════════════════════════════════════ */}
      <div className="glass-card" style={{ marginTop: '2rem' }}>
        <h2 className="card-title">
          <BarChart2 size={24} color="#10b981" /> Evaluasi Model Pre-Trained dengan Dataset CSV
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '2rem', alignItems: 'start' }}>

          {/* LEFT: Configuration Form */}
          <form onSubmit={handleEvaluatePreTrained} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            {/* CSV Upload */}
            <div className="form-group">
              <label>Dataset CSV untuk Evaluasi</label>
              <div className="file-upload-wrapper">
                <input
                  type="file"
                  accept=".csv"
                  onChange={e => { if (e.target.files?.[0]) setEvalFile(e.target.files[0]); }}
                />
                <div
                  className="file-upload-btn"
                  style={{ borderColor: '#10b981', color: '#10b981', background: 'rgba(16,185,129,0.08)' }}
                >
                  <UploadCloud size={20} />
                  {evalFile ? evalFile.name : 'Pilih File CSV Dataset'}
                </div>
              </div>
            </div>

            {/* Model Selector */}
            <div className="form-group">
              <label>Model Pre-Trained yang Dievaluasi</label>
              {pretrainedModels.length === 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1rem', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: '8px', color: '#fbbf24', fontSize: '0.875rem' }}>
                  <ServerCrash size={16} /> Belum ada model tersedia di server
                </div>
              ) : (
                <select
                  className="form-control"
                  value={evalModelName}
                  onChange={e => { setEvalModelName(e.target.value); setEvalResult(null); setEvalError(null); }}
                >
                  {pretrainedModels.map(m => (
                    <option key={m.name} value={m.name}>{m.name} ({m.model_type})</option>
                  ))}
                </select>
              )}
            </div>

            {/* Column Config */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div className="form-group">
                <label>Kolom Target (Y)</label>
                <input
                  type="text"
                  className="form-control"
                  value={evalTargetCol}
                  onChange={e => setEvalTargetCol(e.target.value)}
                  placeholder="Close"
                />
              </div>
              <div className="form-group">
                <label>Kolom Tanggal</label>
                <input
                  type="text"
                  className="form-control"
                  value={evalDateCol}
                  onChange={e => setEvalDateCol(e.target.value)}
                  placeholder="Date"
                />
              </div>
            </div>

            {/* Test Size Slider */}
            <div className="form-group">
              <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Ukuran Data Uji (Test Size)</span>
                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.05rem', letterSpacing: '-0.5px' }}>{evalTestSize}%</span>
              </label>
              <input
                type="range"
                min="5" max="50" step="1"
                value={evalTestSize}
                onChange={e => setEvalTestSize(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#10b981', cursor: 'pointer' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                <span>5%</span>
                <span style={{ color: '#475569' }}>{evalTestSize}% data terakhir sebagai set uji</span>
                <span>50%</span>
              </div>
            </div>

            {/* Error */}
            {evalError && (
              <div style={{ background: 'rgba(239,68,68,0.1)', borderLeft: '4px solid #ef4444', padding: '0.75rem 1rem', borderRadius: '4px', color: '#fca5a5', fontSize: '0.875rem' }}>
                <strong>Error: </strong>{evalError}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              className="btn-primary"
              disabled={evalLoading || pretrainedModels.length === 0}
              style={{ background: 'linear-gradient(135deg, #059669, #10b981)', boxShadow: '0 4px 14px rgba(16,185,129,0.4)' }}
            >
              {evalLoading
                ? <><Loader2 className="loader" size={20} /> Mengevaluasi Model...</>
                : <><BarChart2 size={20} /> Jalankan Evaluasi</>
              }
            </button>
          </form>

          {/* RIGHT: Results Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {evalResult ? (
              <div ref={evalResultRef}>

                {/* Status badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#34d399', fontSize: '0.875rem', fontWeight: '600', flexWrap: 'wrap' }}>
                  <CheckCircle2 size={16} />
                  Evaluasi Berhasil &mdash;&nbsp;
                  <span style={{ color: '#94a3b8', fontWeight: '400' }}>
                    {evalResult.model_name} &middot; {evalResult.n_test_samples} sampel uji &middot; {evalResult.test_size_pct}% test split
                  </span>
                </div>

                {/* Metrics Grid — 4 cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>

                  {/* MAE */}
                  <div className="metric-card" style={{ border: '1px solid rgba(59,130,246,0.3)' }}>
                    <div className="metric-label">MAE</div>
                    <div className="metric-value" style={{ fontSize: '1.35rem', color: '#60a5fa' }}>{evalResult.metrics.MAE}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Mean Abs. Error</div>
                  </div>

                  {/* RMSE */}
                  <div className="metric-card" style={{ border: '1px solid rgba(167,139,250,0.3)' }}>
                    <div className="metric-label">RMSE</div>
                    <div className="metric-value" style={{ fontSize: '1.35rem', color: '#a78bfa' }}>{evalResult.metrics.RMSE}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Root Mean Sq. Err</div>
                  </div>

                  {/* R² */}
                  <div
                    className="metric-card"
                    style={{ border: `1px solid rgba(${evalResult.metrics.R2 >= 0.8 ? '16,185,129' : evalResult.metrics.R2 >= 0.5 ? '245,158,11' : '239,68,68'},0.3)` }}
                  >
                    <div className="metric-label">R² Score</div>
                    <div
                      className="metric-value"
                      style={{ fontSize: '1.35rem', color: evalResult.metrics.R2 >= 0.8 ? '#10b981' : evalResult.metrics.R2 >= 0.5 ? '#f59e0b' : '#ef4444' }}
                    >
                      {evalResult.metrics.R2}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Goodness of Fit</div>
                  </div>

                  {/* MAPE */}
                  <div
                    className="metric-card"
                    style={{ border: `1px solid rgba(${evalResult.metrics.MAPE < 5 ? '16,185,129' : evalResult.metrics.MAPE < 25 ? '245,158,11' : '239,68,68'},0.3)` }}
                  >
                    <div className="metric-label">MAPE</div>
                    <div
                      className="metric-value"
                      style={{ fontSize: '1.35rem', color: evalResult.metrics.MAPE < 5 ? '#10b981' : evalResult.metrics.MAPE < 25 ? '#f59e0b' : '#ef4444' }}
                    >
                      {evalResult.metrics.MAPE}%
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Mean Abs. % Error</div>
                  </div>
                </div>

                {/* Actual vs Predicted Chart */}
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={evalResult.chartData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
                      <Legend />
                      <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={2} dot={false} name="Data Aktual (Test)" />
                      <Line type="monotone" dataKey="Predicted" stroke="#a78bfa" strokeWidth={2} strokeDasharray="5 5" dot={false} name={`Prediksi ${evalResult.model_name}`} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

              </div>
            ) : (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                minHeight: '340px', background: 'rgba(15,23,42,0.4)',
                border: '1px dashed rgba(16,185,129,0.2)', borderRadius: '10px',
                color: 'var(--text-muted)', gap: '0.75rem', fontSize: '0.875rem'
              }}>
                <BarChart2 size={52} style={{ opacity: 0.15, color: '#10b981' }} />
                <span style={{ fontWeight: '500' }}>Hasil evaluasi akan muncul di sini</span>
                <span style={{ fontSize: '0.8rem', color: '#475569', textAlign: 'center', maxWidth: '260px' }}>
                  Upload dataset CSV, pilih model, dan klik Jalankan Evaluasi
                </span>
              </div>
            )}
          </div>

        </div>
      </div>

    </div>
  );
}

export default App;
