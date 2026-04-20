import React, { useState, useEffect, useCallback } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import StatsCards from './components/StatsCards'
import LiveStream from './components/LiveStream'
import UploadBox from './components/UploadBox'
import ResultViewer from './components/ResultViewer'
import ScanHistory from './components/ScanHistory'
import LiveSessions from './components/LiveSessions'
import QRConnect from './components/QRConnect'
import ActivityFeed from './components/ActivityFeed'
import MLDashboard from './components/MLDashboard'
import TrainingLogs from './components/TrainingLogs'
import ReportsPage from './components/ReportsPage'
import ColabPipeline from './components/ColabPipeline'
import DatasetCollector from './components/DatasetCollector'
import AnalysisProgress from './components/AnalysisProgress'
import CameraCapture from './components/CameraCapture'
import YouTubeFrames from './components/YouTubeFrames'
import BatchResults from './components/BatchResults'
import VideoResults from './components/VideoResults'
import { runDetection, pollLlavaResult, checkHealth, findAPI, getApiUrl, runBatchDetection, runVideoDetection } from './services/api'
import { convertToJpeg, SAMPLE_IMAGES } from './services/imageUtils'

export default function App() {
  /* ── Navigation ── */
  const [activePage, setActivePage] = useState('upload')

  /* ── Upload state ── */
  const [selectedFile, setSelectedFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [selectedCrop, setSelectedCrop] = useState('wheat')
  const [areaAcres, setAreaAcres] = useState(1)
  const [growthStage, setGrowthStage] = useState('unknown')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiHealthy, setApiHealthy] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [downloadProgress, setDownloadProgress] = useState(0)

  /* ── Multi-image state ── */
  const [multiFiles, setMultiFiles] = useState([])       // Array of { file, preview, result, error, loading }
  const [multiMode, setMultiMode] = useState(false)
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 })
  const [batchResults, setBatchResults] = useState(null)  // Server batch response { results, summary }
  const [batchProcessing, setBatchProcessing] = useState(false)

  /* ── Video state ── */
  const [videoFile, setVideoFile] = useState(null)
  const [videoMode, setVideoMode] = useState(false)
  const [videoResults, setVideoResults] = useState(null)  // { frames, summary }
  const [videoProcessing, setVideoProcessing] = useState(false)

  /* ── Camera state ── */
  const [showCamera, setShowCamera] = useState(false)

  /* ── Page-wide drag state ── */
  const [pageDragging, setPageDragging] = useState(false)

  /* ── Cumulative stats ── */
  const [scanHistory, setScanHistory] = useState([])
  const [activeScanIndex, setActiveScanIndex] = useState(null)

  useEffect(() => {
    const initAPI = async () => {
      await findAPI()
      try {
        await checkHealth()
        setApiHealthy(true)
      } catch {
        setApiHealthy(false)
      }
    }
    initAPI()
  }, [])

  /* ── File handling (with format conversion) ── */
  const handleFileSelect = async (file) => {
    setMultiMode(false)
    setMultiFiles([])
    setError(null)
    setResult(null)
    try {
      const converted = await convertToJpeg(file)
      setSelectedFile(converted)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(converted)
    } catch {
      setSelectedFile(file)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(file)
    }
  }

  /* ── Multi-file handling ── */
  const handleMultiFileSelect = async (files) => {
    setMultiMode(true)
    setVideoMode(false)
    setVideoFile(null)
    setVideoResults(null)
    setBatchResults(null)
    setSelectedFile(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
    const entries = await Promise.all(
      files.map(async (f) => {
        const converted = await convertToJpeg(f).catch(() => f)
        const preview = await new Promise((resolve) => {
          const reader = new FileReader()
          reader.onload = (e) => resolve(e.target.result)
          reader.readAsDataURL(converted)
        })
        return { file: converted, preview, result: null, error: null, loading: false }
      })
    )
    setMultiFiles(entries)
  }

  /* ── Video file handling ── */
  const handleVideoSelect = (file) => {
    setVideoMode(true)
    setVideoFile(file)
    setVideoResults(null)
    setMultiMode(false)
    setMultiFiles([])
    setBatchResults(null)
    setSelectedFile(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
  }

  /* ── Camera capture handler ── */
  const handleCameraCapture = (file) => {
    setShowCamera(false)
    handleFileSelect(file)
  }

  /* ── Sample image handler ── */
  const handleSampleImage = async (sample) => {
    setError(null)
    setResult(null)
    setMultiMode(false)
    setMultiFiles([])
    try {
      const apiUrl = getApiUrl()
      const resp = await fetch(`${apiUrl}${sample.url}`)
      if (!resp.ok) throw new Error('Sample not available')
      const blob = await resp.blob()
      const file = new File([blob], sample.label.replace(/\s+/g, '_') + '.jpg', { type: 'image/jpeg' })
      setSelectedFile(file)
      setSelectedCrop(sample.crop)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(file)
    } catch {
      // Fallback: create a placeholder to trigger detection with just the label
      setError(`Sample image "${sample.label}" not available from server. Upload your own image.`)
    }
  }

  /* ── Page-wide drag-and-drop ── */
  const handlePageDragOver = useCallback((e) => {
    e.preventDefault()
    setPageDragging(true)
  }, [])
  const handlePageDragLeave = useCallback((e) => {
    e.preventDefault()
    // Only dismiss if leaving the window
    if (e.relatedTarget === null || !e.currentTarget.contains(e.relatedTarget)) {
      setPageDragging(false)
    }
  }, [])
  const handlePageDrop = useCallback((e) => {
    e.preventDefault()
    setPageDragging(false)
    if (activePage !== 'upload') return
    const allFiles = Array.from(e.dataTransfer.files)
    const videos = allFiles.filter(f => f.type.startsWith('video/') || /\.(mp4|avi|mov|webm|mkv)$/i.test(f.name))
    const images = allFiles.filter(
      f => f.type.startsWith('image/') || /\.(heic|heif|bmp|webp)$/i.test(f.name)
    )
    if (videos.length > 0) {
      handleVideoSelect(videos[0])
    } else if (images.length > 1) {
      handleMultiFileSelect(images)
    } else if (images.length === 1) {
      handleFileSelect(images[0])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePage])

  /* ── Run detection automatically after file select ── */
  const handleRunDetection = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setUploadProgress(0)
    setDownloadProgress(0)
    try {
      // Fast path: YOLO + Classifier only (sub-second), LLaVA runs in background
      const res = await runDetection(selectedFile, {
        confidence_threshold: 0.5,
        include_image: true,
        crop_type: selectedCrop,
        use_llava: false,
        area_acres: areaAcres,
        growth_stage: growthStage,
        onProgress: (p) => {
          if (p.type === 'upload') setUploadProgress(p.progress)
          if (p.type === 'download') setDownloadProgress(p.progress)
        },
      })
      setResult(res)
      const scanEntry = {
        id: Date.now(),
        time: new Date(),
        detections: res.detections?.length || 0,
        health: computeHealth(res),
        result: res,
        imagePreview: imagePreview,
        filename: selectedFile.name,
      }
      setScanHistory((prev) => {
        const updated = [...prev, scanEntry]
        setActiveScanIndex(updated.length - 1)
        return updated
      })

      // Poll for background LLaVA result (non-blocking UI update)
      if (res.llava_hash && res.llava_pending) {
        const scanId = scanEntry.id
        pollLlavaResult(res.llava_hash).then((pollData) => {
          if (pollData) {
            const applyLlava = (prev) => {
              if (!prev) return prev
              const updated = { ...prev, llava_analysis: pollData.llava_analysis }
              if (pollData.llm_validation && updated.structured) {
                updated.structured = {
                  ...updated.structured,
                  ai_validation: {
                    agrees: pollData.llm_validation.agrees,
                    agreement_score: pollData.llm_validation.agreement_score,
                    llm_diagnosis: pollData.llm_validation.llm_diagnosis,
                    scenario: pollData.llm_validation.scenario,
                    reasoning_text: pollData.llm_validation.reasoning_text || '',
                    health_score: pollData.llm_validation.health_score,
                    risk_level: pollData.llm_validation.risk_level,
                    model: 'LLaVA',
                  },
                }
              }
              return updated
            }
            setResult(applyLlava)
            // Also update scan history entry
            setScanHistory((prev) =>
              prev.map((s) => (s.id === scanId ? { ...s, result: applyLlava(s.result) } : s))
            )
          }
        })
      }
    } catch (err) {
      setError(err.message || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  /* ── Batch scan all multi-images (server-side batch API) ── */
  const handleScanAll = async () => {
    if (multiFiles.length === 0) return
    setBatchProcessing(true)
    setLoading(true)
    setError(null)
    setUploadProgress(0)
    setDownloadProgress(0)

    try {
      const files = multiFiles.map(e => e.file)
      const res = await runBatchDetection(files, {
        confidence_threshold: 0.5,
        crop_type: selectedCrop,
        area_acres: areaAcres,
        growth_stage: growthStage,
        onProgress: (p) => {
          if (p.type === 'upload') setUploadProgress(p.progress)
          if (p.type === 'download') setDownloadProgress(p.progress)
        },
      })
      setBatchResults(res)

      // Add all to scan history
      for (const r of (res.results || [])) {
        if (!r.rejected) {
          const scanEntry = {
            id: Date.now() + r.index,
            time: new Date(),
            detections: r.num_detections || 0,
            health: r.health_score || 50,
            result: r,
            imagePreview: r.annotated_image || r.original_image,
            filename: r.filename,
          }
          setScanHistory(prev => [...prev, scanEntry])
        }
      }
    } catch (err) {
      setError(err.message || 'Batch detection failed')
    } finally {
      setBatchProcessing(false)
      setLoading(false)
    }
  }

  /* ── Run video detection ── */
  const handleVideoDetection = async () => {
    if (!videoFile) return
    setVideoProcessing(true)
    setLoading(true)
    setError(null)
    setUploadProgress(0)
    setDownloadProgress(0)

    try {
      const res = await runVideoDetection(videoFile, {
        confidence_threshold: 0.5,
        crop_type: selectedCrop,
        frame_interval: 30,
        max_frames: 20,
        onProgress: (p) => {
          if (p.type === 'upload') setUploadProgress(p.progress)
          if (p.type === 'download') setDownloadProgress(p.progress)
        },
      })
      setVideoResults(res)
    } catch (err) {
      setError(err.message || 'Video detection failed')
    } finally {
      setVideoProcessing(false)
      setLoading(false)
    }
  }

  function computeHealth(res) {
    // Use ensemble score if available (most accurate, conservative)
    if (res?.ensemble?.ensemble_health_score != null) {
      return res.ensemble.ensemble_health_score
    }
    if (res?.llava_analysis?.health_score != null) {
      return Math.round(res.llava_analysis.health_score)
    }
    const dets = (res?.detections || []).filter(
      (d) =>
        !['healthy', 'healthy_crop', 'healthy_rice_leaf'].includes(
          d.class_name.toLowerCase().replace(/[\s-]+/g, '_')
        )
    )
    if (dets.length === 0) return 100
    return Math.max(0, Math.round(100 - dets.reduce((s, d) => s + d.confidence * 30, 0)))
  }

  /* ── Export helpers ── */
  const downloadJSON = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detections_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadCSV = () => {
    if (!result?.detections) return
    const hdr = ['Class', 'Confidence', 'Severity', 'X1', 'Y1', 'X2', 'Y2']
    const rows = result.detections.map((d) =>
      [
        d.class_name,
        d.confidence.toFixed(3),
        d.severity_score?.toFixed(3) ?? '',
        d.bbox?.x1?.toFixed(1) ?? '',
        d.bbox?.y1?.toFixed(1) ?? '',
        d.bbox?.x2?.toFixed(1) ?? '',
        d.bbox?.y2?.toFixed(1) ?? '',
      ]
        .map((c) => `"${c}"`)
        .join(',')
    )
    const csv = [hdr.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detections_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  /* ── Computed stats for StatsCards ── */
  const statsData = {
    totalScans: scanHistory.length,
    diseasesFound: scanHistory.reduce((s, h) => s + h.detections, 0),
    avgHealth:
      scanHistory.length > 0
        ? Math.round(scanHistory.reduce((s, h) => s + h.health, 0) / scanHistory.length)
        : null,
    lastScanTime:
      scanHistory.length > 0
        ? scanHistory[scanHistory.length - 1].time.toLocaleTimeString('en-US', { hour12: false })
        : null,
  }

  /* ════════════════════════════════════════════
     PAGE RENDERERS
     ════════════════════════════════════════════ */

  const renderUpload = () => (
    <div className="space-y-6">
      {/* ── Scan History strip (always visible when there are scans) ── */}
      {scanHistory.length > 0 && !batchResults && !videoResults && (
        <ScanHistory
          scans={scanHistory}
          activeIndex={activeScanIndex}
          onSelect={(i) => {
            setActiveScanIndex(i)
            setResult(scanHistory[i].result)
            setImagePreview(scanHistory[i].imagePreview)
            setMultiMode(false)
            setVideoMode(false)
            setBatchResults(null)
            setVideoResults(null)
          }}
        />
      )}

      {/* ── Batch Results View ── */}
      {batchResults ? (
        <BatchResults
          results={batchResults.results}
          summary={batchResults.summary}
          onBack={() => {
            setBatchResults(null)
            setMultiMode(false)
            setMultiFiles([])
            setError(null)
          }}
          onViewDetail={(item) => {
            // View full detail of a batch item by simulating a single-image result structure
            const top5 = item.classifier_top5 || []
            const topPred = top5[0] || {}
            setResult({
              rejected: item.rejected,
              rejection_reason: item.rejection_reason,
              is_plant: !item.rejected,
              face_count: item.face_count || 0,
              green_ratio: item.green_ratio || 0,
              structured: null,
              detections: item.detections || [],
              image: item.original_image,
              annotated_image: item.annotated_image,
              classifier_result: {
                top5,
                top_prediction: item.disease_name || topPred.class_name || 'Unknown',
                top_confidence: item.confidence ?? topPred.confidence ?? 0,
                health_score: item.health_score,
              },
              reasoning: item.treatment ? {
                disease_key: (item.disease_name || '').toLowerCase().replace(/[\s/]+/g, '_'),
                disease_name: item.disease_name || topPred.class_name || 'Unknown',
                confidence: item.confidence ?? topPred.confidence ?? 0,
                treatment: Array.isArray(item.treatment) ? item.treatment : [],
                evidence: Array.isArray(item.evidence) ? item.evidence : [],
                symptoms_matched: [],
                symptoms_detected: [],
                reasoning_chain: [],
                rejections: [],
                differential_diagnosis: [],
                urgency: null,
                yield_loss: null,
              } : null,
              ensemble: {
                ensemble_health_score: item.health_score,
                ensemble_risk_level: item.risk_level,
                models_used: ['Classifier', 'YOLO Detection'],
              },
              processing_time_ms: item.processing_time_ms,
              filename: item.filename,
            })
            setImagePreview(item.annotated_image || item.original_image)
            setBatchResults(null)
            setMultiMode(false)
          }}
        />
      ) : videoResults ? (
        /* ── Video Results View ── */
        <VideoResults
          frames={videoResults.frames}
          summary={videoResults.summary}
          onBack={() => {
            setVideoResults(null)
            setVideoMode(false)
            setVideoFile(null)
            setError(null)
          }}
        />
      ) : !result && !multiMode && !videoMode ? (
        /* ── Pre-upload: drop zone ── */
        <>
          <UploadBox
            onFileSelect={handleFileSelect}
            onMultiFileSelect={handleMultiFileSelect}
            onVideoSelect={handleVideoSelect}
            onCameraClick={() => setShowCamera(true)}
            selectedCrop={selectedCrop}
            setSelectedCrop={setSelectedCrop}
            areaAcres={areaAcres}
            setAreaAcres={setAreaAcres}
            growthStage={growthStage}
            setGrowthStage={setGrowthStage}
            disabled={loading}
          />

          {/* File info + Run button (single file) */}
          {selectedFile && (
            <div className="flex flex-col items-center gap-3">
              {loading ? (
                <div className="w-full max-w-lg">
                  <AnalysisProgress
                    uploadProgress={uploadProgress}
                    downloadProgress={downloadProgress}
                    useLlava={true}
                  />
                </div>
              ) : (
                <>
                  <div className="rounded-lg px-5 py-3 text-center" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <p className="text-sm font-semibold truncate max-w-xs" style={{ color: 'var(--text-primary)' }}>
                      {selectedFile.name}
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB &middot;{' '}
                      <span style={{ color: 'var(--accent)' }}>{selectedCrop.toUpperCase()}</span>
                    </p>
                  </div>
                  <button
                    onClick={handleRunDetection}
                    disabled={loading || !apiHealthy}
                    className="px-8 py-3 rounded-lg text-white font-bold text-sm tracking-wide disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center gap-2"
                    style={{ background: 'var(--accent)', boxShadow: '0 4px 14px color-mix(in srgb, var(--accent) 25%, transparent)' }}
                  >
                    RUN DETECTION
                  </button>
                </>
              )}
              {!apiHealthy && (
                <p className="text-xs" style={{ color: 'var(--danger)' }}>
                  API offline — make sure the backend server is running
                </p>
              )}
              {error && <p className="text-xs" style={{ color: 'var(--danger)' }}>{error}</p>}
            </div>
          )}

          {/* Placeholder when idle — sample images + YouTube */}
          {!selectedFile && (
            <div className="space-y-6">
              <div className="text-center py-4">
                <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
                  Upload images, batch photos, or video to begin analysis
                </p>
              </div>

              {/* Try sample images */}
              <div className="max-w-2xl mx-auto">
                <p className="text-xs font-semibold mb-3 text-center" style={{ color: 'var(--text-muted)' }}>
                  TRY A SAMPLE IMAGE
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {SAMPLE_IMAGES.map((s) => (
                    <button
                      key={s.label}
                      onClick={() => handleSampleImage(s)}
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold transition hover:scale-105"
                      style={{ border: '1px solid var(--border)', color: 'var(--accent)', background: 'var(--bg-card)' }}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* YouTube section */}
              <YouTubeFrames
                selectedCrop={selectedCrop}
                areaAcres={areaAcres}
                growthStage={growthStage}
              />

              {error && <p className="text-xs text-center" style={{ color: 'var(--danger)' }}>{error}</p>}
            </div>
          )}
        </>
      ) : videoMode ? (
        /* ── Video mode: file selected, waiting to process ── */
        <>
          <div className="flex items-center justify-between">
            <button
              onClick={() => { setVideoMode(false); setVideoFile(null); setError(null) }}
              className="px-4 py-2 rounded-lg text-xs font-semibold"
              style={{ border: '1px solid var(--border)', color: 'var(--accent)' }}
            >
              ← Back
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: 'color-mix(in srgb, var(--accent) 10%, transparent)' }}>
                <span className="text-xl">🎥</span>
              </div>
              <div>
                <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{videoFile?.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
                  {((videoFile?.size || 0) / 1024 / 1024).toFixed(1)} MB &middot; {selectedCrop.toUpperCase()}
                </p>
              </div>
            </div>
            <button
              onClick={handleVideoDetection}
              disabled={videoProcessing || !apiHealthy}
              className="px-6 py-2.5 rounded-lg text-white text-sm font-bold disabled:opacity-40"
              style={{ background: 'var(--accent)' }}
            >
              {videoProcessing ? 'Processing...' : 'Analyze Video'}
            </button>
          </div>

          {videoProcessing && (
            <div className="w-full max-w-lg mx-auto">
              <AnalysisProgress
                uploadProgress={uploadProgress}
                downloadProgress={downloadProgress}
                useLlava={false}
              />
            </div>
          )}

          {!videoProcessing && (
            <div className="rounded-xl p-8 text-center" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
              <span className="text-5xl mb-4 block">🎬</span>
              <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Video ready for analysis
              </p>
              <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
                Frames will be extracted at ~1/sec intervals and each analyzed for diseases.
                <br />Annotated frames with bounding boxes will show exactly what was detected.
              </p>
            </div>
          )}

          {error && <p className="text-xs text-center" style={{ color: 'var(--danger)' }}>{error}</p>}
        </>
      ) : multiMode ? (
        /* ── Multi-image mode ── */
        <>
          <div className="flex items-center justify-between">
            <button
              onClick={() => { setMultiMode(false); setMultiFiles([]); setBatchResults(null); setError(null) }}
              className="px-4 py-2 rounded-lg text-xs font-semibold"
              style={{ border: '1px solid var(--border)', color: 'var(--accent)' }}
            >
              ← Back
            </button>
            <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              {multiFiles.length} images selected
            </p>
            <button
              onClick={handleScanAll}
              disabled={batchProcessing || !apiHealthy}
              className="px-6 py-2.5 rounded-lg text-white text-sm font-bold disabled:opacity-40"
              style={{ background: 'var(--accent)' }}
            >
              {batchProcessing ? 'Analyzing All...' : 'Analyze All'}
            </button>
          </div>

          {/* Upload progress */}
          {batchProcessing && (
            <div className="w-full max-w-lg mx-auto">
              <AnalysisProgress
                uploadProgress={uploadProgress}
                downloadProgress={downloadProgress}
                useLlava={false}
              />
            </div>
          )}

          {/* Preview grid */}
          {!batchProcessing && (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {multiFiles.map((entry, i) => (
                <div key={i} className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}>
                  <div className="relative aspect-square">
                    <img src={entry.preview} alt={entry.file.name} className="w-full h-full object-cover" />
                    <div className="absolute top-1 left-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
                      style={{ background: 'rgba(0,0,0,0.6)' }}>
                      {i + 1}
                    </div>
                  </div>
                  <div className="p-2">
                    <p className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{entry.file.name}</p>
                    <p className="text-[9px]" style={{ color: 'var(--text-faint)' }}>{(entry.file.size / 1024).toFixed(0)} KB</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {error && <p className="text-xs text-center" style={{ color: 'var(--danger)' }}>{error}</p>}
        </>
      ) : (
        /* ── Post-upload: results ── */
        <>
          <div className="flex justify-center gap-3 mb-4">
            <button
              onClick={() => {
                setResult(null)
                setSelectedFile(null)
                setImagePreview(null)
                setError(null)
                setActiveScanIndex(null)
                setMultiMode(false)
                setMultiFiles([])
                setVideoMode(false)
                setVideoFile(null)
                setVideoResults(null)
                setBatchResults(null)
              }}
              className="px-6 py-2 rounded-lg text-sm font-semibold transition"
              style={{ border: '1px solid var(--border)', color: 'var(--accent)' }}
            >
              ← New Scan
            </button>
          </div>
          <ResultViewer
            result={result}
            imagePreview={imagePreview}
            onDownloadJSON={downloadJSON}
            onDownloadCSV={downloadCSV}
            cropType={selectedCrop}
          />
        </>
      )}
    </div>
  )

  const renderLive = () => <LiveStream />

  const renderFieldMap = () => (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-20 h-20 rounded-full flex items-center justify-center" style={{ background: 'color-mix(in srgb, var(--accent) 10%, transparent)', border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)' }}>
        <svg className="w-10 h-10" style={{ color: 'var(--accent)', opacity: 0.4 }} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z" />
        </svg>
      </div>
      <h3 className="text-lg font-semibold" style={{ color: 'var(--text-muted)' }}>Field Map</h3>
      <p className="text-sm" style={{ color: 'var(--text-faint)' }}>Prescription maps from drone surveys — coming soon</p>
    </div>
  )

  const renderReports = () => <ReportsPage />

  const renderDataset = () => (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-20 h-20 rounded-full flex items-center justify-center" style={{ background: 'color-mix(in srgb, var(--accent) 10%, transparent)', border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)' }}>
        <svg className="w-10 h-10" style={{ color: 'var(--accent)', opacity: 0.4 }} fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
        </svg>
      </div>
      <h3 className="text-lg font-semibold" style={{ color: 'var(--text-muted)' }}>Dataset Collector</h3>
      <p className="text-sm" style={{ color: 'var(--text-faint)' }}>Automated data collection pipeline — coming soon</p>
    </div>
  )

  const renderPage = () => {
    switch (activePage) {
      case 'upload':       return renderUpload()
      case 'live':         return renderLive()
      case 'fieldmap':     return renderFieldMap()
      case 'reports':      return renderReports()
      case 'mldashboard':  return <MLDashboard />
      case 'logs':         return <TrainingLogs />
      case 'pipeline':     return <ColabPipeline />
      case 'dataset':      return <DatasetCollector />
      case 'livesessions': return <LiveSessions />
      case 'qrconnect':    return <QRConnect />
      case 'activity':     return <ActivityFeed />
      default:             return renderUpload()
    }
  }

  /* ════════════════════════════════════════════
     LAYOUT
     ════════════════════════════════════════════ */
  return (
    <div
      className="min-h-screen"
      style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
      onDragOver={handlePageDragOver}
      onDragLeave={handlePageDragLeave}
      onDrop={handlePageDrop}
    >
      <Navbar />
      <Sidebar activePage={activePage} setActivePage={setActivePage} />

      {/* Main content — offset for navbar (h-14) + sidebar (w-220) */}
      <main className="ml-[220px] pt-14 min-h-screen">
        {/* Stats bar */}
        <div className="px-5 pt-4">
          <StatsCards stats={statsData} />
        </div>

        {/* Page content */}
        <div className="px-5 py-6">{renderPage()}</div>
      </main>

      {/* Page-wide drag overlay */}
      {pageDragging && (
        <div
          className="fixed inset-0 z-40 flex items-center justify-center pointer-events-none"
          style={{ background: 'rgba(0,0,0,0.6)' }}
        >
          <div className="rounded-2xl border-4 border-dashed p-12 text-center"
            style={{ borderColor: 'var(--accent)', background: 'rgba(0,0,0,0.4)' }}
          >
            <p className="text-2xl font-bold text-white mb-2">Drop images or video anywhere</p>
            <p className="text-sm text-white/70">Images: JPG, PNG, WebP, HEIC &middot; Video: MP4, AVI, MOV</p>
          </div>
        </div>
      )}

      {/* Camera modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  )
}
