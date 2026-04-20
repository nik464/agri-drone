import React, { useRef, useState, useCallback, useEffect } from 'react'
import { getApiUrl } from '../services/api'

// Category colors (RGB) matching backend palette
const CATEGORY_COLORS = {
  health: '#00c800',
  disease: '#ff6600',
  weed: '#ff3232',
  lodging: '#ff6400',
  nutrient: '#ffff00',
  stress: '#00c8ff',
  stand: '#c8c800',
}

const CLASS_TO_CATEGORY = {
  healthy_crop: 'health',
  wheat_lodging: 'lodging',
  volunteer_corn: 'weed',
  broadleaf_weed: 'weed',
  grass_weed: 'weed',
  leaf_rust: 'disease',
  goss_wilt: 'disease',
  fusarium_head_blight: 'disease',
  nitrogen_deficiency: 'nutrient',
  water_stress: 'stress',
  good_plant_spacing: 'stand',
  poor_plant_spacing: 'stand',
}

function getCategoryColor(className) {
  const cat = CLASS_TO_CATEGORY[className] || 'health'
  return CATEGORY_COLORS[cat] || '#ffffff'
}

// SVG mini line chart for rolling detection counts
function RollingChart({ history, height = 120 }) {
  const categories = Object.keys(CATEGORY_COLORS)
  const width = 360

  if (history.length < 2) {
    return (
      <div className="rounded-lg p-4 flex items-center justify-center" style={{ height, background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>Waiting for frames…</p>
      </div>
    )
  }

  // Build per-category series from history
  const series = {}
  for (const cat of categories) {
    series[cat] = history.map((h) => h.by_category?.[cat] ?? 0)
  }

  const maxVal = Math.max(
    1,
    ...categories.flatMap((c) => series[c]),
  )

  const padX = 32
  const padY = 12
  const chartW = width - padX * 2
  const chartH = height - padY * 2

  const toX = (i) => padX + (i / (history.length - 1)) * chartW
  const toY = (v) => padY + chartH - (v / maxVal) * chartH

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
      {/* grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = padY + chartH - f * chartH
        return (
          <g key={f}>
            <line x1={padX} y1={y} x2={width - padX} y2={y} stroke="rgba(255,255,255,0.08)" />
            <text x={padX - 4} y={y + 4} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="9">
              {Math.round(maxVal * f)}
            </text>
          </g>
        )
      })}

      {/* category lines */}
      {categories.map((cat) => {
        const pts = series[cat]
        if (pts.every((v) => v === 0)) return null
        const d = pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')
        return <path key={cat} d={d} fill="none" stroke={CATEGORY_COLORS[cat]} strokeWidth="1.5" opacity="0.85" />
      })}
    </svg>
  )
}

// ══════════════════════════════════════════════════════════
// Analysis Result Sub-Components
// ══════════════════════════════════════════════════════════

function InfoRow({ label, value, color }) {
  if (!value || value === 'null' || value === 'unknown') return null
  return (
    <div className="flex items-start justify-between gap-2 py-1" style={{ borderBottom: '1px solid var(--border)' }}>
      <span className="text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="text-[10px] font-semibold text-right" style={{ color: color || 'var(--text-primary)' }}>{String(value)}</span>
    </div>
  )
}

function TagList({ items, color = '#3b82f6' }) {
  if (!items || !Array.isArray(items) || items.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item, i) => (
        <span key={i} className="text-[9px] font-semibold px-1.5 py-0.5 rounded" style={{
          color, background: `${color}12`, border: `1px solid ${color}25`
        }}>{item}</span>
      ))}
    </div>
  )
}

function ObjectView({ data }) {
  if (!data) return null
  return (
    <div className="space-y-2">
      {data.object_name && (
        <div>
          <p className="text-base font-bold" style={{ color: '#3b82f6' }}>{data.object_name}</p>
          {data.category && <span className="text-[9px] px-1.5 py-0.5 rounded font-bold" style={{ color: '#22c55e', background: 'rgba(34,197,94,0.1)' }}>{data.category}</span>}
        </div>
      )}
      <InfoRow label="Brand" value={data.brand} color="#f59e0b" />
      <InfoRow label="Model" value={data.model} />
      <InfoRow label="Condition" value={data.condition} color={data.condition === 'new' ? '#22c55e' : data.condition === 'damaged' ? '#ef4444' : '#eab308'} />
      <InfoRow label="Color" value={data.color} />
      <InfoRow label="Material" value={data.material} />
      <InfoRow label="Text Visible" value={data.text_visible} color="#a78bfa" />
      <InfoRow label="Use Case" value={data.use_case} />
      <InfoRow label="Confidence" value={data.confidence} />
      {data.notable_features && <TagList items={data.notable_features} />}
      {data.description && (
        <p className="text-[10px] leading-relaxed mt-2 p-2 rounded-lg" style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>{data.description}</p>
      )}
    </div>
  )
}

function PlantView({ data }) {
  if (!data) return null
  const scoreColor = (data.health_score || 50) >= 70 ? '#22c55e' : (data.health_score || 50) >= 40 ? '#eab308' : '#ef4444'
  return (
    <div className="space-y-2">
      {data.plant_species && (
        <div>
          <p className="text-base font-bold" style={{ color: '#22c55e' }}>{data.plant_species}</p>
          {data.common_name && <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{data.common_name}</p>}
        </div>
      )}
      <div className="flex items-center gap-3">
        {data.health_score != null && (
          <div className="text-center">
            <span className="text-2xl font-black" style={{ color: scoreColor }}>{data.health_score}</span>
            <p className="text-[8px] uppercase" style={{ color: 'var(--text-faint)' }}>Health</p>
          </div>
        )}
        <div className="flex-1 space-y-0.5">
          <InfoRow label="Status" value={data.health_status} color={scoreColor} />
          <InfoRow label="Growth Stage" value={data.growth_stage} />
          <InfoRow label="Severity" value={data.overall_severity != null ? `${data.overall_severity}%` : null} />
        </div>
      </div>
      {data.diseases_detected?.length > 0 && (
        <div className="space-y-1">
          <span className="text-[9px] uppercase font-semibold" style={{ color: 'var(--text-faint)' }}>Diseases</span>
          {data.diseases_detected.map((d, i) => (
            <div key={i} className="px-2 py-1.5 rounded-lg" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}>
              <div className="flex justify-between">
                <span className="text-[10px] font-semibold" style={{ color: '#ef4444' }}>{d.name}</span>
                <span className="text-[9px]" style={{ color: 'var(--text-faint)' }}>{d.confidence} · {d.severity_pct}%</span>
              </div>
              {d.symptoms && <p className="text-[9px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{d.symptoms}</p>}
            </div>
          ))}
        </div>
      )}
      <InfoRow label="Nutrients" value={data.nutrient_status} />
      <TagList items={data.nutrient_issues} color="#eab308" />
      <TagList items={data.environmental_stress} color="#06b6d4" />
      {data.recommendations?.length > 0 && (
        <div className="space-y-1 mt-1">
          <span className="text-[9px] uppercase font-semibold" style={{ color: 'var(--text-faint)' }}>Treatment</span>
          {data.recommendations.map((r, i) => (
            <p key={i} className="text-[10px] pl-2" style={{ color: '#22c55e', borderLeft: '2px solid rgba(34,197,94,0.3)' }}>• {r}</p>
          ))}
        </div>
      )}
      {data.description && (
        <p className="text-[10px] leading-relaxed p-2 rounded-lg" style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>{data.description}</p>
      )}
    </div>
  )
}

function PersonView({ data }) {
  if (!data) return null
  return (
    <div className="space-y-2">
      <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>Scene Description</p>
      <InfoRow label="Activity" value={data.activity} />
      <InfoRow label="Setting" value={data.setting} />
      {data.objects_nearby && <TagList items={data.objects_nearby} color="#f59e0b" />}
      {data.description && (
        <p className="text-[10px] leading-relaxed p-2 rounded-lg" style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>{data.description}</p>
      )}
    </div>
  )
}

function DeepSceneView({ data }) {
  if (!data) return null
  return (
    <div className="space-y-3">
      {data.scene_type && (
        <div>
          <p className="text-sm font-bold" style={{ color: '#a78bfa' }}>{data.scene_type}</p>
          {data.scene_description && <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{data.scene_description}</p>}
        </div>
      )}

      {/* Objects list */}
      {data.objects?.length > 0 && (
        <div>
          <span className="text-[9px] uppercase font-semibold" style={{ color: 'var(--text-faint)' }}>Objects Found ({data.objects.length})</span>
          <div className="mt-1 space-y-1">
            {data.objects.slice(0, 10).map((obj, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1 rounded" style={{ background: 'var(--bg-secondary)' }}>
                <span className="text-[10px] font-semibold" style={{ color: 'var(--text-primary)' }}>{obj.name}</span>
                {obj.position && <span className="text-[9px]" style={{ color: 'var(--text-faint)' }}>({obj.position})</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {data.people_count > 0 && <InfoRow label="People" value={`${data.people_count} — ${data.people_activity || 'present'}`} />}

      {/* Text found */}
      {data.text_found?.length > 0 && (
        <div>
          <span className="text-[9px] uppercase font-semibold" style={{ color: 'var(--text-faint)' }}>Text Detected</span>
          <TagList items={data.text_found} color="#f59e0b" />
        </div>
      )}

      <InfoRow label="Lighting" value={data.lighting} />
      <TagList items={data.dominant_colors} color="#6b7280" />

      {/* Agricultural */}
      {data.agricultural_elements?.present && (
        <div className="p-2 rounded-lg" style={{ background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
          <span className="text-[9px] uppercase font-semibold" style={{ color: '#22c55e' }}>Agricultural Elements</span>
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-primary)' }}>{data.agricultural_elements.details}</p>
          {data.agricultural_elements.health_assessment && (
            <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>Health: {data.agricultural_elements.health_assessment}</p>
          )}
        </div>
      )}

      {/* Concerns */}
      {data.concerns?.length > 0 && (
        <div>
          <span className="text-[9px] uppercase font-semibold" style={{ color: '#ef4444' }}>Concerns</span>
          {data.concerns.map((c, i) => (
            <p key={i} className="text-[10px] mt-0.5" style={{ color: '#ef4444' }}>⚠ {c}</p>
          ))}
        </div>
      )}

      {/* Recommendations */}
      {data.recommendations?.length > 0 && (
        <div>
          <span className="text-[9px] uppercase font-semibold" style={{ color: 'var(--text-faint)' }}>Recommendations</span>
          {data.recommendations.map((r, i) => (
            <p key={i} className="text-[10px] pl-2 mt-0.5" style={{ color: '#22c55e', borderLeft: '2px solid rgba(34,197,94,0.3)' }}>• {r}</p>
          ))}
        </div>
      )}
    </div>
  )
}

export default function LiveStream() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const ytFrameRef = useRef(null)
  const wsRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const fpsTimestamps = useRef([])
  const objectUrlRef = useRef(null)
  const sourceModeRef = useRef('webcam')
  const detectionsRef = useRef([])  // store latest detections for click handling
  const lastFrameRef = useRef(null) // store last frame data URL for deep analyze

  const [streaming, setStreaming] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [fps, setFps] = useState(0)
  const [detectionCount, setDetectionCount] = useState(0)
  const [processingMs, setProcessingMs] = useState(0)
  const [framesSent, setFramesSent] = useState(0)
  const [history, setHistory] = useState([]) // last 30 stats
  const [error, setError] = useState(null)
  const [sourceMode, setSourceMode] = useState('webcam') // 'webcam' | 'video' | 'youtube'
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [videoFile, setVideoFile] = useState(null)

  // ── Universal Analyzer state ──
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [analysisType, setAnalysisType] = useState(null) // 'object' | 'deep'
  const [selectedDetection, setSelectedDetection] = useState(null)
  const [deepAnalyzing, setDeepAnalyzing] = useState(false)

  // Cleanup on unmount
  useEffect(() => {
    return () => stopStream()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const captureAndSend = useCallback(() => {
    const video = videoRef.current
    const ws = wsRef.current
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return
    if (video.videoWidth === 0) return

    const offscreen = document.createElement('canvas')
    offscreen.width = video.videoWidth
    offscreen.height = video.videoHeight
    const ctx = offscreen.getContext('2d')
    ctx.drawImage(video, 0, 0)
    const dataUrl = offscreen.toDataURL('image/jpeg', 0.7)

    lastFrameRef.current = dataUrl  // store for deep analyze
    ws.send(dataUrl)
    setFramesSent((prev) => prev + 1)
  }, [])

  const drawDetections = useCallback((detections) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    detectionsRef.current = detections  // save for click handler

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    for (const det of detections) {
      const { x1, y1, x2, y2 } = det.bbox
      const color = getCategoryColor(det.class_name)
      const isSelected = selectedDetection?.id === det.id

      // Highlight selected box
      if (isSelected) {
        ctx.shadowColor = color
        ctx.shadowBlur = 12
      }

      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 3 : 2
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      ctx.shadowBlur = 0

      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = '12px sans-serif'
      const tm = ctx.measureText(label)
      const lh = 16
      ctx.fillStyle = color
      ctx.fillRect(x1, Math.max(y1 - lh, 0), tm.width + 6, lh)
      ctx.fillStyle = '#000'
      ctx.fillText(label, x1 + 3, Math.max(y1 - 3, lh - 3))
    }
  }, [selectedDetection])

  const stopStream = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
      videoRef.current.src = ''
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    if (ytFrameRef.current) {
      ytFrameRef.current.src = ''
    }
    setStreaming(false)
    setConnecting(false)
  }, [])

  // ── Click on canvas to select a detected object ──
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video || detectionsRef.current.length === 0) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = video.videoWidth / rect.width
    const scaleY = video.videoHeight / rect.height
    const clickX = (e.clientX - rect.left) * scaleX
    const clickY = (e.clientY - rect.top) * scaleY

    // Find which detection was clicked
    const hit = detectionsRef.current.find((det) => {
      const { x1, y1, x2, y2 } = det.bbox
      return clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2
    })

    if (hit) {
      setSelectedDetection(hit)
      analyzeObject(hit)
    }
  }, [])

  // ── Analyze a single detected object ──
  const analyzeObject = useCallback(async (det) => {
    setAnalyzing(true)
    setAnalysisResult(null)
    setAnalysisType('object')

    try {
      // Crop the object from current frame
      const video = videoRef.current
      if (!video) return

      const offscreen = document.createElement('canvas')
      const { x1, y1, x2, y2 } = det.bbox
      // Add 10% padding
      const pad = Math.max((x2 - x1), (y2 - y1)) * 0.1
      const cx1 = Math.max(0, x1 - pad)
      const cy1 = Math.max(0, y1 - pad)
      const cx2 = Math.min(video.videoWidth, x2 + pad)
      const cy2 = Math.min(video.videoHeight, y2 + pad)

      offscreen.width = cx2 - cx1
      offscreen.height = cy2 - cy1
      const ctx = offscreen.getContext('2d')
      ctx.drawImage(video, cx1, cy1, cx2 - cx1, cy2 - cy1, 0, 0, cx2 - cx1, cy2 - cy1)
      const croppedB64 = offscreen.toDataURL('image/jpeg', 0.85)

      const apiUrl = getApiUrl()
      const resp = await fetch(`${apiUrl}/api/universal/analyze-object`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: croppedB64,
          object_class: det.class_name,
          bbox: det.bbox,
        }),
      })

      if (!resp.ok) throw new Error(`Analysis failed: ${resp.status}`)
      const data = await resp.json()
      setAnalysisResult(data)
    } catch (err) {
      setAnalysisResult({ error: err.message })
    } finally {
      setAnalyzing(false)
    }
  }, [])

  // ── Deep Analyze full frame ──
  const handleDeepAnalyze = useCallback(async () => {
    setDeepAnalyzing(true)
    setAnalysisResult(null)
    setAnalysisType('deep')
    setSelectedDetection(null)

    try {
      let frameB64 = lastFrameRef.current
      if (!frameB64) {
        // Capture current frame
        const video = videoRef.current
        if (!video) return
        const offscreen = document.createElement('canvas')
        offscreen.width = video.videoWidth
        offscreen.height = video.videoHeight
        offscreen.getContext('2d').drawImage(video, 0, 0)
        frameB64 = offscreen.toDataURL('image/jpeg', 0.85)
      }

      const apiUrl = getApiUrl()
      const resp = await fetch(`${apiUrl}/api/universal/deep-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: frameB64 }),
      })

      if (!resp.ok) throw new Error(`Deep analysis failed: ${resp.status}`)
      const data = await resp.json()
      setAnalysisResult(data)
    } catch (err) {
      setAnalysisResult({ error: err.message })
    } finally {
      setDeepAnalyzing(false)
    }
  }, [])

  const startStream = useCallback(async () => {
    setError(null)
    setConnecting(true)
    setFramesSent(0)
    sourceModeRef.current = sourceMode

    const video = videoRef.current

    if (sourceMode === 'webcam') {
      let mediaStream
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
      } catch {
        setError('Camera access denied or unavailable.')
        setConnecting(false)
        return
      }
      streamRef.current = mediaStream
      if (video) {
        video.srcObject = mediaStream
        await video.play()
      }

    } else if (sourceMode === 'video') {
      if (!videoFile) {
        setError('Please select a video file first.')
        setConnecting(false)
        return
      }
      const objectUrl = URL.createObjectURL(videoFile)
      objectUrlRef.current = objectUrl
      if (video) {
        video.srcObject = null
        video.src = objectUrl
        video.loop = true
        try {
          await video.play()
        } catch {
          setError('Cannot play this video file in the browser.')
          setConnecting(false)
          URL.revokeObjectURL(objectUrl)
          objectUrlRef.current = null
          return
        }
      }

    } else if (sourceMode === 'youtube') {
      if (!youtubeUrl.trim()) {
        setError('Please enter a YouTube URL.')
        setConnecting(false)
        return
      }
    }

    // Connect WebSocket — derive ws:// URL from the discovered HTTP API URL
    const apiUrl = getApiUrl()  // e.g. "http://localhost:9000"
    const wsBase = apiUrl.replace(/^http/, 'ws')
    const wsEndpoint = sourceMode === 'youtube'
      ? `${wsBase}/api/stream/video`
      : `${wsBase}/api/stream/live`
    const ws = new WebSocket(wsEndpoint)
    wsRef.current = ws

    ws.onopen = () => {
      setConnecting(false)
      setStreaming(true)
      fpsTimestamps.current = []
      if (sourceMode === 'youtube') {
        ws.send(JSON.stringify({ source: youtubeUrl.trim(), type: 'youtube' }))
      } else {
        timerRef.current = setInterval(captureAndSend, 200)
      }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.error) {
          setError(`Stream error: ${data.error}`)
          console.warn('Stream error:', data.error)
          return
        }

        const now = performance.now()
        fpsTimestamps.current.push(now)
        fpsTimestamps.current = fpsTimestamps.current.filter((t) => now - t < 2000)
        setFps(Math.round(fpsTimestamps.current.length / 2))

        setDetectionCount(data.stats?.total ?? 0)
        setProcessingMs(data.processing_ms ?? 0)
        setHistory((prev) => [...prev, data.stats].slice(-30))

        if (sourceModeRef.current === 'youtube') {
          if (data.annotated_frame_b64 && ytFrameRef.current) {
            ytFrameRef.current.src = `data:image/jpeg;base64,${data.annotated_frame_b64}`
          }
          setFramesSent((prev) => prev + 1)
        } else {
          if (data.detections) {
            drawDetections(data.detections)
          }
        }
      } catch {
        // ignore malformed messages
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection failed. Is the backend running?')
      setConnecting(false)
      stopStream()
    }

    ws.onclose = () => {
      setStreaming(false)
    }
  }, [sourceMode, videoFile, youtubeUrl, captureAndSend, drawDetections, stopStream])

  const cardStyle = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--card-shadow)',
  }

  const sourceTabs = [
    { id: 'webcam', label: '📷 Webcam' },
    { id: 'video', label: '🎬 Video File' },
    { id: 'youtube', label: '▶️ YouTube' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-faint)' }}>Real-Time</p>
          <h2 className="text-xl font-bold mt-1" style={{ color: 'var(--text-primary)' }}>Live Drone Stream</h2>
        </div>
        <div className="flex items-center gap-3">
          {/* Deep Analyze button */}
          {streaming && (
            <button
              onClick={handleDeepAnalyze}
              disabled={deepAnalyzing}
              className="px-4 py-2 text-xs font-bold rounded-lg transition disabled:opacity-50 flex items-center gap-1.5"
              style={{ background: '#a78bfa', color: '#fff', boxShadow: '0 2px 10px rgba(167,139,250,0.3)' }}
            >
              {deepAnalyzing ? (
                <><span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span> Analyzing…</>
              ) : (
                <><span>🔬</span> Deep Analyze</>
              )}
            </button>
          )}
          {streaming ? (
            <button onClick={stopStream} className="px-5 py-2 text-xs font-bold rounded-lg transition" style={{ border: '1px solid rgba(239,68,68,0.4)', color: '#ef4444', background: 'rgba(239,68,68,0.1)' }}>
              ⏹ Stop Stream
            </button>
          ) : (
            <button
              onClick={startStream}
              disabled={connecting}
              className="px-5 py-2 text-xs font-bold rounded-lg text-white transition disabled:opacity-50"
              style={{ background: 'var(--accent)' }}
            >
              {connecting ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  Connecting…
                </span>
              ) : (
                '▶ Start Stream'
              )}
            </button>
          )}
        </div>
      </div>

      {/* Click-to-analyze hint */}
      {streaming && (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px]" style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.2)', color: '#a78bfa' }}>
          <span>💡</span>
          <span><strong>Click any bounding box</strong> to identify the object with AI &middot; <strong>Deep Analyze</strong> scans the entire frame</span>
        </div>
      )}

      {/* Source selector — only visible when not streaming */}
      {!streaming && !connecting && (
        <div className="rounded-xl p-5 space-y-4" style={cardStyle}>
          <h3 className="font-semibold text-sm" style={{ color: 'var(--accent)' }}>Video Source</h3>
          <div className="flex gap-2">
            {sourceTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSourceMode(tab.id)}
                className="px-4 py-2 rounded-lg text-xs font-semibold transition"
                style={{
                  background: sourceMode === tab.id ? 'var(--accent)' : 'var(--bg-primary)',
                  color: sourceMode === tab.id ? '#fff' : 'var(--text-muted)',
                  border: `1px solid ${sourceMode === tab.id ? 'var(--accent)' : 'var(--border)'}`,
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {sourceMode === 'video' && (
            <div className="space-y-2">
              <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Select a local video file (MP4, WebM, MOV…)</label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
                className="block w-full text-xs rounded-lg px-3 py-2 cursor-pointer"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
              />
              {videoFile && <p className="text-xs" style={{ color: '#22c55e' }}>✓ {videoFile.name}</p>}
            </div>
          )}

          {sourceMode === 'youtube' && (
            <div className="space-y-2">
              <label className="text-xs" style={{ color: 'var(--text-muted)' }}>YouTube URL (processed server-side with yt-dlp)</label>
              <input
                type="url"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                className="block w-full text-xs rounded-lg px-3 py-2"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
              <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>Requires yt-dlp installed on the backend server.</p>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-lg p-4" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <p className="text-sm" style={{ color: '#ef4444' }}>❌ {error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video + overlay */}
        <div className="lg:col-span-2">
          <div className="rounded-xl overflow-hidden relative" style={{ ...cardStyle, aspectRatio: '4/3' }}>
            {/* Webcam / Video file mode */}
            <video
              ref={videoRef}
              muted
              playsInline
              className="w-full h-full object-contain bg-black/50"
              style={{ display: (streaming && sourceMode !== 'youtube') ? 'block' : 'none' }}
            />
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className="absolute inset-0 w-full h-full cursor-crosshair"
              style={{ display: (streaming && sourceMode !== 'youtube') ? 'block' : 'none', objectFit: 'contain' }}
            />

            {/* YouTube mode: backend streams annotated frames */}
            <img
              ref={ytFrameRef}
              alt="annotated stream"
              className="w-full h-full object-contain bg-black"
              style={{ display: (streaming && sourceMode === 'youtube') ? 'block' : 'none' }}
            />

            {!streaming && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center space-y-3">
                  <div className="text-6xl">
                    {sourceMode === 'webcam' ? '📷' : sourceMode === 'video' ? '🎬' : '▶️'}
                  </div>
                  <p style={{ color: 'var(--text-muted)' }}>Click <span style={{ color: 'var(--accent)' }} className="font-semibold">Start Stream</span> to begin live detection</p>
                  <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
                    {sourceMode === 'webcam' && 'Uses webcam to simulate drone camera feed'}
                    {sourceMode === 'video' && 'Plays a local video file through YOLO detection'}
                    {sourceMode === 'youtube' && 'Streams a YouTube video through YOLO on the server'}
                  </p>
                </div>
              </div>
            )}

            {/* Live indicator */}
            {streaming && (
              <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/80 text-xs font-bold text-white">
                <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                LIVE
              </div>
            )}
          </div>
        </div>

        {/* Stats + Analysis sidebar */}
        <div className="space-y-4">

          {/* ══ Universal AI Analysis Panel ══ */}
          {(analysisResult || analyzing || deepAnalyzing) && (
            <div className="rounded-xl overflow-hidden" style={cardStyle}>
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2">
                  <span className="text-sm">{analysisType === 'deep' ? '🔬' : '🎯'}</span>
                  <span className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
                    {analysisType === 'deep' ? 'Deep Scene Analysis' : 'Object Analysis'}
                  </span>
                  {analysisResult?.analysis_type && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-bold" style={{
                      color: '#a78bfa', background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)'
                    }}>
                      {analysisResult.analysis_type.replace('_', ' ').toUpperCase()}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => { setAnalysisResult(null); setSelectedDetection(null); setAnalysisType(null) }}
                  className="w-6 h-6 rounded-full flex items-center justify-center text-xs"
                  style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}
                >✕</button>
              </div>

              {/* Loading state */}
              {(analyzing || deepAnalyzing) && !analysisResult && (
                <div className="px-4 py-8 flex flex-col items-center gap-3">
                  <div className="w-8 h-8 border-3 border-purple-400/30 border-t-purple-400 rounded-full animate-spin"></div>
                  <p className="text-xs" style={{ color: '#a78bfa' }}>
                    {analysisType === 'deep' ? 'Analyzing entire scene with LLaVA...' : `Identifying ${selectedDetection?.class_name || 'object'}...`}
                  </p>
                  <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>LLaVA on CPU — may take 30-60 seconds</p>
                </div>
              )}

              {/* Error */}
              {analysisResult?.error && (
                <div className="px-4 py-4">
                  <p className="text-xs" style={{ color: '#ef4444' }}>❌ {analysisResult.error}</p>
                </div>
              )}

              {/* Results content */}
              {analysisResult && !analysisResult.error && (
                <div className="px-4 py-3 space-y-3 overflow-y-auto" style={{ maxHeight: '55vh' }}>
                  {/* Processing time */}
                  {analysisResult.processing_ms && (
                    <div className="flex items-center gap-2 text-[10px]" style={{ color: 'var(--text-faint)' }}>
                      <span>⏱ {(analysisResult.processing_ms / 1000).toFixed(1)}s</span>
                      <span>•</span>
                      <span>Model: {analysisResult.model || 'LLaVA'}</span>
                    </div>
                  )}

                  {/* Render based on analysis type */}
                  {analysisResult.analysis_type === 'deep_scene' && (
                    <DeepSceneView data={analysisResult.result} />
                  )}
                  {analysisResult.analysis_type === 'general_object' && (
                    <ObjectView data={analysisResult.result} />
                  )}
                  {analysisResult.analysis_type === 'plant' && (
                    <PlantView data={analysisResult.result} />
                  )}
                  {analysisResult.analysis_type === 'person' && (
                    <PersonView data={analysisResult.result} />
                  )}

                  {/* Raw text fallback */}
                  {analysisResult.raw_text && !analysisResult.result?.object_name && !analysisResult.result?.scene_type && !analysisResult.result?.plant_species && !analysisResult.result?.activity && (
                    <div className="p-3 rounded-lg text-xs leading-relaxed whitespace-pre-wrap" style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
                      {analysisResult.raw_text}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Metrics */}
          <div className="rounded-xl p-5 space-y-4" style={cardStyle}>
            <h3 className="font-semibold text-sm" style={{ color: 'var(--accent)' }}>Stream Metrics</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <p className="text-2xl font-bold" style={{ color: '#22c55e' }}>{fps}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>FPS</p>
              </div>
              <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <p className="text-2xl font-bold" style={{ color: '#3b82f6' }}>{detectionCount}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Detections</p>
              </div>
              <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <p className="text-2xl font-bold" style={{ color: '#a78bfa' }}>{processingMs.toFixed(0)}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Latency (ms)</p>
              </div>
              <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <p className="text-2xl font-bold" style={{ color: '#eab308' }}>{framesSent}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Frames Sent</p>
              </div>
            </div>
          </div>

          {/* Rolling chart */}
          <div className="rounded-xl p-5 space-y-3" style={cardStyle}>
            <h3 className="font-semibold text-sm" style={{ color: 'var(--accent)' }}>Detections / Frame (last 30)</h3>
            <RollingChart history={history} height={140} />
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
                <div key={cat} className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }}></span>
                  <span className="text-[10px] capitalize" style={{ color: 'var(--text-faint)' }}>{cat}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Connection info */}
          <div className="rounded-xl p-5 space-y-2 text-sm" style={cardStyle}>
            <h3 className="font-semibold text-sm" style={{ color: 'var(--accent)' }}>Connection</h3>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Status</span>
              <span style={{ color: streaming ? '#22c55e' : 'var(--text-faint)' }}>
                {streaming ? 'Connected' : connecting ? 'Connecting…' : 'Disconnected'}
              </span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Source</span>
              <span className="text-xs font-medium" style={{ color: 'var(--text-faint)' }}>
                {sourceMode === 'webcam' ? 'Webcam' : sourceMode === 'video' ? (videoFile?.name ?? 'No file') : 'YouTube'}
              </span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Endpoint</span>
              <span className="font-mono text-xs" style={{ color: 'var(--text-faint)' }}>
                {sourceMode === 'youtube' ? 'ws://.../stream/video' : 'ws://.../stream/live'}
              </span>
            </div>
            <div className="flex justify-between">
              <span style={{ color: 'var(--text-muted)' }}>Capture Rate</span>
              <span style={{ color: 'var(--text-faint)' }}>
                {sourceMode === 'youtube' ? '~10 fps (server)' : '200ms (~5 fps)'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
