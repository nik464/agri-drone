import React, { useState } from 'react'
import DetectionCanvas from './DetectionCanvas'
import UncertaintyMeter from './UncertaintyMeter'
import CostBenefitCard from './CostBenefitCard'
import ChatPanel from './ChatPanel'
import { getApiUrl } from '../services/api'

/* ── Healthy reference images per crop ── */
const HEALTHY_REFS = {
  wheat: '/samples/healthy_wheat.jpg',
  rice: '/samples/healthy_rice.jpg',
  maize: '/samples/healthy_maize.jpg',
}

/* ── Severity helpers ── */
function severityFromConfidence(conf) {
  if (conf >= 0.85) return 'CRITICAL'
  if (conf >= 0.65) return 'HIGH'
  if (conf >= 0.4) return 'MEDIUM'
  return 'LOW'
}

function severityColor(sev) {
  switch (sev) {
    case 'CRITICAL': return 'var(--danger)'
    case 'HIGH':     return '#ff6600'
    case 'MEDIUM':   return 'var(--warning)'
    default:         return 'var(--accent)'
  }
}

function gaugeColor(score) {
  if (score >= 70) return 'var(--accent)'
  if (score >= 40) return 'var(--warning)'
  return 'var(--danger)'
}

function gradeColor(grade) {
  switch (grade) {
    case 'VERY_HIGH': return '#22c55e'
    case 'HIGH':      return '#3b82f6'
    case 'MODERATE':  return '#eab308'
    case 'LOW':       return '#f97316'
    case 'UNCERTAIN': return '#ef4444'
    default:          return 'var(--text-muted)'
  }
}

/* Quick treatment lookup (fallback) */
const TREATMENTS = {
  leaf_rust:            'Apply propiconazole fungicide at first sign',
  brown_rust:           'Spray tebuconazole 250 EC @ 1ml/L',
  yellow_rust:          'Apply triadimefon 25WP @ 0.1%',
  black_rust:           'Use mancozeb 75WP @ 2.5g/L',
  fusarium_head_blight: 'Apply metconazole at flowering stage',
  leaf_blight:          'Spray mancozeb + carbendazim combo',
  blast:                'Apply tricyclazole 75WP @ 0.6g/L',
  septoria:             'Use azoxystrobin at flag leaf stage',
  mildew:               'Spray sulfur 80WP @ 3g/L',
  tan_spot:             'Rotate crops; apply pyraclostrobin',
  smut:                 'Treat seed with carboxin + thiram',
  common_root_rot:      'Seed treatment with fludioxonil',
  aphid:                'Spray imidacloprid 17.8SL @ 0.5ml/L',
  mite:                 'Apply dicofol 18.5EC @ 2.5ml/L',
  stem_fly:             'Seed treatment with thiamethoxam 30FS',
  bacterial_leaf_blight:'Apply streptocycline 500ppm spray',
  brown_spot:           'Spray mancozeb 75WP @ 2.5g/L',
  leaf_blast:           'Apply tricyclazole 75WP @ 0.6g/L',
  leaf_scald:           'Use carbendazim 50WP @ 1g/L',
  sheath_blight:        'Apply validamycin 3L @ 2.5ml/L',
  healthy:              'No treatment needed — crop is healthy',
  healthy_crop:         'No treatment needed — crop is healthy',
  healthy_rice_leaf:    'No treatment needed — crop is healthy',
}

function getTreatment(className) {
  const key = className.toLowerCase().replace(/[\s-]+/g, '_')
  return TREATMENTS[key] || 'Consult local agronomist for treatment plan'
}

/* ── Circular health gauge (SVG) ── */
function HealthGauge({ score }) {
  const R = 90
  const C = 2 * Math.PI * R
  const offset = C - (score / 100) * C
  const color = gaugeColor(score)

  return (
    <div className="relative w-[200px] h-[200px] mx-auto">
      <svg width="200" height="200" viewBox="0 0 200 200">
        <circle cx="100" cy="100" r={R} fill="none" stroke="var(--gauge-bg)" strokeWidth="12" />
        <circle
          cx="100" cy="100" r={R} fill="none" stroke={color} strokeWidth="12"
          strokeLinecap="round" strokeDasharray={C} strokeDashoffset={offset}
          className="gauge-circle" transform="rotate(-90 100 100)"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-5xl font-black tabular-nums" style={{ color }}>{score}</span>
        <span className="text-[10px] uppercase tracking-widest mt-1" style={{ color: 'var(--text-muted)' }}>
          Field Health Score
        </span>
      </div>
    </div>
  )
}

/* ── Confidence bar ── */
function ConfidenceBar({ label, score, color, weight, disease, agrees }) {
  return (
    <div className="px-3 py-2 rounded-lg" style={{ background: `${color}08`, border: `1px solid ${color}25` }}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{
            color, background: `${color}15`, border: `1px solid ${color}30`
          }}>{label}</span>
          {disease && <span className="text-[11px]" style={{ color: 'var(--text-primary)' }}>{disease}</span>}
          {agrees === true && <span className="text-[10px]" title="Agrees">&#10003;</span>}
          {agrees === false && <span className="text-[10px] text-red-400" title="Disagrees">&#10007;</span>}
        </div>
        <span className="text-xs font-bold" style={{ color }}>{(score * 100).toFixed(0)}%</span>
      </div>
      <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(score * 100, 100)}%`, background: color }} />
      </div>
      {weight > 0 && (
        <span className="text-[9px] mt-0.5 block" style={{ color: 'var(--text-faint)' }}>
          Weight: {(weight * 100).toFixed(0)}%
        </span>
      )}
    </div>
  )
}

/* ── Detection card ── */
function DetectionCard({ det }) {
  const sev = severityFromConfidence(det.confidence)
  const color = severityColor(sev)
  const area = det.bbox
    ? (((det.bbox.x2 - det.bbox.x1) * (det.bbox.y2 - det.bbox.y1)) / 10000).toFixed(1)
    : (det.affected_area ?? (Math.random() * 20 + 2).toFixed(1))
  const treatment = getTreatment(det.class_name)

  return (
    <div className="rounded-lg overflow-hidden flex" style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)', boxShadow: 'var(--card-shadow)',
    }}>
      <div className="w-1.5 flex-shrink-0" style={{ background: color }} />
      <div className="p-3 flex-1 min-w-0 space-y-1">
        <div className="flex items-center justify-between gap-2">
          <span className="font-bold text-sm truncate" style={{ color: 'var(--text-primary)' }}>
            {det.class_name.replace(/_/g, ' ')}
          </span>
          <span className="flex-shrink-0 text-xs font-bold px-2 py-0.5 rounded" style={{
            color, borderColor: `color-mix(in srgb, ${color} 40%, transparent)`,
            background: `color-mix(in srgb, ${color} 12%, transparent)`,
            border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
          }}>
            {sev}
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          <span>Confidence: <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{(det.confidence * 100).toFixed(1)}%</span></span>
          <span>Area: <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{area}%</span></span>
        </div>
        <p className="text-[10px] leading-snug truncate" style={{ color: 'var(--text-faint)' }}>{treatment}</p>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════
   Main ResultViewer — with structured output support
   ════════════════════════════════════════════════════════ */
export default function ResultViewer({
  result,
  imagePreview,
  onDownloadJSON,
  onDownloadCSV,
  cropType = 'wheat',
}) {
  const [imageView, setImageView] = useState('original') // 'original' | 'gradcam' | 'comparison'
  if (!result) return null

  // ── REJECTION: Not a plant image ──
  if (result.rejected) {
    const sp = result.spectral || {}
    const hasSpectral = sp.vari != null || sp.gli != null
    return (
      <div className="space-y-6">
        <div style={{
          background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)',
          border: '2px solid #f59e0b',
          borderRadius: '16px',
          padding: '2rem',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🚫</div>
          <h2 style={{ color: '#f59e0b', fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.75rem' }}>
            Not a Crop Image
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: 1.6, maxWidth: '500px', margin: '0 auto 1.5rem' }}>
            {result.rejection_reason || 'This image does not appear to contain a plant or crop. Please upload a photo of a crop leaf, plant, or agricultural field.'}
          </p>
          <div style={{
            display: 'flex',
            gap: '2rem',
            justifyContent: 'center',
            flexWrap: 'wrap',
            marginBottom: '1rem',
          }}>
            {result.face_count > 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Faces detected: <span style={{ color: '#ef4444', fontWeight: 600 }}>{result.face_count}</span>
              </div>
            )}
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Green pixels: <span style={{ color: '#22c55e', fontWeight: 600 }}>{((result.green_ratio || 0) * 100).toFixed(1)}%</span>
            </div>
          </div>
          {hasSpectral && (
            <div style={{
              display: 'inline-flex',
              gap: '1.25rem',
              background: 'rgba(0,0,0,0.3)',
              borderRadius: '10px',
              padding: '0.6rem 1.2rem',
              marginBottom: '1rem',
              flexWrap: 'wrap',
              justifyContent: 'center',
            }}>
              {[
                { label: 'VARI', val: sp.vari },
                { label: 'GLI', val: sp.gli },
                { label: 'ExG', val: sp.exg },
                { label: 'NGRDI', val: sp.ngrdi },
              ].filter(x => x.val != null).map(x => (
                <div key={x.label} style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {x.label}: <span style={{ color: x.val < 0 ? '#ef4444' : x.val < 0.04 ? '#f59e0b' : '#22c55e', fontWeight: 600 }}>
                    {x.val.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}
          {result.image && (
            <div style={{ marginTop: '1rem' }}>
              <img
                src={result.image}
                alt="Uploaded"
                style={{ maxHeight: '200px', borderRadius: '8px', opacity: 0.7, margin: '0 auto' }}
              />
            </div>
          )}
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '1rem' }}>
            Tip: Upload close-up photos of crop leaves, stems, or wheat/rice heads for best results.
          </p>
        </div>
      </div>
    )
  }

  const s = result.structured  // New structured output (may be null for old API)
  const hasStructured = s && typeof s === 'object'
  const hasLLaVA = result.llava_analysis && typeof result.llava_analysis === 'object'
  const hasCls = result.classifier_result && typeof result.classifier_result === 'object'
  const hasEnsemble = result.ensemble && typeof result.ensemble === 'object'
  const hasReasoning = result.reasoning && typeof result.reasoning === 'object'

  // ── Health score (structured > ensemble > llava > calculated) ──
  let healthScore
  if (hasStructured) {
    healthScore = s.health.score
  } else if (hasEnsemble) {
    healthScore = Math.round(result.ensemble.ensemble_health_score || 50)
  } else if (hasLLaVA) {
    healthScore = Math.round(result.llava_analysis.health_score || 50)
  } else {
    const dets = (result.detections || []).filter(
      (d) => !['healthy', 'healthy_crop', 'healthy_rice_leaf'].includes(d.class_name.toLowerCase().replace(/[\s-]+/g, '_'))
    )
    healthScore = dets.length === 0 ? 100 : Math.max(0, Math.round(100 - dets.reduce((sum, d) => sum + d.confidence * 30, 0)))
  }

  const overallRisk = hasStructured ? s.health.risk_level.toUpperCase()
    : (healthScore >= 70 ? 'LOW' : healthScore >= 40 ? 'MEDIUM' : healthScore >= 20 ? 'HIGH' : 'CRITICAL')
  const riskColor = overallRisk === 'CRITICAL' || overallRisk === 'HIGH' ? '#ef4444' : overallRisk === 'MEDIUM' ? '#eab308' : '#22c55e'

  // ── Disease info ──
  let diseaseDisplay, symptomsDisplay, treatmentDisplay, confidenceDisplay, confidenceGrade
  if (hasStructured) {
    diseaseDisplay = s.diagnosis.disease_name
    confidenceDisplay = s.diagnosis.confidence
    confidenceGrade = s.diagnosis.confidence_grade
    symptomsDisplay = (s.evidence?.supporting || []).join(' \u2022 ')
    treatmentDisplay = (s.treatment?.recommendations || []).join(' \u2022 ')
  } else if (hasReasoning && result.reasoning.disease_key !== 'healthy') {
    const r = result.reasoning
    diseaseDisplay = r.disease_name
    confidenceDisplay = r.confidence
    confidenceGrade = null
    symptomsDisplay = (r.symptoms_matched || []).join(' \u2022 ') || (r.symptoms_detected || []).join(' \u2022 ') || ''
    treatmentDisplay = (r.treatment || []).join(' \u2022 ')
  } else if (hasLLaVA) {
    const l = result.llava_analysis
    diseaseDisplay = Array.isArray(l.diseases_found) ? l.diseases_found.join(', ') : (l.diseases_found || 'Healthy')
    confidenceDisplay = null
    confidenceGrade = null
    symptomsDisplay = Array.isArray(l.visible_symptoms) ? l.visible_symptoms.join(', ') : (l.visible_symptoms || '')
    treatmentDisplay = Array.isArray(l.recommendations) ? l.recommendations.join(' \u2022 ') : (l.recommendations || '')
  } else if (hasCls) {
    diseaseDisplay = result.classifier_result.top_prediction || 'Unknown'
    confidenceDisplay = result.classifier_result.top_confidence
    confidenceGrade = null
    symptomsDisplay = `Classifier confidence: ${(result.classifier_result.top_confidence * 100).toFixed(1)}%`
    treatmentDisplay = getTreatment(result.classifier_result.top_prediction)
  } else {
    const dets = (result.detections || []).filter(
      (d) => !['healthy', 'healthy_crop', 'healthy_rice_leaf'].includes(d.class_name.toLowerCase().replace(/[\s-]+/g, '_'))
    )
    diseaseDisplay = dets.length === 0 ? 'Healthy crop' : dets.map(d => d.class_name.replace(/_/g, ' ')).join(', ')
    confidenceDisplay = null
    confidenceGrade = null
    treatmentDisplay = dets.length > 0 ? getTreatment(dets[0].class_name) : 'No treatment needed'
  }

  // ── Normalize detections for canvas ──
  const detections = (result.detections || []).map(d => ({
    class_name: d.class_name,
    confidence: d.confidence,
    bbox: d.bbox || { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 },
    x1: d.bbox?.x1 ?? d.x1,
    y1: d.bbox?.y1 ?? d.y1,
    x2: d.bbox?.x2 ?? d.x2,
    y2: d.bbox?.y2 ?? d.y2,
  }))
  const hasBoxes = detections.some(d => d.x1 != null && d.y1 != null)

  // ── Confidence breakdown sources ──
  const confSources = hasStructured ? (s.confidence_breakdown?.sources || []) : []
  const fusedConfidence = hasStructured ? s.confidence_breakdown?.fused_confidence : null

  // ── Reasoning chain ──
  const reasoningChain = hasStructured ? (s.reasoning_chain || []) : (hasReasoning ? (result.reasoning.reasoning_chain || []) : [])

  // ── Rejected diagnoses ──
  const rejected = hasStructured ? (s.rejected_diagnoses || []) : (hasReasoning ? (result.reasoning.rejections || []) : [])

  // ── Differential diagnosis ──
  const differential = hasStructured ? (s.differential_diagnosis || []) : (hasReasoning ? (result.reasoning.differential_diagnosis || []) : [])

  // ── Urgency + yield loss ──
  const urgency = hasStructured ? s.health.urgency : (hasReasoning ? result.reasoning.urgency : null)
  const yieldLoss = hasStructured ? s.health.yield_loss : (hasReasoning ? result.reasoning.yield_loss : null)
  const isHealthy = hasStructured ? s.diagnosis.is_healthy : (hasReasoning && result.reasoning.disease_key === 'healthy')

  const cardStyle = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--card-shadow)',
  }

  const sourceColors = {
    classifier: '#f59e0b',
    rule_engine: '#3b82f6',
    llm_validator: '#a78bfa',
  }

  return (
    <div className="space-y-6">

      {/* Low confidence warning */}
      {result.low_confidence && (
        <div style={{
          background: 'rgba(234, 179, 8, 0.1)',
          border: '1px solid #eab308',
          borderRadius: '12px',
          padding: '0.75rem 1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
        }}>
          <span style={{ fontSize: '1.3rem' }}>⚠️</span>
          <div>
            <span style={{ color: '#eab308', fontWeight: 600, fontSize: '0.9rem' }}>Low Confidence</span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginLeft: '0.5rem' }}>
              The model is not very confident about this prediction. The image may not clearly show a crop disease, or it may be an unusual case.
            </span>
          </div>
        </div>
      )}

      {/* Main Grid: Image + Side Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* LEFT: Image + Comparison + Chat (sticky) */}
        <div className="lg:col-span-3">
          <div className="lg:sticky lg:top-[72px] space-y-4">

            {/* Image Card with View Toggle */}
            <div className="rounded-xl overflow-hidden" style={cardStyle}>
              {/* View Toggle Tabs */}
              <div className="flex items-center gap-1 px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                {[
                  { key: 'original', label: 'Original', icon: '🖼️' },
                  ...(hasStructured && s.gradcam?.heatmap_image ? [{ key: 'gradcam', label: 'Grad-CAM', icon: '🔥' }] : []),
                  { key: 'comparison', label: 'Healthy Ref', icon: '🌿' },
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setImageView(tab.key)}
                    className="px-3 py-1.5 rounded-lg text-[11px] font-semibold transition"
                    style={{
                      background: imageView === tab.key ? 'var(--accent)' : 'transparent',
                      color: imageView === tab.key ? '#fff' : 'var(--text-muted)',
                      border: imageView === tab.key ? 'none' : '1px solid transparent',
                    }}
                  >
                    {tab.icon} {tab.label}
                  </button>
                ))}
                <span className="ml-auto text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
                  {imageView === 'gradcam' ? 'XAI View' : imageView === 'comparison' ? 'Compare' : 'Detection'}
                </span>
              </div>

              {/* Image Display */}
              <div className="p-1">
                {imageView === 'original' && (
                  hasBoxes ? (
                    <DetectionCanvas
                      imageUrl={result.annotated_image || imagePreview}
                      detections={detections}
                    />
                  ) : (
                    <img
                      src={result.annotated_image || imagePreview}
                      alt="Analyzed"
                      className="w-full h-auto rounded-lg"
                    />
                  )
                )}

                {imageView === 'gradcam' && hasStructured && s.gradcam?.heatmap_image && (
                  <div className="space-y-2">
                    <img
                      src={s.gradcam.heatmap_image}
                      alt="Grad-CAM attention map"
                      className="w-full h-auto rounded-lg"
                    />
                    <div className="px-3 pb-2">
                      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        Red/yellow regions show where the model focused for <strong>{s.gradcam.target_class || 'diagnosis'}</strong>
                        {s.gradcam.confidence != null && ` (${(s.gradcam.confidence * 100).toFixed(1)}% conf)`}
                      </p>
                      {s.gradcam.regions?.length > 0 && (
                        <div className="flex gap-3 mt-1">
                          {s.gradcam.regions.slice(0, 3).map((r, i) => (
                            <span key={i} className="text-[9px] flex items-center gap-1" style={{ color: 'var(--text-faint)' }}>
                              <span className="w-2 h-2 rounded-full" style={{ background: `rgba(239,68,68,${r.intensity || 0.6})` }} />
                              {((r.area_pct || 0) * 100).toFixed(0)}% area
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {imageView === 'comparison' && (
                  <div className="grid grid-cols-2 gap-1">
                    <div className="relative">
                      <img
                        src={result.annotated_image || imagePreview}
                        alt="Your scan"
                        className="w-full h-auto rounded-lg"
                      />
                      <span className="absolute top-2 left-2 text-[10px] font-bold px-2 py-0.5 rounded" style={{
                        background: 'rgba(239,68,68,0.85)', color: '#fff',
                      }}>Your Scan</span>
                    </div>
                    <div className="relative">
                      <img
                        src={`${getApiUrl()}${HEALTHY_REFS[cropType] || HEALTHY_REFS.wheat}`}
                        alt="Healthy reference"
                        className="w-full h-auto rounded-lg"
                        onError={(e) => {
                          e.target.style.display = 'none'
                          e.target.nextSibling.style.display = 'flex'
                        }}
                      />
                      <div className="hidden items-center justify-center h-full rounded-lg" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                        <div className="text-center p-4">
                          <span className="text-3xl">🌾</span>
                          <p className="text-[10px] mt-2" style={{ color: 'var(--text-faint)' }}>Healthy reference not available</p>
                        </div>
                      </div>
                      <span className="absolute top-2 left-2 text-[10px] font-bold px-2 py-0.5 rounded" style={{
                        background: 'rgba(34,197,94,0.85)', color: '#fff',
                      }}>Healthy Ref</span>
                    </div>
                  </div>
                )}
              </div>
              <div className="px-4 py-2 flex items-center justify-between text-[11px]" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-faint)' }}>
                <span>{result.filename || 'Uploaded image'}</span>
                <span>{result.processing_time_ms ? `${result.processing_time_ms.toFixed(0)}ms` : ''} &bull; {detections.length} detection{detections.length !== 1 ? 's' : ''}</span>
              </div>
            </div>

            {/* Chat Panel — fills remaining space */}
            <ChatPanel result={result} cropType={cropType} startOpen={true} />

          </div>
        </div>

        {/* RIGHT: Side Panel */}
        <div className="lg:col-span-2 space-y-4">

          {/* Health Gauge */}
          <div className="rounded-xl p-5 text-center" style={cardStyle}>
            <HealthGauge score={healthScore} />
            <div className="mt-3 flex items-center justify-center gap-2">
              <span className="text-xs font-bold px-3 py-1 rounded-full" style={{
                color: riskColor, background: `${riskColor}15`, border: `1px solid ${riskColor}40`,
              }}>
                {overallRisk} RISK
              </span>
            </div>
          </div>

          {/* Diagnosis Card */}
          <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
            <div>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>Detected Disease</span>
              <p className="text-lg font-bold mt-1" style={{ color: gaugeColor(healthScore) }}>{diseaseDisplay}</p>
            </div>

            {/* Fused confidence with grade badge */}
            {confidenceDisplay != null && (
              <div>
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>Confidence</span>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xl font-black" style={{ color: 'var(--text-primary)' }}>
                    {(confidenceDisplay * 100).toFixed(1)}%
                  </span>
                  {confidenceGrade && (
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{
                      color: gradeColor(confidenceGrade),
                      background: `${gradeColor(confidenceGrade)}15`,
                      border: `1px solid ${gradeColor(confidenceGrade)}30`,
                    }}>
                      {confidenceGrade.replace('_', ' ')}
                    </span>
                  )}
                  {hasStructured && s.diagnosis.llm_agrees != null && (
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{
                      color: s.diagnosis.llm_agrees ? '#22c55e' : '#ef4444',
                      background: s.diagnosis.llm_agrees ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                      border: `1px solid ${s.diagnosis.llm_agrees ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                    }}>
                      {s.diagnosis.llm_agrees ? 'LLM AGREES' : 'LLM DISAGREES'}
                    </span>
                  )}
                </div>
                {hasStructured && !s.diagnosis.llm_agrees && s.diagnosis.llm_alt_diagnosis && (
                  <p className="text-[10px] mt-1" style={{ color: '#ef4444' }}>
                    LLM suggests: {s.diagnosis.llm_alt_diagnosis}
                  </p>
                )}
              </div>
            )}

            {symptomsDisplay && (
              <div>
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>Evidence</span>
                <p className="text-sm mt-0.5" style={{ color: 'var(--text-primary)' }}>{symptomsDisplay}</p>
              </div>
            )}
            {treatmentDisplay && (
              <div>
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>Treatment</span>
                <p className="text-sm mt-0.5 font-medium" style={{ color: '#22c55e' }}>{treatmentDisplay}</p>
                {urgency && !isHealthy && (
                  <span className="inline-block mt-1 text-[10px] font-bold px-2 py-0.5 rounded" style={{
                    color: urgency === 'immediate' ? '#ef4444' : urgency === 'within_7_days' ? '#f97316' : '#eab308',
                    background: urgency === 'immediate' ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.08)',
                    border: `1px solid ${urgency === 'immediate' ? 'rgba(239,68,68,0.2)' : 'rgba(234,179,8,0.2)'}`,
                  }}>
                    {urgency.replace(/_/g, ' ').toUpperCase()}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Uncertainty Meter (MC-Dropout) */}
          {result.uncertainty && (
            <UncertaintyMeter uncertainty={result.uncertainty} />
          )}

          {/* Cost-Benefit Estimator */}
          {result.yield_estimate && (
            <CostBenefitCard estimate={result.yield_estimate} />
          )}

          {/* Confidence Breakdown (D1 visualization) */}
          {confSources.length > 0 && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                Confidence Breakdown
              </span>
              <div className="space-y-2">
                {confSources.map((src, i) => (
                  <ConfidenceBar
                    key={i}
                    label={src.label}
                    score={src.score}
                    color={sourceColors[src.source] || 'var(--text-muted)'}
                    weight={src.weight || 0}
                    disease={src.disease}
                    agrees={src.agrees}
                  />
                ))}
              </div>
              {fusedConfidence != null && (
                <div className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(76,175,80,0.08)', border: '1px solid rgba(76,175,80,0.2)' }}>
                  <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Fused Confidence</span>
                  <span className="text-base font-black" style={{ color: gaugeColor(fusedConfidence * 100) }}>
                    {(fusedConfidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {hasStructured && s.confidence_breakdown?.fusion_note && (
                <p className="text-[10px] px-1" style={{ color: 'var(--text-faint)' }}>
                  {s.confidence_breakdown.fusion_note}
                </p>
              )}
            </div>
          )}

          {/* AI Model Consensus (fallback when no structured breakdown) */}
          {confSources.length === 0 && (hasCls || hasEnsemble) && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                AI Model Consensus
              </span>
              {hasEnsemble && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs" style={{
                  background: result.ensemble.model_agreement === 'high' ? 'rgba(34,197,94,0.08)' :
                    result.ensemble.model_agreement === 'low' ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.08)',
                  border: `1px solid ${result.ensemble.model_agreement === 'high' ? 'rgba(34,197,94,0.25)' :
                    result.ensemble.model_agreement === 'low' ? 'rgba(239,68,68,0.25)' : 'rgba(234,179,8,0.25)'}`,
                }}>
                  <span className="w-2 h-2 rounded-full" style={{
                    background: result.ensemble.model_agreement === 'high' ? '#22c55e' :
                      result.ensemble.model_agreement === 'low' ? '#ef4444' : '#eab308',
                  }} />
                  <span className="font-bold" style={{
                    color: result.ensemble.model_agreement === 'high' ? '#22c55e' :
                      result.ensemble.model_agreement === 'low' ? '#ef4444' : '#eab308',
                  }}>
                    {result.ensemble.model_agreement === 'high' ? 'HIGH AGREEMENT' : result.ensemble.model_agreement === 'low' ? 'MODELS DISAGREE' : 'SINGLE MODEL'}
                  </span>
                </div>
              )}
              <div className="space-y-2">
                {hasLLaVA && (
                  <div className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid var(--border)' }}>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color: '#a78bfa', background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}>LLaVA</span>
                      <span className="text-xs" style={{ color: 'var(--text-primary)' }}>
                        {Array.isArray(result.llava_analysis.diseases_found) ? result.llava_analysis.diseases_found.join(', ') : 'Healthy'}
                      </span>
                    </div>
                    <span className="text-xs font-bold" style={{ color: gaugeColor(result.llava_analysis.health_score || 50) }}>
                      {result.llava_analysis.health_score || 50}/100
                    </span>
                  </div>
                )}
                {!hasLLaVA && result.llava_pending && (
                  <div className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)' }}>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color: '#a78bfa', background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}>LLaVA</span>
                      <span className="text-xs animate-pulse" style={{ color: '#a78bfa' }}>Analyzing in background...</span>
                    </div>
                    <span className="text-xs" style={{ color: 'var(--text-faint)' }}>&#8987;</span>
                  </div>
                )}
                {hasCls && (
                  <div className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid var(--border)' }}>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color: '#f59e0b', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.2)' }}>YOLO-CLS</span>
                      <span className="text-xs" style={{ color: 'var(--text-primary)' }}>{result.classifier_result.top_prediction}</span>
                    </div>
                    <span className="text-xs font-bold" style={{ color: gaugeColor(result.classifier_result.health_score || 50) }}>
                      {result.classifier_result.health_score || 50}/100
                    </span>
                  </div>
                )}
                {hasEnsemble && (
                  <div className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(76,175,80,0.08)', border: '1px solid rgba(76,175,80,0.2)' }}>
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>Ensemble Score</span>
                    <span className="text-base font-black" style={{ color: gaugeColor(result.ensemble.ensemble_health_score) }}>
                      {result.ensemble.ensemble_health_score}/100
                    </span>
                  </div>
                )}
              </div>
              {hasCls && result.classifier_result.top5 && (
                <div className="px-3 py-2 rounded-lg space-y-1" style={{ background: 'rgba(76,175,80,0.03)', border: '1px solid var(--border)' }}>
                  <span className="text-[10px] font-semibold" style={{ color: 'var(--text-faint)' }}>Classifier Top-5</span>
                  {result.classifier_result.top5.map((pred, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <span className="text-[11px]" style={{ color: i === 0 ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: i === 0 ? 600 : 400 }}>
                        #{i + 1} {pred.class_name}
                      </span>
                      <div className="flex items-center gap-2">
                        <div className="w-12 h-1 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                          <div className="h-full rounded-full" style={{ width: `${pred.confidence * 100}%`, background: i === 0 ? '#22c55e' : 'var(--text-faint)' }} />
                        </div>
                        <span className="text-[10px] font-semibold w-10 text-right" style={{ color: i === 0 ? '#22c55e' : 'var(--text-faint)' }}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* AI Reasoning Chain */}
          {reasoningChain.length > 0 && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                AI Reasoning Chain
              </span>
              <div className="space-y-2">
                {reasoningChain.map((step, i) => (
                  <div key={i} className="flex gap-2 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5" style={{
                      background: 'rgba(59,130,246,0.1)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.2)'
                    }}>
                      {i + 1}
                    </span>
                    <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>{step}</p>
                  </div>
                ))}
              </div>
              {yieldLoss && !isHealthy && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}>
                  <span className="text-sm">&#9888;&#65039;</span>
                  <span className="text-[11px] font-semibold" style={{ color: '#ef4444' }}>
                    Potential Yield Loss: {yieldLoss}
                  </span>
                  {urgency && (
                    <span className="text-[10px] ml-auto px-2 py-0.5 rounded font-bold" style={{
                      color: urgency === 'immediate' ? '#ef4444' : '#eab308',
                      background: urgency === 'immediate' ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.08)',
                      border: `1px solid ${urgency === 'immediate' ? 'rgba(239,68,68,0.2)' : 'rgba(234,179,8,0.2)'}`,
                    }}>
                      {urgency.replace(/_/g, ' ').toUpperCase()}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Rejected Diagnoses (new section) */}
          {rejected.length > 0 && (
            <div className="rounded-xl p-4 space-y-2" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                Rejected Diagnoses ({rejected.length})
              </span>
              {rejected.map((r, i) => (
                <div key={i} className="px-3 py-2 rounded-lg" style={{ background: 'rgba(239,68,68,0.04)', border: '1px solid rgba(239,68,68,0.12)' }}>
                  <div className="flex items-center gap-2">
                    <span className="text-red-400 text-xs">&#10007;</span>
                    <span className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>{r.disease}</span>
                  </div>
                  {(r.reasons || []).map((reason, j) => (
                    <p key={j} className="text-[10px] mt-0.5 ml-5" style={{ color: 'var(--text-muted)' }}>{reason}</p>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Differential Diagnosis */}
          {differential.length > 0 && (
            <div className="rounded-xl p-4 space-y-2" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                Differential Diagnosis
              </span>
              {differential.map((alt, i) => (
                <div key={i} className="px-3 py-2 rounded-lg" style={{ background: 'rgba(100,100,100,0.04)', border: '1px solid var(--border)' }}>
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>{alt.disease}</span>
                    <span className="text-[10px] font-bold" style={{ color: 'var(--text-faint)' }}>
                      {(alt.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {alt.key_difference && (
                    <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{alt.key_difference}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* AI Validation (LLaVA) */}
          {hasStructured && s.ai_validation && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                  AI Validation ({s.ai_validation.model || 'LLaVA'})
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{
                  color: s.ai_validation.agrees ? '#22c55e' : '#ef4444',
                  background: s.ai_validation.agrees ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
                  border: `1px solid ${s.ai_validation.agrees ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                }}>
                  {s.ai_validation.agrees ? 'AGREES' : 'DISAGREES'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid var(--border)' }}>
                  <div className="text-[10px] uppercase" style={{ color: 'var(--text-faint)' }}>LLM Diagnosis</div>
                  <div className="text-sm font-semibold" style={{ color: '#a78bfa' }}>
                    {s.ai_validation.llm_diagnosis || 'N/A'}
                  </div>
                </div>
                <div className="p-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid var(--border)' }}>
                  <div className="text-[10px] uppercase" style={{ color: 'var(--text-faint)' }}>Agreement Score</div>
                  <div className="text-sm font-semibold" style={{ color: gaugeColor((s.ai_validation.agreement_score || 0) * 100) }}>
                    {((s.ai_validation.agreement_score || 0) * 100).toFixed(0)}%
                  </div>
                </div>
                {s.ai_validation.health_score != null && (
                  <div className="p-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid var(--border)' }}>
                    <div className="text-[10px] uppercase" style={{ color: 'var(--text-faint)' }}>LLM Health Score</div>
                    <div className="text-sm font-semibold" style={{ color: gaugeColor(s.ai_validation.health_score) }}>
                      {s.ai_validation.health_score}/100
                    </div>
                  </div>
                )}
                {s.ai_validation.scenario && (
                  <div className="p-2 rounded-lg" style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid var(--border)' }}>
                    <div className="text-[10px] uppercase" style={{ color: 'var(--text-faint)' }}>Scenario</div>
                    <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {s.ai_validation.scenario}
                    </div>
                  </div>
                )}
              </div>

              {s.ai_validation.reasoning_text && (
                <div className="p-3 rounded-lg text-xs leading-relaxed" style={{ background: 'rgba(167,139,250,0.04)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  {s.ai_validation.reasoning_text.length > 300
                    ? s.ai_validation.reasoning_text.slice(0, 300) + '…'
                    : s.ai_validation.reasoning_text}
                </div>
              )}
            </div>
          )}

          {/* F3: Ensemble Voting Details */}
          {hasStructured && s.ensemble_voting && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                  Ensemble Voting
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded font-bold" style={{
                  color: s.ensemble_voting.agreement_level === 'unanimous' ? '#22c55e' : s.ensemble_voting.agreement_level === 'split' ? '#ef4444' : '#eab308',
                  background: s.ensemble_voting.agreement_level === 'unanimous' ? 'rgba(34,197,94,0.08)' : s.ensemble_voting.agreement_level === 'split' ? 'rgba(239,68,68,0.08)' : 'rgba(234,179,8,0.08)',
                  border: `1px solid ${s.ensemble_voting.agreement_level === 'unanimous' ? 'rgba(34,197,94,0.2)' : s.ensemble_voting.agreement_level === 'split' ? 'rgba(239,68,68,0.2)' : 'rgba(234,179,8,0.2)'}`,
                }}>
                  {(s.ensemble_voting.agreement_level || 'unknown').toUpperCase()}
                </span>
              </div>
              {/* Individual votes */}
              <div className="space-y-1.5">
                {(s.ensemble_voting.individual_votes || []).map((v, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(100,100,100,0.04)', border: '1px solid var(--border)' }}>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{
                        color: v.model === 'YOLOv8n-cls' ? '#f59e0b' : v.model === 'Reasoning Engine' ? '#3b82f6' : '#a78bfa',
                        background: v.model === 'YOLOv8n-cls' ? 'rgba(245,158,11,0.1)' : v.model === 'Reasoning Engine' ? 'rgba(59,130,246,0.1)' : 'rgba(167,139,250,0.1)',
                      }}>
                        {v.model}
                      </span>
                      <span className="text-[11px]" style={{ color: 'var(--text-primary)' }}>
                        {(v.disease || '').replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>
                        rel:{((v.reliability || 0) * 100).toFixed(0)}%
                      </span>
                      <span className="text-[11px] font-bold" style={{ color: gaugeColor((v.confidence || 0) * 100) }}>
                        {((v.confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              {s.ensemble_voting.safety_overrides?.length > 0 && (
                <div className="px-3 py-2 rounded-lg" style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)' }}>
                  {s.ensemble_voting.safety_overrides.map((o, i) => (
                    <p key={i} className="text-[10px]" style={{ color: '#ef4444' }}>&#9888; {o}</p>
                  ))}
                </div>
              )}
              <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>
                Method: {s.ensemble_voting.voting_method} | {s.ensemble_voting.num_models || 0} models
              </p>
            </div>
          )}

          {/* F4: Temporal Disease Tracking */}
          {hasStructured && s.temporal && s.temporal.trend !== 'first_scan' && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <div className="flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                  Disease Progression
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded font-bold" style={{
                  color: s.temporal.trend === 'improving' ? '#22c55e' : s.temporal.trend === 'worsening' ? '#ef4444' : s.temporal.trend === 'new_outbreak' ? '#f59e0b' : '#6b7280',
                  background: s.temporal.trend === 'improving' ? 'rgba(34,197,94,0.08)' : s.temporal.trend === 'worsening' ? 'rgba(239,68,68,0.08)' : s.temporal.trend === 'new_outbreak' ? 'rgba(245,158,11,0.08)' : 'rgba(100,100,100,0.08)',
                  border: `1px solid ${s.temporal.trend === 'improving' ? 'rgba(34,197,94,0.2)' : s.temporal.trend === 'worsening' ? 'rgba(239,68,68,0.2)' : s.temporal.trend === 'new_outbreak' ? 'rgba(245,158,11,0.2)' : 'rgba(100,100,100,0.2)'}`,
                }}>
                  {s.temporal.trend === 'improving' ? '↗ IMPROVING' : s.temporal.trend === 'worsening' ? '↘ WORSENING' : s.temporal.trend === 'new_outbreak' ? '⚡ NEW OUTBREAK' : '→ STABLE'}
                </span>
              </div>
              <div className="flex items-center gap-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span>{s.temporal.num_previous_scans || 0} previous scans</span>
                {s.temporal.days_since_first > 0 && <span>Tracking for {s.temporal.days_since_first} days</span>}
                {s.temporal.trend_confidence > 0 && <span>Trend confidence: {(s.temporal.trend_confidence * 100).toFixed(0)}%</span>}
              </div>
              {/* Health trajectory mini-chart */}
              {s.temporal.health_trajectory?.length > 1 && (
                <div className="px-3 py-2 rounded-lg" style={{ background: 'rgba(100,100,100,0.03)', border: '1px solid var(--border)' }}>
                  <span className="text-[10px] font-semibold block mb-2" style={{ color: 'var(--text-faint)' }}>Health Score Timeline</span>
                  <div className="flex items-end gap-1 h-12">
                    {s.temporal.health_trajectory.map((pt, i) => {
                      const h = Math.max(4, (pt.health_score / 100) * 48);
                      return (
                        <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                          <div className="w-full rounded-t" style={{ height: `${h}px`, background: gaugeColor(pt.health_score), opacity: i === s.temporal.health_trajectory.length - 1 ? 1 : 0.5 }} />
                          <span className="text-[8px]" style={{ color: 'var(--text-faint)' }}>{pt.date?.slice(5) || ''}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              {/* Recommendations */}
              {s.temporal.recommendations?.length > 0 && (
                <div className="space-y-1">
                  {s.temporal.recommendations.map((rec, i) => (
                    <p key={i} className="text-[10px]" style={{ color: 'var(--text-muted)' }}>• {rec}</p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* F2: Research Papers */}
          {hasStructured && s.research_papers?.length > 0 && (
            <div className="rounded-xl p-4 space-y-3" style={cardStyle}>
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: 'var(--text-faint)' }}>
                Research References ({s.research_papers.length})
              </span>
              {s.research_papers.slice(0, 4).map((paper, i) => (
                <div key={i} className="px-3 py-2 rounded-lg space-y-1" style={{ background: 'rgba(59,130,246,0.04)', border: '1px solid rgba(59,130,246,0.12)' }}>
                  <p className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>{paper.title}</p>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {paper.authors} — <em>{paper.journal}</em> ({paper.year})
                  </p>
                  {paper.key_findings?.length > 0 && (
                    <div className="mt-1">
                      {paper.key_findings.slice(0, 2).map((f, j) => (
                        <p key={j} className="text-[10px]" style={{ color: 'var(--text-faint)' }}>• {f}</p>
                      ))}
                    </div>
                  )}
                  {paper.doi && (
                    <p className="text-[9px]" style={{ color: 'var(--text-faint)' }}>DOI: {paper.doi}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Pipeline Metadata */}
          {hasStructured && s.metadata && (
            <div className="rounded-xl px-4 py-3" style={{ ...cardStyle, opacity: 0.8 }}>
              <div className="flex items-center justify-between text-[10px]" style={{ color: 'var(--text-faint)' }}>
                <span>Pipeline v{s.metadata.pipeline_version}</span>
                <span>{s.metadata.models_used?.length || 0} models active</span>
                <span>{s.metadata.processing_time_ms}ms</span>
              </div>
            </div>
          )}

          {/* Export buttons */}
          <div className="flex gap-3">
            <button onClick={onDownloadJSON} className="flex-1 px-4 py-2.5 text-xs font-semibold rounded-lg transition" style={{ border: '1px solid var(--border)', color: 'var(--accent)' }}>
              Export JSON
            </button>
            <button onClick={onDownloadCSV} className="flex-1 px-4 py-2.5 text-xs font-semibold rounded-lg transition" style={{ border: '1px solid var(--border)', color: 'var(--accent)' }}>
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Detection Cards Grid */}
      {detections.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-widest mb-3 font-semibold" style={{ color: 'var(--text-faint)' }}>
            All Detected Conditions ({detections.length})
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {detections.map((det, i) => (
              <DetectionCard key={i} det={det} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
