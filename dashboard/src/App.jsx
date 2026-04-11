/**
 * Main Application Component - AgriDrone Detection Dashboard
 * 
 * Orchestrates the entire detection workflow including:
 * - Image upload and processing
 * - Real-time AI inference (YOLOv8 object detection)
 * - Results visualization with canvas rendering
 * - Detection history management and export (JSON/CSV)
 * 
 * Architecture:
 * - Uses React Hooks for state management (useState, useCallback)
 * - Framer Motion for smooth section transitions and animations
 * - Centralized API layer for backend communication
 * - Component-based UI with reusable composable modules
 */

import React, { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, FileJson, FileText } from 'lucide-react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import UploadBox from './components/UploadBox'
import ResultViewer from './components/ResultViewer'
import DetectionTable from './components/DetectionTable'
import StatsCards from './components/StatsCards'
import { api } from './services/api'

/**
 * App - Root component orchestrating dashboard sections and detection workflow
 * 
 * @component
 * @returns {JSX.Element} Dashboard layout with three main sections (upload, dashboard, history)
 */
function App() {
  // ============================================================
  // STATE MANAGEMENT
  // ============================================================

  /** @type {[string, Function]} Currently active dashboard section: 'upload' | 'dashboard' | 'history' */
  const [activeSection, setActiveSection] = useState('upload')

  /** @type {[Array|null, Function]} Array of detection objects from latest inference */
  const [detections, setDetections] = useState(null)

  /** @type {[string|null, Function]} Base64-encoded image returned from API after processing */
  const [originalImage, setOriginalImage] = useState(null)

  /** @type {[boolean, Function]} Loading state during API request */
  const [isLoading, setIsLoading] = useState(false)

  /** @type {[number, Function]} End-to-end processing time in milliseconds */
  const [processingTime, setProcessingTime] = useState(0)

  /** @type {[string|null, Function]} Error message from API or processing */
  const [error, setError] = useState(null)

  /** @type {[boolean, Function]} Toggle between mock demo mode and real YOLOv8 model */
  const [useMockModel, setUseMockModel] = useState(true)

  /** @type {[Array, Function]} Recent detection sessions (stores last 10 uploads) */
  const [detectionHistory, setDetectionHistory] = useState([])

  // ============================================================
  // EVENT HANDLERS
  // ============================================================

  /**
   * Handles image file selection from UploadBox component
   * 
   * Workflow:
   * 1. Send image to backend API for YOLOv8 detection
   * 2. Measure processing time
   * 3. Store detections and image for visualization
   * 4. Add entry to detection history (max 10 sessions)
   * 5. Auto-switch to dashboard view
   * 6. Handle errors gracefully
   * 
   * @async
   * @param {File} file - Image file from drag/drop or file input
   * @returns {Promise<void>}
   */
  const handleFileSelect = useCallback(async (file) => {
    setIsLoading(true)
    setError(null)

    try {
      // Measure API round-trip time
      const startTime = performance.now()
      const response = await api.detectObjects(file, useMockModel)
      const endTime = performance.now()

      // Store results
      setProcessingTime(endTime - startTime)
      setDetections(response.detections || [])
      setOriginalImage(response.image || response.image_base64)

      // Track in history - keep only last 10 for performance
      setDetectionHistory((prev) => [
        {
          id: Date.now(),
          fileName: file.name,
          detectionCount: response.detections?.length || 0,
          timestamp: new Date().toLocaleString(),
          detections: response.detections
        },
        ...prev.slice(0, 9)
      ])

      // Auto-navigate to results view
      setActiveSection('dashboard')
    } catch (err) {
      setError(err.message || 'Failed to process image')
      console.error('Detection error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [useMockModel])

  /**
   * Export detection results as JSON file
   * 
   * Output format:
   * {
   *   timestamp: ISO 8601 string,
   *   processingTime: milliseconds,
   *   detections: [{class_name, confidence, x1, y1, x2, y2, ...}],
   *   fileName: original image filename
   * }
   * 
   * @returns {void} Triggers browser download
   */
  const downloadJSON = () => {
    const data = JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        processingTime: processingTime,
        detections: detections,
        fileName: originalImage
      },
      null,
      2
    )

    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detections-${Date.now()}.json`
    a.click()
  }

  /**
   * Export detection results as CSV file
   * 
   * Output format:
   * Class,Confidence,X1,Y1,X2,Y2,Area
   * weed,0.9523,100,200,300,400,40000
   * disease,0.8721,450,300,600,500,30000
   * 
   * Useful for spreadsheet analysis and GIS workflows
   * 
   * @returns {void} Triggers browser download
   */
  const downloadCSV = () => {
    // CSV header with standard bounding box columns
    let csv = 'Class,Confidence,X1,Y1,X2,Y2,Area\n'

    // Add detection rows with calculated area (width × height)
    detections.forEach((det) => {
      const area = (det.x2 - det.x1) * (det.y2 - det.y1)
      csv += `${det.class_name},${det.confidence.toFixed(4)},${det.x1},${det.y1},${det.x2},${det.y2},${area}\n`
    })

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detections-${Date.now()}.csv`
    a.click()
  }

  return (
    <div className="flex h-screen overflow-hidden bg-dark">
      {/* ============================================================
          SIDEBAR - Navigation menu with section selection
          ============================================================ */}
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />

      {/* ============================================================
          MAIN CONTENT AREA - Scrollable layout with three sections
          ============================================================ */}
      <main className="flex-1 lg:ml-64 flex flex-col overflow-hidden">
        {/* Top navigation bar with branding and API status indicator */}
        <Navbar activeSection={activeSection} />

        {/* Scrollable content area - switches between sections */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 max-w-7xl mx-auto">
            <AnimatePresence mode="wait">
              {/* ============================================================
                  DASHBOARD SECTION - Results view with visualizations
                  ============================================================ */}
              {activeSection === 'dashboard' && (
                <motion.div
                  key="dashboard"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Section header */}
                  <div>
                    <h2 className="text-3xl font-bold text-white mb-2">
                      Detection Dashboard
                    </h2>
                    <p className="text-gray-400">
                      Real-time AI-powered field analysis and hotspot detection
                    </p>
                  </div>

                  {/* Error banner with styled border */}
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="glass border-l-4 border-red-500 rounded-lg p-4 text-red-200"
                    >
                      <p className="font-semibold">Error</p>
                      <p className="text-sm">{error}</p>
                    </motion.div>
                  )}

                  {/* Render detection results if available */}
                  {detections && (
                    <>
                      {/* Key metrics: total detections, avg confidence, processing time */}
                      <StatsCards
                        detections={detections}
                        processingTime={processingTime}
                      />

                      {/* Canvas visualization with bounding boxes and overlays */}
                      <ResultViewer
                        detections={detections}
                        originalImage={originalImage}
                        processingTime={processingTime}
                      />

                      {/* Export buttons - only show if detections exist */}
                      {detections.length > 0 && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="flex gap-4"
                        >
                          {/* JSON export with gradient button */}
                          <motion.button
                            onClick={downloadJSON}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-neon-blue to-neon-purple hover:from-neon-blue/90 hover:to-neon-purple/90 rounded-lg font-semibold transition-all shadow-lg shadow-neon-blue/50"
                          >
                            <FileJson size={20} />
                            <span>Download JSON</span>
                          </motion.button>

                          {/* CSV export - useful for spreadsheet analysis */}
                          <motion.button
                            onClick={downloadCSV}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-neon-cyan to-neon-blue hover:from-neon-cyan/90 hover:to-neon-blue/90 rounded-lg font-semibold transition-all shadow-lg shadow-neon-cyan/50"
                          >
                            <FileText size={20} />
                            <span>Download CSV</span>
                          </motion.button>
                        </motion.div>
                      )}

                      {/* Interactive sortable table of all detections */}
                      <DetectionTable detections={detections} />
                    </>
                  )}

                  {/* Empty state - guide user if no results yet */}
                  {!detections && !isLoading && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center py-12"
                    >
                      <p className="text-gray-400">
                        Upload an image to see detection results here
                      </p>
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* ============================================================
                  UPLOAD SECTION - File input with model mode toggle
                  ============================================================ */}
              {activeSection === 'upload' && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Section header */}
                  <div>
                    <h2 className="text-3xl font-bold text-white mb-2">
                      Upload & Detect
                    </h2>
                    <p className="text-gray-400">
                      Upload an aerial or field image to detect hotspots
                    </p>
                  </div>

                  {/* Drag-drop upload box with model toggle */}
                  <UploadBox
                    onFileSelect={handleFileSelect}
                    isLoading={isLoading}
                    useMockModel={useMockModel}
                    onToggleMock={() => setUseMockModel(!useMockModel)}
                  />

                  {/* Error banner on this section too */}
                  {error && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="glass border-l-4 border-red-500 rounded-lg p-4 text-red-200"
                    >
                      <p className="font-semibold">Error</p>
                      <p className="text-sm">{error}</p>
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* ============================================================
                  HISTORY SECTION - Recent detection sessions
                  ============================================================ */}
              {activeSection === 'history' && (
                <motion.div
                  key="history"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Section header */}
                  <div>
                    <h2 className="text-3xl font-bold text-white mb-2">
                      Detection History
                    </h2>
                    <p className="text-gray-400">
                      View your recent detection sessions
                    </p>
                  </div>

                  {/* History list - clickable cards to re-view results */}
                  {detectionHistory.length > 0 ? (
                    <div className="grid gap-4">
                      {detectionHistory.map((entry, idx) => (
                        <motion.div
                          key={entry.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.05 }}
                          onClick={() => {
                            setDetections(entry.detections)
                            setActiveSection('dashboard')
                          }}
                          className="glass rounded-lg p-6 cursor-pointer hover:bg-dark-tertiary/30 transition-all"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="text-lg font-semibold text-white">
                                {entry.fileName}
                              </h3>
                              <p className="text-sm text-gray-400 mt-1">
                                {entry.detectionCount} detection
                                {entry.detectionCount !== 1 ? 's' : ''} •{' '}
                                {entry.timestamp}
                              </p>
                            </div>
                            {/* Count badge */}
                            <div className="px-4 py-2 bg-neon-blue/20 rounded-lg">
                              <p className="text-neon-blue font-semibold">
                                {entry.detectionCount}
                              </p>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center py-12"
                    >
                      <p className="text-gray-400">
                        No detection history yet. Upload an image to get started.
                      </p>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
