import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { AlertCircle, CheckCircle } from 'lucide-react'

export default function ResultViewer({ detections, originalImage, processingTime }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !originalImage || !detections) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()

    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width
      canvas.height = img.height

      // Draw image
      ctx.drawImage(img, 0, 0)

      // Colors for different classes
      const colors = {
        'weed': '#ff6b6b',
        'disease': '#ffa500',
        'pest': '#ff1493',
        'anomaly': '#00d4ff',
        'default': '#0ea5e9'
      }

      // Draw bounding boxes
      detections.forEach((det, idx) => {
        const color = colors[det.class_name] || colors.default

        // Bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1)

        // Background for label
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`
        ctx.font = 'bold 14px Arial'
        const textWidth = ctx.measureText(label).width
        const textX = det.x1
        const textY = det.y1 - 10

        ctx.fillStyle = color
        ctx.fillRect(textX - 2, textY - 20, textWidth + 8, 25)

        // Text
        ctx.fillStyle = '#fff'
        ctx.fillText(label, textX + 2, textY)
      })
    }

    img.src = originalImage
  }, [originalImage, detections])

  const totalDetections = detections?.length || 0
  const hasDetections = totalDetections > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-lg overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-dark-tertiary/50">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
            {hasDetections ? (
              <>
                <CheckCircle size={20} className="text-green-400" />
                <span>Detection Results</span>
              </>
            ) : (
              <>
                <AlertCircle size={20} className="text-yellow-400" />
                <span>No Detections Found</span>
              </>
            )}
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Found {totalDetections} hotspot{totalDetections !== 1 ? 's' : ''} •{' '}
            Processed in {processingTime?.toFixed(2) || 0}ms
          </p>
        </div>
      </div>

      {/* Canvas */}
      <div className="overflow-auto bg-black/20 max-h-[600px]">
        <canvas
          ref={canvasRef}
          className="w-full h-auto"
          style={{ maxHeight: '600px', objectFit: 'contain' }}
        />
      </div>

      {/* No Detections Message */}
      {!hasDetections && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-12 text-center"
        >
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="inline-block"
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 flex items-center justify-center mx-auto mb-4">
              <CheckCircle size={32} className="text-neon-blue" />
            </div>
          </motion.div>
          <p className="text-gray-300 text-lg font-medium">
            No problematic areas detected
          </p>
          <p className="text-gray-500 text-sm mt-2">
            The field looks healthy. No intervention needed.
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}
