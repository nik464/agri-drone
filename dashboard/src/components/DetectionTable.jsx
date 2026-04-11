import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown, ChevronUp } from 'lucide-react'

export default function DetectionTable({ detections }) {
  const [sortConfig, setSortConfig] = useState({
    key: 'confidence',
    direction: 'desc'
  })

  const getClassColor = (className) => {
    const colorMap = {
      'weed': 'bg-red-500/20 text-red-300',
      'disease': 'bg-orange-500/20 text-orange-300',
      'pest': 'bg-pink-500/20 text-pink-300',
      'anomaly': 'bg-cyan-500/20 text-cyan-300',
      'default': 'bg-blue-500/20 text-blue-300'
    }
    return colorMap[className] || colorMap.default
  }

  const sortedDetections = [...(detections || [])].sort((a, b) => {
    const aVal = a[sortConfig.key]
    const bVal = b[sortConfig.key]

    if (sortConfig.direction === 'asc') {
      return aVal - bVal
    }
    return bVal - aVal
  })

  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }))
  }

  if (!detections || detections.length === 0) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-lg overflow-hidden"
    >
      <div className="overflow-x-auto">
        <table className="w-full">
          {/* Header */}
          <thead>
            <tr className="border-b border-dark-tertiary/50">
              <th className="px-6 py-4 text-left">
                <button
                  onClick={() => handleSort('class_name')}
                  className="flex items-center space-x-2 text-gray-400 hover:text-neon-blue transition-colors font-semibold"
                >
                  <span>Class Name</span>
                  {sortConfig.key === 'class_name' && (
                    sortConfig.direction === 'desc' ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronUp size={16} />
                    )
                  )}
                </button>
              </th>
              <th className="px-6 py-4 text-center">
                <button
                  onClick={() => handleSort('confidence')}
                  className="flex items-center justify-center space-x-2 text-gray-400 hover:text-neon-blue transition-colors font-semibold w-full"
                >
                  <span>Confidence</span>
                  {sortConfig.key === 'confidence' && (
                    sortConfig.direction === 'desc' ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronUp size={16} />
                    )
                  )}
                </button>
              </th>
              <th className="px-6 py-4 text-center">
                <button
                  onClick={() => handleSort('x1')}
                  className="flex items-center justify-center space-x-2 text-gray-400 hover:text-neon-blue transition-colors font-semibold w-full"
                >
                  <span>Bounding Box</span>
                  {sortConfig.key === 'x1' && (
                    sortConfig.direction === 'desc' ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronUp size={16} />
                    )
                  )}
                </button>
              </th>
              <th className="px-6 py-4 text-center font-semibold text-gray-400">
                Area
              </th>
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {sortedDetections.map((det, idx) => {
              const width = det.x2 - det.x1
              const height = det.y2 - det.y1
              const area = (width * height).toFixed(0)

              return (
                <motion.tr
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="border-b border-dark-tertiary/50 hover:bg-dark-tertiary/30 transition-colors"
                >
                  <td className="px-6 py-4">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-semibold ${getClassColor(
                        det.class_name
                      )}`}
                    >
                      {det.class_name}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <div className="flex items-center justify-center">
                      <div className="w-24 bg-dark-tertiary rounded-full h-2 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{
                            width: `${det.confidence * 100}%`
                          }}
                          transition={{ delay: idx * 0.05 + 0.3, duration: 0.8 }}
                          className="h-full bg-gradient-to-r from-neon-blue to-neon-purple"
                        />
                      </div>
                      <span className="ml-3 text-white font-semibold min-w-fit">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-center text-gray-300 font-mono text-sm">
                    ({det.x1}, {det.y1}) → ({det.x2}, {det.y2})
                  </td>
                  <td className="px-6 py-4 text-center text-gray-300 font-mono">
                    {area} px²
                  </td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  )
}
