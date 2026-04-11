import React from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Target, Zap } from 'lucide-react'

export default function StatsCards({ detections, processingTime }) {
  const totalDetections = detections?.length || 0
  const avgConfidence =
    totalDetections > 0
      ? (
          (detections.reduce((sum, det) => sum + det.confidence, 0) /
            totalDetections) *
          100
        ).toFixed(1)
      : 0

  const stats = [
    {
      label: 'Total Detections',
      value: totalDetections.toString(),
      icon: BarChart3,
      color: 'from-neon-blue to-cyan-500',
      bgColor: 'bg-neon-blue/20'
    },
    {
      label: 'Avg. Confidence',
      value: `${avgConfidence}%`,
      icon: Target,
      color: 'from-neon-purple to-pink-500',
      bgColor: 'bg-neon-purple/20'
    },
    {
      label: 'Processing Time',
      value: `${(processingTime || 0).toFixed(0)}ms`,
      icon: Zap,
      color: 'from-cyan-400 to-neon-blue',
      bgColor: 'bg-cyan-500/20'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {stats.map((stat, idx) => {
        const Icon = stat.icon

        return (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            whileHover={{ y: -8 }}
            className="glass rounded-lg p-6 cursor-pointer overflow-hidden group relative"
          >
            {/* Animated background */}
            <div
              className={`absolute inset-0 bg-gradient-to-br ${stat.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}
            />

            {/* Content */}
            <div className="relative z-10">
              <div
                className={`w-12 h-12 rounded-lg ${stat.bgColor} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}
              >
                <Icon className={`text-transparent bg-gradient-to-br ${stat.color} bg-clip-text w-6 h-6`} />
              </div>

              <p className="text-gray-400 text-sm font-medium mb-2">
                {stat.label}
              </p>

              <motion.h3
                key={stat.value}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-3xl font-bold text-white"
              >
                {stat.value}
              </motion.h3>
            </div>

            {/* Bottom accent */}
            <div
              className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.color} transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left`}
            />
          </motion.div>
        )
      })}
    </div>
  )
}
