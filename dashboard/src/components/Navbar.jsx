import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Zap } from 'lucide-react'
import { api } from '../services/api'

export default function Navbar({ activeSection }) {
  const [apiStatus, setApiStatus] = useState('loading')

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.getHealth()
        setApiStatus(response.status === 'ok' ? 'connected' : 'disconnected')
      } catch {
        setApiStatus('disconnected')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  const statusColor = apiStatus === 'connected' ? 'text-green-400' : 'text-red-400'
  const statusDot = apiStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass sticky top-0 z-40 border-b border-neon-blue/20"
    >
      <div className="flex items-center justify-between px-6 py-4">
        {/* Logo */}
        <motion.div
          className="flex items-center space-x-3"
          whileHover={{ scale: 1.05 }}
        >
          <div className="p-2 rounded-lg bg-gradient-to-br from-neon-blue to-neon-purple">
            <Zap size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">AgriDrone</h1>
            <p className="text-xs text-gray-400">AI Detection System</p>
          </div>
        </motion.div>

        {/* Center - Page Title */}
        <div className="hidden md:block">
          <h2 className="text-lg font-semibold text-gray-300">
            {activeSection === 'dashboard' && 'Dashboard'}
            {activeSection === 'upload' && 'Upload & Detect'}
            {activeSection === 'history' && 'Detection History'}
          </h2>
        </div>

        {/* API Status */}
        <motion.div
          className="flex items-center space-x-3 px-4 py-2 rounded-lg glass-hover cursor-pointer"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <motion.div
            className={`w-2 h-2 rounded-full ${statusDot}`}
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span className={`text-sm font-medium ${statusColor}`}>
            {apiStatus === 'connected' ? 'API Connected' : 'API Disconnected'}
          </span>
        </motion.div>
      </div>
    </motion.nav>
  )
}
