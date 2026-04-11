import React from 'react'
import { motion } from 'framer-motion'
import { LayoutDashboard, Upload, History, Settings, LogOut } from 'lucide-react'

const menuItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'upload', label: 'Upload & Detect', icon: Upload },
  { id: 'history', label: 'History', icon: History },
  { id: 'settings', label: 'Settings', icon: Settings }
]

export default function Sidebar({ activeSection, onSectionChange }) {
  return (
    <motion.aside
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="hidden lg:block fixed left-0 top-0 h-screen w-64 glass border-r border-neon-blue/20 pt-20"
    >
      <div className="p-6 space-y-8">
        {/* Navigation */}
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = activeSection === item.id

            return (
              <motion.button
                key={item.id}
                onClick={() => onSectionChange(item.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-neon-blue/20 to-neon-purple/20 border border-neon-blue/50'
                    : 'hover:bg-dark-tertiary/50'
                }`}
                whileHover={{ x: 5 }}
                whileTap={{ scale: 0.95 }}
              >
                <Icon
                  size={20}
                  className={isActive ? 'text-neon-blue' : 'text-gray-400'}
                />
                <span
                  className={`font-medium ${
                    isActive ? 'text-neon-blue' : 'text-gray-300'
                  }`}
                >
                  {item.label}
                </span>
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-neon-blue"
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
              </motion.button>
            )
          })}
        </nav>

        {/* Divider */}
        <div className="h-px bg-gradient-to-r from-transparent via-neon-blue/20 to-transparent" />

        {/* Footer */}
        <div className="space-y-3 border-t border-dark-tertiary/50 pt-6">
          <motion.button
            className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-dark-tertiary/50 transition-all"
            whileHover={{ x: 5 }}
            whileTap={{ scale: 0.95 }}
          >
            <LogOut size={20} />
            <span className="font-medium">Logout</span>
          </motion.button>
          <p className="text-xs text-gray-500 px-4">
            AgriDrone v1.0.0 • Research Prototype
          </p>
        </div>
      </div>
    </motion.aside>
  )
}
