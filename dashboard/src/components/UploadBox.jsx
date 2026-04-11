import React, { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

export default function UploadBox({ onFileSelect, isLoading, useMockModel, onToggleMock }) {
  const [isDragActive, setIsDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      processFile(files[0])
    }
  }

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    setSelectedFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target.result)
    }
    reader.readAsDataURL(file)
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileInput = (e) => {
    const files = e.target.files
    if (files && files[0]) {
      processFile(files[0])
    }
  }

  const clearSelection = () => {
    setSelectedFile(null)
    setPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDetect = () => {
    if (selectedFile) {
      onFileSelect(selectedFile)
    }
  }

  return (
    <div className="space-y-4">
      {/* Toggle Mock Model */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-between glass rounded-lg p-4"
      >
        <div>
          <p className="font-semibold text-white">Model Mode</p>
          <p className="text-sm text-gray-400">
            {useMockModel ? 'Using Mock Model (Demo)' : 'Using Real Model'}
          </p>
        </div>
        <motion.button
          onClick={onToggleMock}
          className={`relative w-16 h-8 rounded-full transition-colors ${
            useMockModel ? 'bg-neon-blue/30' : 'bg-dark-tertiary'
          }`}
          whileTap={{ scale: 0.95 }}
        >
          <motion.div
            className="absolute top-1 left-1 w-6 h-6 bg-neon-blue rounded-full"
            animate={{ x: useMockModel ? 32 : 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          />
        </motion.button>
      </motion.div>

      {/* Upload Zone */}
      <motion.div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
        animate={{
          scale: isDragActive ? 1.02 : 1,
          borderColor: isDragActive ? '#0ea5e9' : 'rgba(148, 163, 184, 0.1)'
        }}
        className="glass rounded-2xl border-2 border-dashed cursor-pointer transition-all p-8 space-y-4"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />

        <motion.div
          animate={{ y: isDragActive ? -10 : 0 }}
          className="text-center"
        >
          <motion.div
            animate={{ scale: isDragActive ? 1.2 : 1 }}
            className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-neon-blue/20 to-neon-purple/20 mb-4"
          >
            <Upload
              size={32}
              className={isDragActive ? 'text-neon-blue' : 'text-neon-purple'}
            />
          </motion.div>

          <h3 className="text-lg font-semibold text-white mb-2">
            Drop your image here
          </h3>
          <p className="text-gray-400">
            or click to select from your computer
          </p>
          <p className="text-sm text-gray-500 mt-2">
            PNG, JPG, GIF up to 50MB
          </p>
        </motion.div>
      </motion.div>

      {/* Preview Section */}
      {preview && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-lg overflow-hidden"
        >
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-96 object-cover"
            />
            <motion.button
              onClick={clearSelection}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="absolute top-4 right-4 p-2 bg-red-500/80 hover:bg-red-600 rounded-lg transition-colors"
            >
              <X size={20} />
            </motion.button>
          </div>

          <div className="p-4 border-t border-dark-tertiary/50">
            <p className="text-sm text-gray-400 mb-3">
              <ImageIcon size={16} className="inline mr-2" />
              {selectedFile?.name}
            </p>

            <motion.button
              onClick={handleDetect}
              disabled={isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full py-3 px-4 bg-gradient-to-r from-neon-blue to-neon-purple hover:from-neon-blue/90 hover:to-neon-purple/90 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-all shadow-lg shadow-neon-blue/50"
            >
              {isLoading ? (
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full spinner" />
                  <span>Processing...</span>
                </div>
              ) : (
                'Run AI Detection'
              )}
            </motion.button>
          </div>
        </motion.div>
      )}
    </div>
  )
}
